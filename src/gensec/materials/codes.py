# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
r"""
Normative codes as first-class, declared objects.

The principle this module implements
------------------------------------
**GenSec is not tied to any code.**  There is a generic container ---
fibre integration, sectional equilibrium, ageing viscoelasticity --- and
everything normative lives behind an interface.  EN 1992-1-1 and
NTC 2018 are the *shipped defaults*, not the mechanism.

What was missing until now was the **declaration**.  The container was
agnostic and the providers were replaceable, but nothing in the input
file said *which code this model is verified to*, so each derived
quantity resolved its own convention locally --- and where a material
could derive a value from what it happened to carry, it did, silently
applying the Eurocode to a model that never named one.  Measured at
:math:`f_{ck} = 25`, that silent choice is **33.9 %** away from
ACI 318-19.

So the rule is now the one the project always stated:

    **The input declares the code, and everything is derived from it.
    Or the input declares that the material is custom, and then GenSec
    derives nothing.**

There is no third option, and in particular there is no *implicit*
option.

How a code is resolved
----------------------
In order, first hit wins:

1. ``code:`` on the material itself --- one material under another code
   is a legitimate model (a precast beam under EC2 with an ACI topping);
2. ``code:`` at the top level of the input file --- the ordinary case,
   declared once for the whole model.

There is no third source and **no default beneath these two**.  A
material with neither is :data:`UNDECLARED`, and
:func:`gensec.io_yaml.load_yaml` refuses to build a model that contains
one, naming the material.

The material **type** is deliberately not read as a declaration; see
:func:`resolve_code` for the two measured reasons.

:data:`CUSTOM` is the **explicit** opt-out and is not the same thing as
:data:`UNDECLARED`.  Both decline to derive; they differ in what the
refusal means and therefore in what it tells the caller to do.  Custom
says *you told me you would supply this yourself, and you did not*;
undeclared says *you never told me which code to read this under*.
Collapsing them would lose the only information the message carries.

What a code supplies
--------------------
A :class:`NormativeCode` is a small object, not a plugin system.  It
answers for the quantities that are genuinely code-dependent and
declines for the rest:

* :meth:`~NormativeCode.reference_modulus` --- the conventional secant
  modulus of a concrete, with its reference age and its citation;
* :meth:`~NormativeCode.rheological_model` --- the creep, shrinkage and
  relaxation provider, i.e. the four material functions of the ageing
  viscoelastic container.

Adding a code is adding a subclass and one registry line.  Nothing in
the resolver, the solver or the YAML loader changes, which is the
property that makes the claim of agnosticism checkable rather than
aspirational.

Examples
--------
Declared once, for the whole model::

    code: aci318-19
    materials:
      c1: {type: concrete_ec2_gen1_custom, fck: 28.0}

Per material, overriding the model::

    code: ec2
    materials:
      topping: {type: concrete_ec2_gen1_custom, fck: 28.0, code: aci318-19}

Explicitly custom --- GenSec derives nothing and the input supplies
everything::

    materials:
      resin: {type: tabulated, code: custom, strains: [...], stresses: [...]}
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Optional

from typing import NamedTuple

from .base import NormativeChoiceNotDeclared, ReferenceModulus

__all__ = [
    "NormativeCode",
    "EC2Code",
    "NTC18Code",
    "ACI318Code",
    "CustomCode",
    "UndeclaredCode",
    "CUSTOM",
    "UNDECLARED",
    "register_code",
    "resolve_code",
    "known_codes",
    "CodeReferenceModulus",
    "SectionalState",
    "DesignReduction",
    "CapacityFactor",
]


def _pos(value):
    """
    *value* as a positive float, or ``None``.

    Parameters
    ----------
    value : object

    Returns
    -------
    float or None
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


class SectionalState(NamedTuple):
    r"""
    The solved state a code needs in order to factor a capacity.

    Everything here is an **outcome of the solve**, which is the whole
    reason this type exists.  A code whose reduction happens on the
    materials never looks at it; a code whose reduction happens on the
    resistance cannot be applied without it, because the factor is a
    function of the strain distribution the section reached.

    Attributes
    ----------
    eps_t : float or None
        Net tensile strain in the extreme tension steel or strand,
        excluding any locked-in prestrain.  ``None`` when the section
        carries no tension element, in which case a strain-controlled
        factor cannot be evaluated and the code must say so rather
        than assume a branch.
    eps_c : float or None
        Compressive strain at the extreme compression fibre, signed
        negative.
    eps_yield : float or None
        Yield strain of the element ``eps_t`` was read at, so that a
        rule expressed against :math:`\varepsilon_{ty}` --- as
        ACI 318-19 §21.2.2 is since the 2019 edition --- does not have
        to re-derive it from a grade it cannot see.
    confinement : str
        ``'tied'``, ``'spiral'``, or ``'none'``.  Some codes read a
        different factor for spirally confined members.
    action : str
        Which resistance is being factored: ``'flexure'``,
        ``'axial'``, ``'shear'``, ``'torsion'``, ``'bearing'``.  One
        section can need several factors at once, so the action is
        part of the question and not a property of the section.

    Notes
    -----
    ``eps_t`` is the *net* strain because that is what strain-based
    factor rules are written against.  For a prestressed element the
    total strain includes ``eps_pe``, which is a locked-in state and
    not a measure of how far the section is from a brittle failure.
    Passing the total would move every prestressed section into the
    tension-controlled branch and never trip the transition.
    """

    eps_t: _Optional[float] = None
    eps_c: _Optional[float] = None
    eps_yield: _Optional[float] = None
    confinement: str = "none"
    action: str = "flexure"


class DesignReduction(NamedTuple):
    r"""
    Where a code turns characteristic values into design values.

    The two shipped families do it at opposite ends of the
    computation, and the difference is not cosmetic:

    * **at the material**, before integration.  EN 1992-1-1 divides
      :math:`f_{ck}` by :math:`\gamma_c` and :math:`f_{yk}` by
      :math:`\gamma_s`, and the section is integrated with laws that
      are already reduced;
    * **at the capacity**, after integration.  ACI 318-19 integrates
      with the specified :math:`f'_c` and :math:`f_y` and multiplies
      the nominal strength by :math:`\phi`.

    Attributes
    ----------
    where : str
        ``'material'`` or ``'capacity'``.
    gamma_c, gamma_s, gamma_p : float or None
        Partial factors, when ``where == 'material'``.  ``None``
        otherwise --- **not** 1.0, because "this code does not reduce
        materials" and "this code reduces materials by unity" are
        different statements and only one of them is true.
    source : str
        The clause.
    """

    where: str
    gamma_c: _Optional[float] = None
    gamma_s: _Optional[float] = None
    gamma_p: _Optional[float] = None
    source: str = ""


class CapacityFactor(NamedTuple):
    r"""
    A multiplier applied to a nominal resistance, and its provenance.

    Attributes
    ----------
    value : float
        The multiplier.  ``1.0`` from a code that reduced at the
        material end, and that ``1.0`` is a *statement* -- the
        reduction already happened -- not an absence.
    source : str
        The clause.
    governed_by : str
        What decided the value: ``'already reduced at the material'``,
        ``'tension-controlled'``, ``'transition'``, and so on.  A bare
        number cannot be checked by a reader; a number with the branch
        that produced it can.
    """

    value: float
    source: str = ""
    governed_by: str = ""


class NormativeCode(ABC):
    r"""
    A structural code, as an object the model can name.

    Attributes
    ----------
    key : str
        The token used in YAML, lower-case.
    title : str
        Human-readable name, used in refusals and reports.
    """

    key = ""
    title = ""

    def __repr__(self):
        return f"<{type(self).__name__} {self.key!r}>"

    @abstractmethod
    def reference_modulus(self, material):
        r"""
        The conventional secant modulus of *material*.

        Parameters
        ----------
        material : Material

        Returns
        -------
        ReferenceModulus

        Raises
        ------
        NormativeChoiceNotDeclared
            If this code cannot state one for this material --- because
            the material lacks the input the code needs, or because the
            code is :data:`CUSTOM` or :data:`UNDECLARED`.
        """

    def design_reduction(self, material):
        r"""
        Where and how this code turns characteristic into design.

        Parameters
        ----------
        material : Material

        Returns
        -------
        DesignReduction

        Raises
        ------
        NormativeChoiceNotDeclared
            When this code states no reduction for this material.
        """
        raise NormativeChoiceNotDeclared(
            f"{self.title} states no design reduction for a "
            f"{type(material).__name__}.",
            subject="design reduction",
            material=type(material).__name__)

    def capacity_factor(self, state):
        r"""
        The multiplier this code applies to a **nominal** resistance.

        The counterpart of :meth:`design_reduction`, at the other end
        of the computation.  Exactly one of the two carries the safety
        format of a given code, and which one is a property of the
        code and not of GenSec:

        * a code that reduces at the material returns ``1.0`` here,
          and that ``1.0`` says *the reduction already happened*;
        * a code that reduces at the capacity returns its factor here
          and declares no material reduction.

        Parameters
        ----------
        state : SectionalState
            The solved state.  A factor that depends on the strain
            distribution cannot be evaluated before the solve, which
            is why this takes a state and :meth:`design_reduction`
            takes a material.

        Returns
        -------
        CapacityFactor

        Raises
        ------
        NormativeChoiceNotDeclared
            When this code's factor is not implemented in this build.
            Declining is the honest answer and never ``1.0``:
            returning unity for an unimplemented factor would publish
            an **unfactored** resistance as if it were a design value,
            which is the one error in this file that would be
            unconservative rather than merely wrong.
        """
        raise NormativeChoiceNotDeclared(
            f"{self.title} states no capacity factor in this build.",
            subject="capacity factor", material="section")

    def rheological_model(self, material, **kwargs):
        r"""
        The creep/shrinkage/relaxation provider of this code.

        Parameters
        ----------
        material : Material
        **kwargs
            Passed through to the provider's constructor.

        Returns
        -------
        RheologicalModel

        Raises
        ------
        NormativeChoiceNotDeclared
            When this code ships no provider in this build.  Declining
            is the honest answer: an empty slot is documented as empty
            in ``7_0-GENSEC_PHASE5_PRIMER.md §13``, and substituting
            another code's provider for a missing one would be the
            silent choice this whole module exists to remove.
        """
        raise NormativeChoiceNotDeclared(
            f"{self.title} ships no rheological provider in this "
            f"build, so GenSec has no creep, shrinkage or relaxation "
            f"law to apply under it. Supply one explicitly "
            f"(ConstructionTimeline(rheology=...)), or declare the "
            f"model under a code that ships one "
            f"({', '.join(k for k in known_codes() if k not in ('custom', 'undeclared'))}).",
            subject="rheological model", material=type(material).__name__)


    # ------------------------------------------------------------------
    #  Characteristic -> design, in two hooks rather than one.
    #
    #  Codes do not agree on WHERE the reduction happens, and until
    #  Phase 22 GenSec assumed they did.  EN 1992-1-1 divides the
    #  material strengths and integrates the reduced laws, so the
    #  design resistance falls out of the integration.  ACI 318-19
    #  integrates the specified strengths and multiplies the resulting
    #  NOMINAL resistance by phi.  Those are different points in the
    #  chain, and collapsing them into one is the shape of the
    #  Eurocode wearing the costume of generality.
    #
    #  So there are two hooks.  Each code implements both, and each is
    #  the identity in the other's half:
    #
    #      code            design_material      capacity_factor
    #      EN 1992-1-1     f_ck/gamma_c         1.0
    #      ACI 318-19      identity             phi(eps_t, action)
    #
    #  Why not one hook with a material factor.  Because pushing phi
    #  inside as a scale on the strengths is exact only where the
    #  resistance is homogeneous of degree 1 in them, and it is not.
    #  Measured, scaling both f'c and f_y by lambda:
    #
    #      singly reinforced RC        exact  (beta_1 never enters M_n)
    #      prestressed, f_ps capped    exact
    #      prestressed, f_ps STRAIN-   +16.68 % at lambda = 0.65
    #        governed
    #      doubly reinforced           +0.15 %
    #
    #  It breaks wherever a STRAIN governs instead of a strength: a
    #  locked-in eps_pe does not scale, and neither does eps_cu.  That
    #  it is exact on the singly reinforced case is what makes it a
    #  trap: it is green on the first thing anyone would test.
    #
    #  And phi is not a constant anyway.  ACI 318-19 21.2.2 defines it
    #  on eps_t, which is an OUTPUT of the solve, so a factor fixed
    #  before integration cannot express it without circularity.
    # ------------------------------------------------------------------

    def axial_cap(self, P_o, confinement="tied"):
        r"""
        The cap this code puts on the axial branch, or ``None``.

        Separate from :meth:`capacity_factor` because a cap is a
        **truncation**, not a scaling, and a scalar multiplier cannot
        express one.  ACI 318-19 §22.4.2.1 limits the compression
        branch to :math:`0.80 P_o` for tied members and
        :math:`0.85 P_o` for spirally reinforced ones, which cuts the
        top of the interaction domain flat.  A caller that only ever
        multiplied by :meth:`capacity_factor` would produce a domain
        that is right in the tension-controlled region and too tall in
        the other --- unconservative exactly where it matters.

        Parameters
        ----------
        P_o : float
            Nominal concentric axial resistance [N], compression
            negative in GenSec's sign convention.
        confinement : str, optional
            ``'tied'``, ``'spiral'`` or ``'none'``.

        Returns
        -------
        float or None
            The capped resistance, or ``None`` when this code states
            no cap.  ``None`` and "a cap equal to :math:`P_o`" are
            different statements and only one of them is true of a
            given code.
        """
        del P_o, confinement
        return None

    def capacity_envelope(self, N, Mx, My, states):
        r"""
        Factor a whole interaction domain, point by point.

        Separate from :meth:`capacity_factor` because a domain is not
        always a scaled copy of itself: a code may apply a different
        factor along the curve, because the factor is a function of
        the strain state and the strain state varies along it.

        Parameters
        ----------
        N, Mx, My : array_like
            Nominal resistance points [N, N.mm].
        states : sequence of SectionalState
            One per point, in the same order.

        Returns
        -------
        N_d, Mx_d, My_d : numpy.ndarray
            The factored domain.
        factors : numpy.ndarray
            The scalar applied at each point, returned so that a
            report can show *where* the factor changed rather than
            only the curve it produced.

        Raises
        ------
        ValueError
            If the number of states does not match the number of
            points.  Reusing one state across points that do not share
            it is the one mistake this signature exists to prevent: a
            factor defined on the solved strain is meaningless
            anywhere but at the point that produced it.

        Notes
        -----
        The base implementation applies :meth:`capacity_factor` point
        by point, which is exactly right for every code that only
        scales, and it does **not** apply :meth:`axial_cap` --- a
        truncation changes which points exist, not what they are worth,
        so it belongs to whoever assembles the domain and not to
        whoever factors it.
        """
        import numpy as np
        N = np.asarray(N, dtype=float)
        Mx = np.asarray(Mx, dtype=float)
        My = np.asarray(My, dtype=float)
        if len(states) != N.size:
            raise ValueError(
                f"capacity_envelope: {N.size} point(s) but "
                f"{len(states)} state(s). One state per point: a "
                f"factor defined on the solved strain cannot be "
                f"reused across points that do not share it.")
        phi = np.array([float(self.capacity_factor(s).value)
                        for s in states], dtype=float)
        return N * phi, Mx * phi, My * phi, phi

    @staticmethod
    def _fck_of(material):
        r"""
        The characteristic cylinder strength of *material*, or ``None``.

        Parameters
        ----------
        material : Material

        Returns
        -------
        float or None
        """
        fck = getattr(material, "fck", None)
        if fck is None:
            return None
        try:
            fck = float(fck)
        except (TypeError, ValueError):
            return None
        return fck if fck > 0.0 else None


class EC2Code(NormativeCode):
    r"""
    EN 1992-1-1:2004.

    The conventional secant modulus is Table 3.1:

    .. math::

        E_{cm} = 22\,000\left(\frac{f_{cm}}{10}\right)^{0.3}
        \qquad [\mathrm{MPa}],\; f_{cm} = f_{ck} + 8

    Read from the material's own EC2 property object where it carries
    one, so that the number a datum is drawn on and the number the
    material's other clauses use are the same object's answer, and
    derived through the same provider otherwise --- one formula, one
    home.  At the 28-day reference age :math:`\beta_{cc} = 1`, so the
    cement class drops out and choosing one here decides nothing.
    """

    key = "ec2"
    title = "EN 1992-1-1:2004"
    _GAMMA_SOURCE = ("EN 1992-1-1:2004 §2.4.2.4 -- partial factors for "
                     "materials at the ultimate limit state")
    _SOURCE = ("EN 1992-1-1:2004 §3.1.3 Table 3.1 -- E_cm, secant "
               "modulus at the 28-day reference age")

    def reference_modulus(self, material):
        r"""
        :math:`E_{cm}` at 28 days.  See :class:`EC2Code`.

        Parameters
        ----------
        material : Material

        Returns
        -------
        ReferenceModulus

        Raises
        ------
        NormativeChoiceNotDeclared
            If the material states neither an EC2 property object nor a
            positive ``fck``.
        """
        ecm = getattr(getattr(material, "ec2", None), "ecm", None)
        if ecm is None or not float(ecm) > 0.0:
            fck = self._fck_of(material)
            if fck is None:
                raise NormativeChoiceNotDeclared(
                    f"{self.title}: this material states neither an "
                    f"EC2 property object nor a positive f_ck, and "
                    f"Table 3.1 needs f_ck. Declare f_ck, or supply "
                    f"the modulus directly.",
                    material=type(material).__name__)
            from .rheology import EC2RheologicalModel
            ecm = EC2RheologicalModel(fck=fck, cement_class="N").E_c(28.0)
        return ReferenceModulus(value=float(ecm), reference_age=28.0,
                                source=self._SOURCE)

    def design_reduction(self, material):
        r"""
        Partial factors on the materials, EN 1992-1-1 §2.4.2.4.

        Read off the material's own declaration where it carries one,
        so that the factor reported here and the factor the
        constitutive law already applied are the same number and not
        two homes for one fact.

        Parameters
        ----------
        material : Material

        Returns
        -------
        DesignReduction
            With ``where='material'``.
        """
        return DesignReduction(
            where="material",
            gamma_c=_pos(getattr(material, "gamma_c", None)),
            gamma_s=_pos(getattr(material, "gamma_s", None)),
            gamma_p=_pos(getattr(material, "gamma_p", None)),
            source=self._GAMMA_SOURCE)

    def capacity_factor(self, state):
        r"""
        Unity, and that unity is a statement.

        EN 1992-1-1 carries its safety format on the materials, so the
        resistance a section integration produces is **already** a
        design resistance and multiplying it again would double-count.

        Returning ``1.0`` here is therefore not "no factor known" --- a
        code with no factor implemented declines instead, see
        :meth:`NormativeCode.capacity_factor`. It is the positive
        claim that the reduction has happened elsewhere, and
        ``governed_by`` says where.

        Parameters
        ----------
        state : SectionalState
            Accepted and unused: the Eurocode's factor does not depend
            on the solved strain distribution.  Present because the
            *interface* has to accommodate codes whose factor does.

        Returns
        -------
        CapacityFactor
        """
        del state
        return CapacityFactor(
            value=1.0, source=self._GAMMA_SOURCE,
            governed_by="already reduced at the material")

    def rheological_model(self, material, **kwargs):
        r"""
        :class:`~gensec.materials.rheology.EC2RheologicalModel`.

        Parameters
        ----------
        material : Material
        **kwargs
            Forwarded; ``fck`` is filled from the material when absent.

        Returns
        -------
        EC2RheologicalModel
        """
        from .rheology import EC2RheologicalModel
        params = dict(kwargs)
        params.setdefault("fck", self._fck_of(material))
        if params["fck"] is None:
            raise NormativeChoiceNotDeclared(
                f"{self.title}: no f_ck to build a rheological model "
                f"from.", subject="rheological model",
                material=type(material).__name__)
        params.setdefault("cement_class", "N")
        return EC2RheologicalModel(**params)


class NTC18Code(EC2Code):
    r"""
    NTC 2018 (D.M. 17/01/2018).

    A thin specialisation of :class:`EC2Code`: for the conventional
    modulus and for creep and shrinkage, NTC 2018 §11.2.10.3 adopts the
    Eurocode formulation, so the *value* is the same and only the
    **citation** differs.

    That the citation differs is not cosmetic.  A report that derives a
    number under NTC and cites EN 1992-1-1 is not verifiable by the
    engineer signing it.  Subclassing rather than aliasing keeps the
    two apart in every output while keeping one home for the formula.
    """

    key = "ntc18"
    title = "NTC 2018 (D.M. 17/01/2018)"
    _GAMMA_SOURCE = ("NTC 2018 §4.1.2.1.1.1 -- coefficienti parziali "
                     "sui materiali allo stato limite ultimo")
    _SOURCE = ("NTC 2018 §11.2.10.3 -- modulo elastico secante del "
               "calcestruzzo a 28 giorni, formulazione coincidente con "
               "EN 1992-1-1:2004 Table 3.1")


class ACI318Code(NormativeCode):
    r"""
    ACI 318-19.

    The modulus of elasticity of normalweight concrete, §19.2.2.1:

    .. math::

        E_{c} = 4700\sqrt{f'_{c}} \qquad [\mathrm{MPa}]

    stated in SI.  ACI 318-19 writes it as
    :math:`E_c = 57\,000\sqrt{f'_c}` in psi; the two are the same
    clause and the conversion is done here rather than at the boundary,
    because unit leakage is the provider's responsibility and never the
    container's (``7_0-GENSEC_PHASE5_PRIMER.md §2``).

    GenSec maps ``fck`` to :math:`f'_c` directly.  Both are the
    characteristic **cylinder** strength, so no cube conversion is
    involved; the residual difference between a 5 %-fractile
    :math:`f_{ck}` and ACI's specified :math:`f'_c` is a matter of
    which value the engineer enters, not of what this formula does with
    it, and GenSec does not silently convert between them.

    This class exists as much to be *used* as to be a proof: ACI is the
    genericity test bench named in the Phase 5 primer, because its
    structure is genuinely different --- hyperbolic time function,
    volume-to-surface ratio, :math:`4700\sqrt{f'_c}` --- and a
    container that swallows it is a container that is not secretly
    Eurocode-shaped.

    .. note::

       Only the reference modulus is implemented.  The ACI 209R-92
       creep and shrinkage provider is a **declared empty slot**
       (``7_0 §13``): asking for it raises rather than falling back to
       the Eurocode's.
    """

    key = "aci318-19"
    title = "ACI 318-19"
    _SOURCE = ("ACI 318-19 §19.2.2.1 -- Ec = 4700 sqrt(f'c) [MPa] for "
               "normalweight concrete (57000 sqrt(f'c) in psi)")

    def design_reduction(self, material):
        r"""
        **No reduction at the material.** ACI reduces the capacity.

        ACI 318 integrates with the specified :math:`f'_c` and
        :math:`f_y` and applies :math:`\phi` to the nominal strength.
        The partial factors are therefore reported as ``None`` and not
        as ``1.0``: "this code does not reduce materials" and "this
        code reduces materials by unity" are different statements, and
        a caller that dividing by a returned 1.0 would silently turn
        the first into the second.

        Parameters
        ----------
        material : Material

        Returns
        -------
        DesignReduction
            With ``where='capacity'`` and every factor ``None``.
        """
        del material
        return DesignReduction(
            where="capacity",
            source=("ACI 318-19 §5.3 / §21.2 -- load factors and "
                    "strength reduction factors; materials enter at "
                    "their specified values"))

    def capacity_factor(self, state):
        r"""
        **Declared empty slot**: :math:`\phi` is not implemented here.

        The interface exists and this class is the reason it has the
        shape it has, but the factor itself is not shipped, and this
        method declines rather than returning a number.

        Why declining and not ``1.0``.  Unity would publish a
        **nominal** strength as though it were a design strength ---
        unconservative by 10 % to 35 % --- and it would do so silently.
        Every other refusal in GenSec costs the caller an answer; this
        one would cost them a section.

        What implementing it requires, so the slot is a specification
        and not a gap:

        * :math:`\phi = 0.90` tension-controlled, :math:`0.65` (tied)
          or :math:`0.75` (spiral) compression-controlled, linear in
          between, with the branch decided by :math:`\varepsilon_t`
          against :math:`\varepsilon_{ty}` and
          :math:`\varepsilon_{ty} + 0.003` (§21.2.2, and note the 2019
          edition writes the limits against
          :math:`\varepsilon_{ty}` rather than the fixed 0.002/0.005
          of earlier editions);
        * a different value per **action**: 0.75 shear and torsion,
          0.65 bearing (§21.2.1).  ``SectionalState.action`` carries
          the question;
        * the axial cap :math:`\phi P_{n,max} = 0.80\,\phi P_o` (tied)
          or :math:`0.85\,\phi P_o` (spiral), §22.4.2.1, which is a
          **truncation** of the interaction domain and not a scaling
          of it --- so it cannot be expressed as a factor at all and
          needs its own hook when the ACI work is scoped;
        * the equivalent rectangular stress block, §22.2.2.4:
          :math:`\beta_1` and the 0.85 are one calibrated pair.  **The
          0.85 there is not** :math:`\alpha_{cc}`.  They coincide
          numerically at the Eurocode's recommended value, which makes
          conflating them unusually tempting and unusually silent.

        Parameters
        ----------
        state : SectionalState

        Raises
        ------
        NormativeChoiceNotDeclared
            Always, in this build.
        """
        del state
        raise NormativeChoiceNotDeclared(
            "ACI 318-19 §21.2 strength-reduction factors are a "
            "declared empty slot in this build: phi is not "
            "implemented, and GenSec will not return 1.0 in its "
            "place. Unity would publish a NOMINAL strength as a "
            "design strength, which is unconservative by 10 % to "
            "35 % and silent. Supply the factor yourself, or verify "
            "to a code whose reduction happens on the materials.",
            subject="capacity factor (phi)", material="section",
            remedy="implement ACI318Code.capacity_factor, or declare "
                   "the model under a code that reduces at the "
                   "material end.")

    def axial_cap(self, P_o, confinement="tied"):
        r"""
        **Declared empty slot**: the cap exists in the code, not here.

        ACI 318-19 §22.4.2.1 truncates the compression branch of the
        interaction domain at :math:`0.80 P_o` for tied members and
        :math:`0.85 P_o` for spirally reinforced ones, to account for
        the accidental eccentricity a nominally concentric column
        always has.

        This class declines instead of inheriting the base
        implementation, and the distinction is the whole reason the
        method exists.  The base returns ``None``, which is the
        statement *this code puts no cap on the axial branch*.  That
        statement is **false** of ACI 318-19, so inheriting it would
        publish an interaction domain whose compression branch runs
        20 % too tall --- unconservative precisely where a column is
        governed, and silent about it.

        ``None`` and "not implemented" are different answers to
        different questions, and a code that confuses them hands the
        engineer a number instead of a refusal.

        Parameters
        ----------
        P_o : float
            Nominal concentric axial resistance [N].
        confinement : str, optional
            ``'tied'``, ``'spiral'`` or ``'none'``.

        Raises
        ------
        NormativeChoiceNotDeclared
            Always, in this build.

        Notes
        -----
        Implementing it is three lines and one decision that is not
        mine: whether ``confinement`` should be a property of the
        section --- it describes the transverse reinforcement, which
        is section geometry GenSec does not currently model --- or an
        argument the caller supplies per verification.
        """
        del P_o
        raise NormativeChoiceNotDeclared(
            f"{self.title} §22.4.2.1 caps the compression branch at "
            f"0.80 P_o (tied) or 0.85 P_o (spiral), and that cap is a "
            f"declared empty slot in this build. GenSec declines "
            f"rather than reporting no cap: an uncapped ACI "
            f"interaction domain is about 20 % too tall on the "
            f"compression branch, which is unconservative exactly "
            f"where a column is governed. Confinement asked for: "
            f"{confinement!r}.",
            subject="axial cap", material=self.title,
            remedy="supply the capped domain yourself, or verify "
                   "flexure only until the slot is filled.")

    def reference_modulus(self, material):
        r"""
        :math:`E_c = 4700\sqrt{f'_c}`.  See :class:`ACI318Code`.

        Parameters
        ----------
        material : Material

        Returns
        -------
        ReferenceModulus

        Raises
        ------
        NormativeChoiceNotDeclared
            If the material states no positive ``fck``.
        """
        fck = self._fck_of(material)
        if fck is None:
            raise NormativeChoiceNotDeclared(
                f"{self.title}: this material states no positive "
                f"f'_c, and §19.2.2.1 needs it.",
                material=type(material).__name__)
        # ACI has no 28-day *derivation*; f'c is by definition the
        # 28-day specified strength, so the reference age is 28 d for
        # the same reason and not by analogy with the Eurocode.
        return ReferenceModulus(value=4700.0 * math.sqrt(fck),
                                reference_age=28.0, source=self._SOURCE)


class CustomCode(NormativeCode):
    r"""
    The explicit opt-out: the input supplies everything.

    Declared as ``code: custom``.  GenSec derives nothing --- not as a
    failure, but because that is what was asked for.  Every refusal
    this code issues says so, which is the difference between it and
    :class:`UndeclaredCode` and the reason the two are not merged.
    """

    key = "custom"
    title = "custom (no code)"

    def axial_cap(self, confinement="none"):
        r"""
        Declines: a custom declaration says the input supplies
        every code-dependent value, and the axial cap is one of them.

        Parameters
        ----------
        confinement : str

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.  The base implementation returns ``None``, meaning
            *this code puts no cap on the axial branch* -- a positive
            statement about a code, and one this object has no
            authority to make.  Declining and returning "no cap" are
            different answers and must not be spelled the same way.
        """
        del confinement
        raise NormativeChoiceNotDeclared(
            "this material is declared 'code: custom', so GenSec states no cap on the axial branch of the interaction domain. Apply one yourself, or name the code it should be read under.",
            subject="axial cap", material="section")

    def reference_modulus(self, material):
        r"""
        Always declines, naming the declaration that caused it.

        Parameters
        ----------
        material : Material

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.
        """
        raise NormativeChoiceNotDeclared(
            f"this material is declared 'code: custom', so GenSec "
            f"derives nothing for it --- that is what the declaration "
            f"asks for. It must therefore supply its conventional "
            f"modulus itself. Either give it directly, or name the "
            f"code it should be read under "
            f"({', '.join(k for k in known_codes() if k not in ('custom', 'undeclared'))}).",
            material=type(material).__name__,
            remedy="supply the value, or replace 'code: custom' with a "
                   "code name.")

    def design_reduction(self, material):
        r"""
        Always declines: a custom material states its own values.

        Parameters
        ----------
        material : Material

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.
        """
        raise NormativeChoiceNotDeclared(
            "this material is declared 'code: custom', so GenSec "
            "applies no partial factor to it. Declare the design "
            "values directly, or name a code.",
            subject="design reduction",
            material=type(material).__name__)

    def capacity_factor(self, state):
        r"""
        Always declines.

        Never ``1.0``: unity would silently publish an unfactored
        resistance as a design value, and the whole point of ``code:
        custom`` is that the input owns the safety format and GenSec
        does not guess at it.

        Parameters
        ----------
        state : SectionalState

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.
        """
        del state
        raise NormativeChoiceNotDeclared(
            "this model is declared 'code: custom', so GenSec applies "
            "no capacity factor and will not substitute 1.0 for one: "
            "that would publish a nominal resistance as a design "
            "resistance. Apply your own factor to the reported "
            "nominal values.",
            subject="capacity factor", material="section")

    def rheological_model(self, material, **kwargs):
        r"""
        Always declines.  See :meth:`reference_modulus`.

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.
        """
        raise NormativeChoiceNotDeclared(
            f"this material is declared 'code: custom', so GenSec "
            f"applies no creep, shrinkage or relaxation law to it. "
            f"Supply a provider explicitly "
            f"(ConstructionTimeline(rheology=...)), or name a code.",
            subject="rheological model",
            material=type(material).__name__)


class UndeclaredCode(NormativeCode):
    r"""
    No code was declared anywhere, and none could be inferred.

    Distinct from :class:`CustomCode` in exactly one way, and it is the
    way that matters to whoever reads the error: this is an **omission**
    and custom is a **choice**.  The remedy differs accordingly.
    """

    key = "undeclared"
    title = "undeclared"

    def axial_cap(self, confinement="none"):
        r"""
        Declines: no code is declared, so there is no cap to state
        and no code that could be asked for one.

        Parameters
        ----------
        confinement : str

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.  The base implementation returns ``None``, meaning
            *this code puts no cap on the axial branch* -- a positive
            statement about a code, and one this object has no
            authority to make.  Declining and returning "no cap" are
            different answers and must not be spelled the same way.
        """
        del confinement
        raise NormativeChoiceNotDeclared(
            "no code is declared, so GenSec cannot say whether the axial branch of the interaction domain is capped. Declare 'code:' on the model or on the material.",
            subject="axial cap", material="section")

    def design_reduction(self, material):
        r"""
        Always declines, naming the omission.

        Parameters
        ----------
        material : Material

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.
        """
        raise NormativeChoiceNotDeclared(
            "no code is declared, so there is no safety format to "
            "apply: whether the reduction belongs on the materials or "
            "on the capacity is a property of the code, not of the "
            "section.",
            subject="design reduction",
            material=type(material).__name__)

    def capacity_factor(self, state):
        r"""
        Always declines.  Never ``1.0``.

        Parameters
        ----------
        state : SectionalState

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.
        """
        del state
        raise NormativeChoiceNotDeclared(
            "no code is declared, so there is no capacity factor. "
            "GenSec will not substitute 1.0: an undeclared model that "
            "returned unity here would publish a nominal resistance "
            "as a design resistance.",
            subject="capacity factor", material="section")

    def reference_modulus(self, material):
        r"""
        Always declines, naming the omission.

        Parameters
        ----------
        material : Material

        Raises
        ------
        NormativeChoiceNotDeclared
            Always.
        """
        names = ", ".join(k for k in known_codes()
                          if k not in ("custom", "undeclared"))
        raise NormativeChoiceNotDeclared(
            f"no code is declared for this material, so there is no "
            f"conventional modulus to derive. f_ck is an input, not a "
            f"code: EN 1992-1-1 Table 3.1 and ACI 318-19 §19.2.2.1 "
            f"read the same f_ck and disagree by 33.9 % at C25 and "
            f"18.5 % at C40. Declare 'code:' at the top level of the "
            f"input or on this material ({names}), use a material "
            f"type that names one (e.g. 'type: concrete_ec2_gen1'), "
            f"or declare 'code: custom' and supply the value "
            f"yourself.",
            material=type(material).__name__,
            remedy="add 'code:' to the input, or 'code: custom'.")


#: The shipped registry.  Adding a code is a subclass and one line.
_REGISTRY: Dict[str, NormativeCode] = {}

#: Aliases accepted in YAML, resolved to a registry key.
_ALIASES: Dict[str, str] = {
    "en1992": "ec2", "en1992-1-1": "ec2", "eurocode2": "ec2",
    "eurocode": "ec2", "ec2-1-1": "ec2",
    "ntc": "ntc18", "ntc2018": "ntc18", "ntc-18": "ntc18",
    "aci": "aci318-19", "aci318": "aci318-19", "aci-318": "aci318-19",
    "aci318-19m": "aci318-19",
    "none": "custom", "user": "custom",
}

def register_code(code: NormativeCode) -> None:
    r"""
    Add a code to the registry.

    Parameters
    ----------
    code : NormativeCode

    Raises
    ------
    ValueError
        On a duplicate or empty key.
    """
    if not code.key:
        raise ValueError("a NormativeCode must declare a key.")
    if code.key in _REGISTRY:
        raise ValueError(f"code {code.key!r} is already registered.")
    _REGISTRY[code.key] = code


def known_codes():
    r"""
    The registered code keys, in registration order.

    Returns
    -------
    tuple of str
    """
    return tuple(_REGISTRY)


for _c in (EC2Code(), NTC18Code(), ACI318Code(), CustomCode(),
           UndeclaredCode()):
    register_code(_c)

#: The explicit opt-out.
CUSTOM = _REGISTRY["custom"]

#: The absence of a declaration.
UNDECLARED = _REGISTRY["undeclared"]


def resolve_code(material_spec: Optional[dict] = None,
                 model_code=None,
                 material_type: Optional[str] = None) -> NormativeCode:
    r"""
    Resolve which code governs a material.

    Order, first hit wins: the material's own ``code:``, then the
    model's top-level ``code:``.  There is **no third source** and no
    default.  A material with neither resolves to :data:`UNDECLARED`,
    and :func:`gensec.io_yaml.load_yaml` refuses to build a model that
    contains one.

    Inference was tried and removed.  Until this was tightened, a
    material *type* that happened to name a code --- ``type:
    concrete_ec2_gen1`` --- was read as a declaration of it.  That
    reading is defensible and it is still wrong, for two reasons that
    only became visible once the declaration existed:

    * a type names a **constitutive shape**, not a verification code.
      A parabola-rectangle stress block is a curve; which code's
      partial factors, which conventional modulus and which creep law
      accompany it are separate facts, and a model may legitimately
      pair that curve with a code that did not invent it;
    * nothing says a GenSec section is concrete at all.  The container
      integrates fibres of arbitrary materials, and a rule that reads
      the code out of a concrete type name is a rule that only works
      for the sections someone happened to think of first.

    Parameters
    ----------
    material_spec : dict, optional
        The material's YAML mapping; only its ``'code'`` key is read.
    model_code : str or NormativeCode, optional
        The model-level declaration.
    material_type : str, optional
        Accepted and **ignored**.  Kept in the signature so that a
        caller written against the inferring version fails visibly ---
        by getting :data:`UNDECLARED` and a refusal that names the
        omission --- rather than silently resolving to something.

    Returns
    -------
    NormativeCode
        :data:`UNDECLARED` when nothing was declared.

    Raises
    ------
    ValueError
        On a code name that is not registered.  Declining loudly is
        deliberate: a typo silently falling back to a default would
        reintroduce exactly the silent choice this module removes.
    """
    del material_type                       # accepted, deliberately unused
    own = (material_spec or {}).get("code")
    for candidate in (own, model_code):
        if candidate is None:
            continue
        if isinstance(candidate, NormativeCode):
            return candidate
        key = str(candidate).strip().lower()
        key = _ALIASES.get(key, key)
        if key not in _REGISTRY:
            raise ValueError(
                f"unknown code {candidate!r}. Known codes: "
                f"{', '.join(known_codes())}. GenSec will not fall "
                f"back to a default here: a mistyped code name that "
                f"silently became the Eurocode is the failure this "
                f"declaration exists to prevent.")
        return _REGISTRY[key]
    return UNDECLARED


class CodeReferenceModulus:
    r"""
    The agnostic reference-modulus rule: ask each material's own code.

    A :class:`~gensec.materials.base.ReferenceModulusRule` in every way
    that matters to the resolver, but it derives nothing itself.  It
    reads the :class:`NormativeCode` stamped on the material by the
    loader and asks *that* --- so one model can legitimately mix a
    precast beam under EN 1992-1-1 with a topping under ACI 318-19, and
    each element answers under the code it was declared with.

    This is what ``code:`` in the input file buys at the timeline
    level, and since Phase 21 it is the **only** rule a timeline
    resolves to.  There is no shipped default beneath it: an
    undeclared model used to fall through to the Eurocode's rule, and
    that fall-through was the last surface on which a normative choice
    happened without being declared.

    Parameters
    ----------
    fallback : NormativeCode, optional
        Applied to a material that carries no stamp of its own --- one
        built directly in Python rather than through the loader, since
        the loader refuses an undeclared model outright.  Defaults to
        :data:`UNDECLARED`, which declines and names the omission.

    Attributes
    ----------
    reference_age : float
        ``28.0``.  Every shipped code states its conventional modulus
        at 28 days; a code that did not would answer with its own age
        in the :class:`~gensec.materials.base.ReferenceModulus` it
        returns, and this attribute exists only for callers that need
        one number before asking.
    source : str
    """

    reference_age = 28.0
    source = ("resolved per material from its declared code "
              "(NormativeCode); see the citation on each returned "
              "ReferenceModulus")

    def __init__(self, fallback=None):
        self.fallback = fallback if fallback is not None else UNDECLARED

    def __repr__(self):
        return f"<CodeReferenceModulus fallback={self.fallback.key!r}>"

    def E_c(self, material):
        r"""
        :math:`E_c` [MPa] of *material* under its own declared code.

        Parameters
        ----------
        material : Material

        Returns
        -------
        float or None
            ``None`` when the material's code declines --- a steel, a
            tabulated law, a material declared ``code: custom``, or one
            with no declaration at all.  Declining is the correct
            answer here for the same reason it is in
            :class:`~gensec.materials.rheology.EC2ReferenceModulus`:
            the caller, not the rule, owns the refusal, and a rule that
            invented a value for a material it does not describe would
            be the silent choice all of this removes.
        """
        try:
            declared = material.reference_modulus()
        except (NotImplementedError, AttributeError):
            declared = None
        if declared is not None and float(declared.value) > 0.0:
            return float(declared.value)
        code = getattr(material, "code", None) or self.fallback
        try:
            value = code.reference_modulus(material).value
        except (NotImplementedError, AttributeError):
            return None
        value = float(value)
        return value if value > 0.0 else None
