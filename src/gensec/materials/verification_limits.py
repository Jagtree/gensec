# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.
#
# GenSec is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# GenSec is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GenSec.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
r"""
SLS stress-limit sets — generic container plus normative providers.

Architecture
------------
The verification engine (:mod:`gensec.solver.sls`) consumes **numbers**,
never normative logic: a :class:`StressLimits` instance is a plain
container of stress thresholds and decompression parameters.  Which
values those thresholds take is the job of *providers* — one function
per normative document, mirroring the pattern established by
``gamma_s_prestress`` and ``delta_sigma_p_uls`` in
:mod:`gensec.materials.prestress_properties`.

This module ships the EC2 provider (:func:`ec2_stress_limits`).
NTC 2018 values are **deliberately not asserted** here (same policy as
``delta_sigma_p_uls``): they are structurally identical to EC2 and are
supplied through the ``k*`` override hooks or by constructing
:class:`StressLimits` directly.  Future normative bridges (ACI, BS,
national annexes) plug in as sibling provider functions with the same
return type; nothing in the engine changes.

All strength inputs to the providers are **explicit**.  In particular,
time-dependent strengths at transfer (:math:`f_{ck}(t)`) are computed
by the caller — e.g. through the EC2 bridge object
(``concrete.ec2`` → :class:`~gensec.materials.ec2_properties.fben2`
with its ``time`` argument) — and passed in as numbers.  The provider
never reaches into a material or a rheological model on its own; this
is the same never-silent policy that governs the SLS moduli
(:func:`gensec.solver.sls.resolve_sls_moduli`).
"""

from dataclasses import InitVar, dataclass, field
from typing import Optional

import numpy as np


# ==================================================================
#  Generic container
# ==================================================================


#: The strength fields a ``custom_zone_strengths`` entry may declare.
#:
#: One entry today, and the tuple exists anyway: it is what makes the
#: block **per field** rather than all-or-nothing (Phase-32 constraint
#: 3).  A zone that declares ``f_ck`` must keep deriving everything it
#: did not declare, and a set of known fields is what lets a future
#: strength be added without turning the block into a replacement.
CUSTOM_ZONE_STRENGTH_FIELDS = ("f_ck",)


@dataclass(frozen=True)
class StressLimits:
    r"""
    Normative-agnostic set of SLS stress thresholds for one check.

    All stress values are **positive magnitudes** in MPa; the engine
    applies them with the GenSec sign convention (compression
    negative, tension positive):

    - concrete compression check, **per zone**:
      :math:`|\min_i \sigma_{c,i}|_z \le k_c\, f_{ck}(z)`;
    - reinforcement tension check:
      :math:`\max_i \sigma_{s,i} \le \sigma_{s,\mathrm{tens}}`;
    - prestressing-steel check:
      :math:`\max_i \sigma_{p,i} \le \sigma_{p,\max}`;
    - decompression: concrete stress non-tensile at probe points
      located ``c_dec`` from each bonded tendon toward the tensile
      side (see :func:`gensec.solver.sls._decompression_probe`).

    Every threshold is optional: ``None`` disables the corresponding
    check (the engine reports it as ``skipped``, never silently).

    The concrete limit is a coefficient, not a product
    --------------------------------------------------
    Until Phase 32 this container carried ``sigma_c_comp``: **one
    number for the whole stage**, compared against the largest
    compression anywhere in the section.  On a composite that is not an
    imprecise limit, it is a **false pass**.  Measured on a two-zone
    bench (base C45, topping C30, :math:`M_x = -350` kNm) the topping
    reached 22.5753 MPa, passed against the section-wide 27.0, and the
    stage was reported verified while that zone stood at 125 % of what
    its own material prescribes.

    The limit is therefore derived where the material lives:

    .. math::

        \sigma_{\lim}(z) = k_c \times
        \begin{cases}
            f_{ck}^{\,\mathrm{custom}}(z)
                & \text{if declared in } \texttt{custom\_zone\_strengths}\\
            f_{ck}\big(\text{material of } z\big)
                & \text{otherwise}
        \end{cases}

    The ``otherwise`` branch derives **per zone, always**, so a
    section-wide product has no place in the formula and is gone
    rather than kept beside it with a precedence rule -- the one move
    ``30_2`` §4 forbids by name, and the move Phase 30 already made on
    ``fct_eff`` for the same reason: a field the formula does not use
    has no title to be here.

    What did **not** move, and why
    ------------------------------
    ``sigma_s_tens`` and ``sigma_p_max`` stay section-wide.  A rebar
    and a tendon are *elements*, not zones: each already carries its
    own material, and the quantity being limited is a property of that
    material.  Sweeping them onto the zone axis would have been a
    change of subject, not a repair.

    ``decompression`` and ``c_dec`` stay too.  Neither is a strength
    and neither is derivable from any material: they belong to the
    combination and to the normative document.

    There is deliberately **no tensile threshold here**.  Until Phase 27
    the SLS solve was linear-elastic and uncracked, so a
    :math:`f_{ct,\mathrm{eff}}` was carried in this container to flag a
    solution whose own basis had been violated.  Phase 27 put the
    service law into the view: the fibre now releases at
    :math:`f_{ct,\mathrm{eff}}` and the solve returns the cracked
    equilibrium directly, so the state that flag warned about is no
    longer reachable.  The tensile strength is a property of the
    concrete and is declared on
    :class:`gensec.materials.service_concrete.ServiceConcrete`; a second
    copy here would be the same normative quantity entering through two
    doors.

    Parameters
    ----------
    name : str
        Identifier of the limit set (e.g. ``"EC2 characteristic"``).
    k_c : float or None, optional
        The concrete coefficient for this kind of combination --
        :math:`k_1` at characteristic, :math:`k_2` at quasi-permanent,
        :math:`k_t` at transfer under EC2, whatever the governing
        document calls it elsewhere.  **A coefficient, never a
        product**: a value above one is refused, because the failure
        this creates is the obvious one, a caller who used to write
        ``sigma_c_comp=0.6*35`` writing ``k_c=0.6*35``.  Taken at face
        value that would be multiplied by :math:`f_{ck}` a second time
        and produce a limit no check could ever fail.
        ``None`` disables the concrete check, and the engine records a
        ``skipped`` entry **per zone** naming this field.
    custom_zone_strengths : dict or None, optional
        ``{zone name: {"f_ck": value}}`` -- strengths the caller
        declares for named zones, overriding what the zone's own
        material would give, **per field**.  Fields outside
        :data:`CUSTOM_ZONE_STRENGTH_FIELDS` are refused by name.

        A strength here is a **strength**, never a product: the
        coefficient multiplies it exactly once, whichever branch
        produced it.

        This is the block through which a strength that GenSec cannot
        derive enters -- most importantly :math:`f_{ck}(t)` at
        transfer, which is not the material's 28-day value and which no
        zone material knows.  See :func:`ec2_stress_limits`.
    sigma_s_tens : float or None, optional
        Reinforcing-steel tensile stress limit [MPa].
    sigma_p_max : float or None, optional
        Prestressing-steel stress limit [MPa] (mean value after
        losses at SLS, EC2 §7.2(5) :math:`k_5 f_{pk}`).
    decompression : bool, optional
        Whether to run the decompression check on every bonded
        tendon of the stage.  Default ``False``.
    c_dec : float, optional
        Decompression cover distance [mm]: the probe point is placed
        ``c_dec`` from the tendon axis toward the tensile side.
        Default ``25.0`` (EC2 §7.3.1(5): all parts of bonded tendons
        or ducts at least 25 mm within concrete in compression).
        Normative-agnostic parameter: any other document's distance
        is supplied here.

    Raises
    ------
    ValueError
        If any provided threshold is non-finite or non-positive; if
        ``c_dec`` is negative; if ``k_c`` is not a finite number in
        :math:`(0, 1]`; if ``custom_zone_strengths`` is malformed or
        declares an unknown field; or if the removed ``sigma_c_comp``
        is named.

    Examples
    --------
    The ordinary case declares no strengths at all -- every zone is
    satisfied by its own material:

    >>> lim = StressLimits(name="EC2 characteristic", k_c=0.6,
    ...                    sigma_s_tens=360.0)
    >>> lim.strength_for("topping", 30.0)
    (30.0, 'material')

    A declared strength displaces the material's, and says so:

    >>> lim = StressLimits(name="transfer", k_c=0.6,
    ...                    custom_zone_strengths={"base": {"f_ck": 28.0}})
    >>> lim.strength_for("base", 45.0)
    (28.0, 'custom')
    >>> lim.strength_for("topping", 30.0)
    (30.0, 'material')
    """

    name: str = ""
    k_c: Optional[float] = None
    custom_zone_strengths: Optional[dict] = None
    sigma_s_tens: Optional[float] = None
    sigma_p_max: Optional[float] = None
    decompression: bool = False
    c_dec: float = 25.0

    #: Not a field: an :class:`~dataclasses.InitVar` whose only purpose
    #: is to catch the name Phase 32 removed.  A pseudo-field is absent
    #: from :func:`dataclasses.fields`, so the container genuinely no
    #: longer carries a section-wide concrete limit -- but a caller
    #: whose script still writes ``sigma_c_comp=`` gets a message that
    #: names the replacement instead of a bare ``TypeError`` naming
    #: only the keyword.  This is the pattern Phase 30 established when
    #: it removed ``fct_eff``.
    sigma_c_comp: InitVar[Optional[float]] = None

    def __post_init__(self, sigma_c_comp):
        if sigma_c_comp is not None:
            raise ValueError(
                "'sigma_c_comp' was removed from StressLimits in Phase "
                "32.  It was ONE concrete limit for the whole stage, "
                "compared against the largest compression anywhere in "
                "the section; on a composite that is a false pass, not "
                "an imprecise limit -- a topping at 125 % of its own "
                "material's prescription passed against the base's "
                "stronger limit and the stage was reported verified.  "
                "The limit is now derived per zone as k_c * f_ck(zone). "
                "Declare the COEFFICIENT as k_c (0.6, not 0.6*f_ck); "
                "each zone supplies its own strength from its material, "
                "or from custom_zone_strengths where you declare it."
            )
        for attr in ("sigma_s_tens", "sigma_p_max"):
            v = getattr(self, attr)
            if v is None:
                continue
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(
                    f"StressLimits.{attr} must be a finite, strictly "
                    f"positive magnitude [MPa] or None; got {v!r}."
                )
        if not np.isfinite(self.c_dec) or self.c_dec < 0.0:
            raise ValueError(
                f"StressLimits.c_dec must be a finite, non-negative "
                f"distance [mm]; got {self.c_dec!r}."
            )
        self._validate_k_c()
        self._validate_custom_zone_strengths()

    # --------------------------------------------------------------
    #  Validation
    # --------------------------------------------------------------

    def _validate_k_c(self) -> None:
        r"""
        Refuse anything that is not a coefficient.

        Two refusals with two different messages, because they are two
        different mistakes.  A non-positive or non-finite value is
        malformed.  A value **above one** is almost certainly a
        product: no stress-limit coefficient in EN 1992-1-1, NTC 2018
        or ACI 318-19 exceeds unity, and accepting 21.0 here would
        multiply it by :math:`f_{ck}` again -- a limit of 735 MPa that
        no compression could ever reach, which is worse than the defect
        this field was introduced to repair, because it is silent.
        """
        if self.k_c is None:
            return
        k = float(self.k_c)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError(
                f"StressLimits.k_c must be a finite, strictly positive "
                f"coefficient or None; got {self.k_c!r}."
            )
        if k > 1.0:
            raise ValueError(
                f"StressLimits.k_c = {k!r} is not a coefficient: a "
                f"stress-limit coefficient is a fraction of a strength "
                f"and no normative document puts one above 1.  This is "
                f"almost certainly a product -- 'k_c' replaced the "
                f"section-wide 'sigma_c_comp' in Phase 32, and it takes "
                f"the coefficient ALONE (0.6, not 0.6*f_ck).  The "
                f"strength comes from each zone's own material, or from "
                f"'custom_zone_strengths' where the caller declares it. "
                f"Accepting this number would multiply it by f_ck a "
                f"second time."
            )

    def _validate_custom_zone_strengths(self) -> None:
        r"""
        Refuse a malformed or unrecognised custom-strength block.

        Same policy as the YAML top level and as ``material_zones``: a
        field read by nothing must be accepted by nothing.  A misspelt
        ``f_ckk`` that loaded silently would leave the zone deriving
        from its material while its author believed otherwise -- the
        exact shape of the defect Phase 32 exists to close, one level
        down.
        """
        block = self.custom_zone_strengths
        if block is None:
            return
        if not isinstance(block, dict):
            raise ValueError(
                f"StressLimits.custom_zone_strengths must be a mapping "
                f"{{zone name: {{'f_ck': value}}}}, got "
                f"{type(block).__name__}."
            )
        for zone, spec in block.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"custom_zone_strengths['{zone}'] must be a mapping "
                    f"of strength fields, got {type(spec).__name__}."
                )
            unknown = [f for f in spec
                       if f not in CUSTOM_ZONE_STRENGTH_FIELDS]
            if unknown:
                raise ValueError(
                    f"custom_zone_strengths['{zone}'] declares unknown "
                    f"field(s) {sorted(unknown)}.  Recognised: "
                    f"{sorted(CUSTOM_ZONE_STRENGTH_FIELDS)}.  A field "
                    f"nothing reads is refused rather than ignored: "
                    f"accepting it would leave the zone deriving from "
                    f"its material while the declaration says otherwise."
                )
            for field_name, value in spec.items():
                v = float(value)
                if not np.isfinite(v) or v <= 0.0:
                    raise ValueError(
                        f"custom_zone_strengths['{zone}']"
                        f"['{field_name}'] must be a finite, strictly "
                        f"positive strength [MPa]; got {value!r}."
                    )

    # --------------------------------------------------------------
    #  Resolution
    # --------------------------------------------------------------

    def strength_for(self, zone_name, material_f_ck):
        r"""
        The strength this limit set applies to *zone_name*, and its origin.

        Parameters
        ----------
        zone_name : str
            The zone's label, as it appears in
            ``GenericSection.zone_names``.
        material_f_ck : float or None
            The characteristic strength of the zone's own material, or
            ``None`` when that material declares none (a zone whose
            bulk material is not a concrete, for instance).

        Returns
        -------
        tuple
            ``(f_ck, provenance)`` with *provenance* one of
            ``'custom'`` or ``'material'``, or ``(None, 'undeclared')``
            when neither branch yields a strength.

        Notes
        -----
        **``provenance`` says who produced the number, not whether it
        complies.**  ``'material'`` means GenSec derived it from the
        zone's own material.  ``'custom'`` means the caller declared
        it and GenSec did not derive it -- which is the true and
        weakest statement available, because a strength arriving as a
        ``float`` carries no evidence of where it came from.  An
        :math:`f_{ck}(t)` computed strictly under EN 1992-1-1 §3.1.2(6)
        and an invented number are indistinguishable here, so the label
        claims only what can be known.  A check at transfer will
        therefore always read ``'custom'``, and that is correct: at
        transfer the strength is always a declared fact.
        """
        block = self.custom_zone_strengths or {}
        spec = block.get(zone_name)
        if spec is not None and "f_ck" in spec:
            return float(spec["f_ck"]), "custom"
        if material_f_ck is None:
            return None, "undeclared"
        f = float(material_f_ck)
        if not np.isfinite(f) or f <= 0.0:
            return None, "undeclared"
        return f, "material"


# ==================================================================
#  EC2 provider
# ==================================================================


#: EC2 recommended coefficients (EN 1992-1-1).  Every entry is an
#: override hook of :func:`ec2_stress_limits`; national annexes and
#: NTC 2018 supply their values through the same keywords.
EC2_RECOMMENDED = {
    "k1": 0.6,          # §7.2(2)  sigma_c <= k1*fck   (characteristic)
    "k2": 0.45,         # §7.2(3)  sigma_c <= k2*fck   (quasi-permanent)
    "k3": 0.8,          # §7.2(5)  sigma_s <= k3*fyk   (characteristic)
    "k5": 0.75,         # §7.2(5)  sigma_p <= k5*fpk   (mean, after losses)
    "k_transfer": 0.6,  # §5.10.2.2(5)  sigma_c <= 0.6*fck(t) at transfer
    "c_dec": 25.0,      # §7.3.1(5)  decompression cover distance [mm]
}


def _refuse_fct_eff():
    r"""
    Refuse a tensile threshold on the limit set (Phase 30).

    Its own function so that the refusal is reachable before
    the override loop, where keyword order would otherwise
    decide which of two removed fields gets reported.

    Raises
    ------
    ValueError
        Always.
    """
    raise ValueError(
                "'fct_eff' is not a stress limit and was removed from "
                "StressLimits in Phase 30.  It was the uncracked-basis "
                "threshold, and the basis it guarded no longer exists: "
                "since Phase 27 the SLS view carries the service law, "
                "the fibre releases at f_ct,eff, and the solve returns "
                "the cracked equilibrium directly.  The effective "
                "tensile strength is a property of the concrete -- "
                "declare it as ServiceConcrete(fct_eff=...), where it "
                "is read."
    )


def ec2_stress_limits(kind, *, fyk=None, fpk=None,
                      decompression=False, custom_zone_strengths=None,
                      **overrides):
    r"""
    Build a :class:`StressLimits` set from EC2 recommended values.

    What this provider hands over, since Phase 32
    ---------------------------------------------
    The **coefficient**, not the product.  Which strength that
    coefficient multiplies is a property of a *zone*, and a zone is not
    visible from a provider that builds one object for the whole stage.
    Before Phase 32 this function took ``fck`` and returned
    :math:`k f_{ck}`; on a composite that single number was then
    compared against every zone, which is checking one zone against
    another zone's material.

    So ``fck`` is gone from the signature.  It was here only to form a
    product that no longer exists, and a parameter that produces
    nothing is a parameter that misleads.

    Parameters
    ----------
    kind : {'transfer', 'characteristic', 'quasi_permanent'}
        Which EC2 limit family to build:

        ``'transfer'``
            :math:`\sigma_c \le k_t\, f_{ck}(t)` (EC2 §5.10.2.2(5),
            recommended :math:`k_t = 0.6`; may be raised to 0.7 for
            pretensioned members if justified -- supply
            ``k_transfer=0.7``).  **Requires** ``custom_zone_strengths``:
            see the note below.
        ``'characteristic'``
            :math:`\sigma_c \le k_1 f_{ck}`,
            :math:`\sigma_s \le k_3 f_{yk}`,
            :math:`\sigma_p \le k_5 f_{pk}` (EC2 §7.2(2), (5)).
        ``'quasi_permanent'``
            :math:`\sigma_c \le k_2 f_{ck}` (EC2 §7.2(3), linear
            creep validity).

    fyk : float, optional
        Reinforcing-steel characteristic yield strength [MPa].
        Consumed by ``'characteristic'``; when omitted, the steel
        check is disabled (``sigma_s_tens=None``).
    fpk : float, optional
        Prestressing-steel characteristic strength [MPa].  Consumed
        by ``'characteristic'``; when omitted, the tendon check is
        disabled.
    decompression : bool, optional
        Enable the decompression check.  Default ``False``.
    custom_zone_strengths : dict or None, optional
        ``{zone name: {"f_ck": value}}``, passed through to
        :class:`StressLimits`.  Mandatory for ``kind='transfer'``.
    **overrides
        Any key of :data:`EC2_RECOMMENDED` (``k1``, ``k2``, ``k3``,
        ``k5``, ``k_transfer``, ``c_dec``).  This is the national
        annex / NTC hook.

    Returns
    -------
    StressLimits

    Raises
    ------
    ValueError
        Unknown ``kind``; unknown override key; ``kind='transfer'``
        with no ``custom_zone_strengths``; or a reference to one of
        the two fields earlier phases removed (``fct_eff``,
        ``sigma_c_comp``).

    Notes
    -----
    **Why transfer refuses rather than falling back.**  At transfer the
    governing strength is :math:`f_{ck}(t)`, the strength at the age
    the tendons are released.  That is not the material's 28-day
    :math:`f_{ck}`, and no zone material knows it: EN 1992-1-1
    §3.1.2(6) gives it as a function of age and cement class, and the
    age is a property of the construction history.  Deriving it from
    the 28-day value would be a silent normative fallback in the
    **unsafe** direction -- a limit higher than the concrete can carry
    on the day it is loaded.  So the caller computes it, through the
    concrete's ``ec2`` bridge or otherwise, and declares it per zone.

    The check it produces will report ``provenance='custom'``.  That is
    correct and not a demotion: at transfer the strength is always a
    declared fact, and ``provenance`` records who produced a number,
    not whether it complies -- see :meth:`StressLimits.strength_for`.

    Examples
    --------
    Characteristic limits.  No strength is named anywhere: every zone
    derives its own.

    >>> lim = ec2_stress_limits('characteristic', fyk=450.0, fpk=1860.0)
    >>> lim.k_c
    0.6

    Transfer for a pretensioned member released at 5 days, the strength
    at that age computed by the caller:

    >>> lim = ec2_stress_limits(
    ...     'transfer', k_transfer=0.7,
    ...     custom_zone_strengths={'base': {'f_ck': 28.0}})
    >>> lim.k_c
    0.7

    NTC 2018 quasi-permanent limit (same coefficient as EC2 in this
    case, shown as an explicit override for clarity):

    >>> ec2_stress_limits('quasi_permanent', k2=0.45).k_c
    0.45
    """
    k = dict(EC2_RECOMMENDED)
    # Checked before the loop, not inside it: keyword order is the
    # caller's, and a call naming BOTH removed fields must hear about
    # the one whose replacement lives furthest away.  ``fct_eff`` moved
    # to another class (ServiceConcrete); ``fck`` merely stopped being
    # needed here.  Iteration order decided which message arrived,
    # which is a refusal whose content depends on how the call was
    # typed.
    if "fct_eff" in overrides:
        _refuse_fct_eff()
    for key, val in overrides.items():
        if key == "fct_eff":
            _refuse_fct_eff()
        if key in ("fck", "sigma_c_comp"):
            raise ValueError(
                f"'{key}' is no longer an input to ec2_stress_limits.  "
                f"Since Phase 32 this provider returns the COEFFICIENT "
                f"k_c and the strength is resolved per zone, because a "
                f"single section-wide product is checked against zones "
                f"whose materials it does not describe.  Each zone "
                f"derives f_ck from its own material; a strength GenSec "
                f"cannot derive -- f_ck(t) at transfer above all -- is "
                f"declared in 'custom_zone_strengths'."
            )
        if key not in k:
            raise ValueError(
                f"Unknown EC2 override {key!r}. "
                f"Valid keys: {sorted(k)}."
            )
        k[key] = float(val)

    if kind == "transfer":
        if not custom_zone_strengths:
            raise ValueError(
                "ec2_stress_limits('transfer') requires "
                "'custom_zone_strengths': the governing strength at "
                "transfer is f_ck(t) at the age the tendons are "
                "released, which is NOT the material's 28-day f_ck and "
                "which no zone material knows.  Deriving it from the "
                "28-day value would raise the limit above what the "
                "concrete carries on the day it is loaded -- a silent "
                "fallback in the unsafe direction.  Compute f_ck(t) "
                "(EN 1992-1-1 3.1.2(6), e.g. via the concrete's 'ec2' "
                "bridge with its 'time' argument) and declare it per "
                "zone: custom_zone_strengths={'base': {'f_ck': ...}}."
            )
        return StressLimits(
            name=f"EC2 transfer (k={k['k_transfer']})",
            k_c=k["k_transfer"],
            custom_zone_strengths=custom_zone_strengths,
            decompression=decompression,
            c_dec=k["c_dec"],
        )
    if kind == "characteristic":
        return StressLimits(
            name="EC2 characteristic",
            k_c=k["k1"],
            custom_zone_strengths=custom_zone_strengths,
            sigma_s_tens=(k["k3"] * fyk) if fyk is not None else None,
            sigma_p_max=(k["k5"] * fpk) if fpk is not None else None,
            decompression=decompression,
            c_dec=k["c_dec"],
        )
    if kind == "quasi_permanent":
        return StressLimits(
            name="EC2 quasi-permanent",
            k_c=k["k2"],
            custom_zone_strengths=custom_zone_strengths,
            decompression=decompression,
            c_dec=k["c_dec"],
        )
    raise ValueError(
        f"Unknown kind {kind!r}. Valid: 'transfer', "
        f"'characteristic', 'quasi_permanent'."
    )
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=88c28da9e7c937f6ac0ecc9f1739df75f568f0b86984cc5337350bab3e65c408 body_lines=638
