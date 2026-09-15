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
Abstract base class for constitutive material laws.

All concrete material implementations must inherit from :class:`Material`
and provide scalar/vectorized stress evaluation as well as tangent
modulus evaluation for the analytical Jacobian.
"""

import copy
import dataclasses
import hashlib
import struct
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


# ======================================================================
#  Content tokens — reproducible identity of declared content
# ======================================================================

#: Length in bytes of every digest produced by this module.  Sixteen
#: bytes is 128 bits: a collision between two materials that a project
#: could plausibly declare is not a risk anyone needs to reason about,
#: and the token stays short enough to embed in a JSON dump.
DIGEST_SIZE: int = 16

#: Attribute names never folded into a content token, whatever class
#: declares them.  ``name`` is a relation label assigned by the YAML
#: loader and the SLS layer, not a constitutive parameter: a section
#: built in code carries ``name == ""`` on every material, so a token
#: that read it would discriminate only along the YAML path and would
#: move whenever a material is renamed in a file.
CONTENT_EXCLUDED_NAMES: frozenset = frozenset({"name"})


def digest_bytes(*chunks: bytes) -> bytes:
    r"""
    BLAKE2b digest of the concatenated *chunks*.

    Parameters
    ----------
    *chunks : bytes
        Pre-encoded byte strings, folded in the order given.

    Returns
    -------
    bytes
        Digest of length :data:`DIGEST_SIZE`.

    Notes
    -----
    BLAKE2b is used here as a *reproducible* hash, not a
    cryptographic guarantee: what matters is that the same content
    yields the same bytes in every process, on every machine and in
    every Python build, which the interpreter's built-in ``hash`` does
    not provide for ``bytes`` or ``str``.
    """
    h = hashlib.blake2b(digest_size=DIGEST_SIZE)
    for chunk in chunks:
        h.update(chunk)
    return h.digest()


def _u32(n: int) -> bytes:
    """
    Encode a non-negative length as four little-endian bytes.

    Parameters
    ----------
    n : int

    Returns
    -------
    bytes
    """
    return struct.pack("<I", int(n))


def encode_value(value) -> bytes:
    r"""
    Encode *value* as a canonical, type-tagged byte string.

    Every supported type carries a distinct one-byte tag, and every
    variable-length payload is length-prefixed.  Consequently ``1``,
    ``1.0``, ``True`` and ``"1"`` encode to four different strings, and
    no concatenation of two values can be read as a different pair —
    which is what makes a digest over the concatenation trustworthy.

    ==================  ===  =============================================
    type                tag  payload
    ==================  ===  =============================================
    ``None``            N    (none)
    ``bool``            B    one byte, 0 or 1
    ``int``             I    length-prefixed little-endian, signed
    ``float``           F    :func:`struct.pack` ``'<d'``
    ``str``             S    length-prefixed UTF-8
    ``bytes``           Y    length-prefixed raw
    ``numpy.ndarray``   A    dtype, shape, and a digest of the buffer
    ``list``/``tuple``  T    count, then each item encoded
    ``set``             E    count, then encoded items sorted as bytes
    ``dict``            D    count, then ``(key, value)`` pairs sorted
    ``content_token``   O    the object's own token
    callable            C    length-prefixed ``module.qualname``
    ==================  ===  =============================================

    Parameters
    ----------
    value : object

    Returns
    -------
    bytes

    Raises
    ------
    TypeError
        If *value* has no canonical encoding.  Refusing is deliberate:
        a silent fallback such as ``repr`` would embed a memory
        address for most objects and reintroduce exactly the
        irreproducibility this encoding exists to remove.
    ValueError
        If *value* is an array of Python objects, or a callable that
        cannot be named across processes (a lambda, or a closure
        defined inside another function).

    Examples
    --------
    >>> encode_value(1) != encode_value(1.0)
    True
    >>> encode_value(0.0) == encode_value(-0.0)
    True
    """
    if value is None:
        return b"N"

    # bool before int: bool is a subclass of int.
    if isinstance(value, (bool, np.bool_)):
        return b"B" + (b"\x01" if value else b"\x00")

    if isinstance(value, (int, np.integer)):
        iv = int(value)
        width = max(1, (iv.bit_length() + 8) // 8)
        return b"I" + _u32(width) + iv.to_bytes(width, "little",
                                                signed=True)

    if isinstance(value, (float, np.floating)):
        fv = float(value)
        # Normalise the signed zero: -0.0 == 0.0 for every physical
        # purpose, but packs to different bytes.
        if fv == 0.0:
            fv = 0.0
        return b"F" + struct.pack("<d", fv)

    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + _u32(len(raw)) + raw

    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return b"Y" + _u32(len(raw)) + raw

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise ValueError(
                "encode_value: object-dtype arrays have no canonical "
                "content; declare the elements individually."
            )
        arr = np.ascontiguousarray(value)
        dt = str(arr.dtype).encode("ascii")
        out = [b"A", _u32(len(dt)), dt, _u32(arr.ndim)]
        out += [_u32(d) for d in arr.shape]
        out.append(digest_bytes(arr.tobytes()))
        return b"".join(out)

    if isinstance(value, (list, tuple)):
        items = [encode_value(v) for v in value]
        return b"T" + _u32(len(items)) + b"".join(items)

    if isinstance(value, (set, frozenset)):
        items = sorted(encode_value(v) for v in value)
        return b"E" + _u32(len(items)) + b"".join(items)

    if isinstance(value, dict):
        pairs = sorted((encode_value(k), encode_value(v))
                       for k, v in value.items())
        out = [b"D", _u32(len(pairs))]
        for k, v in pairs:
            out += [k, v]
        return b"".join(out)

    token = getattr(value, "content_token", None)
    if callable(token):
        return b"O" + token()

    if callable(value):
        qual = f"{getattr(value, '__module__', '?')}." \
               f"{getattr(value, '__qualname__', '?')}"
        if "<lambda>" in qual or "<locals>" in qual:
            raise ValueError(
                f"encode_value: {qual!r} cannot be identified across "
                f"processes.  A lambda or a closure has no stable "
                f"name; declare a module-level function instead."
            )
        raw = qual.encode("utf-8")
        return b"C" + _u32(len(raw)) + raw

    raise TypeError(
        f"encode_value: no canonical encoding for "
        f"{type(value).__module__}.{type(value).__qualname__}.  "
        f"Give the class a content_token() method, or declare the "
        f"content as plain values."
    )


def _class_token(cls) -> bytes:
    """
    Encode the identity of a class.

    Two classes with identical parameters are still different laws, so
    the class name is part of the content.

    Parameters
    ----------
    cls : type

    Returns
    -------
    bytes
    """
    return encode_value(f"{cls.__module__}.{cls.__qualname__}")


class Material(ABC):
    r"""
    Abstract base class for all constitutive material laws.

    Every material must expose:

    - ``stress(eps)`` — scalar stress evaluation.
    - ``stress_array(eps)`` — vectorized evaluation over an array
      of **arbitrary shape** (1-D, 2-D, …).
    - ``tangent(eps)`` — scalar tangent modulus
      :math:`E_t = \mathrm{d}\sigma/\mathrm{d}\varepsilon`.
    - ``tangent_array(eps)`` — vectorized tangent modulus over an
      array of arbitrary shape.
    - ``eps_min`` / ``eps_max`` — admissible strain range.

    The tangent modulus is used by
    :meth:`~gensec.solver.FiberSolver.integrate_with_tangent` to
    assemble the **analytical** tangent stiffness matrix, avoiding
    the cost of finite-difference Jacobians in Newton-Raphson solvers.

    The strain limits are consumed by the N-M diagram generator to
    automatically determine the scan range for ultimate configurations.

    Attributes
    ----------
    eps_min : float
        Most compressive admissible strain (typically negative).
    eps_max : float
        Most tensile admissible strain (typically positive).
    name : str
        Human-readable identifier set by the YAML loader.
    """

    name: str = ""

    #: Attribute names that are part of this material's declared
    #: content but are **not** dataclass fields.  Empty for every
    #: material shipped today.
    #:
    #: The slot exists because a material is a constitutive law in
    #: time as well as in stress, and in this tree the two halves live
    #: in different objects: the instantaneous law is the dataclass
    #: below, while creep, shrinkage, relaxation and :math:`E_c(t)`
    #: live behind :class:`RheologicalModel`, carried by
    #: :class:`~gensec.solver.losses.LossModel` and bound to a zone by
    #: the construction timeline.  A material that acquires a time law
    #: names the attribute here and :meth:`content_token` folds it in,
    #: with no change to the token machinery.
    CONTENT_EXTRA: tuple = ()

    def content_token(self) -> bytes:
        r"""
        Reproducible digest of this material's declared content.

        The token is a :data:`DIGEST_SIZE`-byte BLAKE2b digest over

        1. the fully qualified class name;
        2. every dataclass field, in declaration order, as
           ``(field name, value)`` — excluding the names in
           :data:`CONTENT_EXCLUDED_NAMES`;
        3. every attribute listed in :attr:`CONTENT_EXTRA`.

        Two materials with identical declared content produce
        identical tokens in any process, on any machine, whether they
        are the same object, two separate constructions or a deep
        copy.  Two materials that differ in any declared parameter
        produce different tokens.

        Scope
        -----
        The token covers the *declared* content, which for every
        material shipped today is the instantaneous constitutive law.
        Time-dependent behaviour is not read here because no material
        instance carries it; it lives behind
        :class:`RheologicalModel`, which has a
        :meth:`RheologicalModel.content_token` of its own.  A caller
        that needs "the material, in full" folds the two together;
        a caller that needs the ultimate-limit-state resistance — the
        capacity domain — correctly needs only this one, because creep
        does not move a ULS surface.

        Derived attributes set in ``__post_init__`` (``fcd``,
        ``eps_ct``, …) are not folded in: they are functions of the
        fields, so including them would add no discrimination and
        would make the token sensitive to a refactor of the
        derivation.

        Returns
        -------
        bytes
            Digest of length :data:`DIGEST_SIZE`.

        Raises
        ------
        NotImplementedError
            If the class is not a dataclass and does not override this
            method.  Refusing is deliberate: a fallback token computed
            from ``vars(self)`` would silently include cached
            intermediates and exclude anything held in ``__slots__``,
            and the caller would never learn that the identity it is
            relying on is not the identity it thinks it has.

        Examples
        --------
        >>> from gensec.materials.concrete import Concrete
        >>> a, b = Concrete(fck=45.0), Concrete(fck=45.0)
        >>> a.name, b.name = "C45 precast", "C45 topping"
        >>> a.content_token() == b.content_token()
        True
        >>> Concrete(fck=45.0).content_token() == \
        ...     Concrete(fck=50.0).content_token()
        False
        """
        cls = type(self)
        if not dataclasses.is_dataclass(cls):
            raise NotImplementedError(
                f"{cls.__module__}.{cls.__qualname__} is not a "
                f"dataclass, so the declared-content token cannot be "
                f"derived from its fields.  Implement "
                f"content_token() on the class, listing exactly the "
                f"parameters that define its constitutive law."
            )
        parts = [_class_token(cls)]
        for fld in dataclasses.fields(cls):
            if fld.name in CONTENT_EXCLUDED_NAMES:
                continue
            parts.append(encode_value(fld.name))
            parts.append(encode_value(getattr(self, fld.name)))
        for attr in cls.CONTENT_EXTRA:
            parts.append(encode_value(attr))
            parts.append(encode_value(getattr(self, attr)))
        return digest_bytes(*parts)

    def stress_breakpoints(self):
        r"""
        Strains at which this law's stress attains a local extremum or
        jumps [-].

        Why a law has to answer this
        ----------------------------
        A plane section makes the strain field **affine** over each
        material region, whatever the constitutive law is: the
        kinematics are linear even when the material is not.  The
        extremum of an affine function over a polygon is attained at a
        vertex, so on a straight line the extreme stress is a vertex
        value and nothing else needs asking.

        On a curve that is false.  Stress is a non-monotone function of
        strain — it peaks, and it can drop to zero on cracking — so the
        most stressed point of a region is a vertex **or** a point where
        the law itself turns over, and that second candidate is a
        property of the law and of nothing else.  Sampling the interval
        instead would reintroduce exactly the discretisation error that
        the closed-form extreme-fibre evaluation was written to remove
        (hotfix F6).

        Returns
        -------
        tuple of float
            Candidate strains, in no particular order.  Empty for a law
            whose stress is monotone over its whole admissible range,
            which is the default and is correct for a straight line.
        """
        return ()

    @property
    @abstractmethod
    def eps_min(self):
        """Most compressive admissible strain."""
        ...

    @property
    @abstractmethod
    def eps_max(self):
        """Most tensile admissible strain."""
        ...

    @abstractmethod
    def stress(self, eps):
        """
        Compute stress for a single strain value.

        Parameters
        ----------
        eps : float
            Strain.

        Returns
        -------
        float
            Stress [MPa].
        """
        ...

    def stress_array(self, eps):
        r"""
        Vectorized stress computation.

        Default implementation loops over :meth:`stress`.  Subclasses
        should override for performance.  The input may have
        **any shape** (1-D, 2-D, …); the output must have the same
        shape.

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return np.array([self.stress(e) for e in eps.ravel()]).reshape(
            eps.shape)

    # ------------------------------------------------------------------
    #  Tangent modulus — analytical Jacobian support
    # ------------------------------------------------------------------

    def tangent(self, eps):
        r"""
        Scalar tangent modulus :math:`E_t = d\sigma / d\varepsilon`.

        Default implementation uses a central finite difference with
        step :math:`h = 10^{-8}`.  Subclasses should override with the
        closed-form derivative for best accuracy and speed.

        Parameters
        ----------
        eps : float

        Returns
        -------
        float
            Tangent modulus [MPa].
        """
        h = 1e-8
        return (self.stress(eps + h) - self.stress(eps - h)) / (2.0 * h)

    def reference_modulus(self):
        r"""
        The material's **conventional service modulus**, no time axis.

        Every material class must answer *"what is your service
        stiffness when the analysis states no age?"*, because every
        design code names one and because the alternative -- the
        resolver deducing it -- is a normative choice made by the
        engine.  Under EN 1992-1-1 the answer is :math:`E_{cm}` at 28
        days (Table 3.1); under ACI 318-19 it is
        :math:`4700\sqrt{f'_c}` (§19.2.2.1, normalweight); for a steel
        it is :math:`E_s`, at no age at all.  The engine reads one
        method and never learns which code it is talking to.

        Returns
        -------
        ReferenceModulus
            ``(value [MPa], reference_age [days] or None, source)``.

        Raises
        ------
        NotImplementedError
            The base implementation.  A material that declares no
            conventional modulus refuses loudly rather than letting a
            caller invent one: no class can stay silent without it
            being seen.

        See Also
        --------
        ReferenceModulusRule : the override hook, for forcing another
            code's convention onto a material.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no conventional service "
            f"modulus.  A material used where the model states no time "
            f"axis must state one -- the value, the reference age it "
            f"belongs to, and the clause that states it.  Implement "
            f"reference_modulus(); the engine will not guess it.")

    def tangent_array(self, eps):
        r"""
        Vectorized tangent modulus.

        Default implementation uses a central finite difference.
        Subclasses should override with the closed-form derivative.
        Input may have **any shape**.

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Same shape as *eps*.
        """
        h = 1e-8
        return (self.stress_array(eps + h) - self.stress_array(eps - h)) / (
            2.0 * h)


# ======================================================================
#  Rheology — time-dependent constitutive providers
#
#  APPEND TO gensec/materials/base.py (after the Material ABC).
#  Requires, at the top of base.py:  import copy, numpy as np
#                                    from abc import ABC, abstractmethod
# ======================================================================


# ======================================================================
#  Declared validity domains — a provider states where it is defined
# ======================================================================


class UnknownQuantity(KeyError):
    r"""
    A provider was asked about a quantity name it does not speak.

    Measured in Phase 16 and recorded as N-8: ``domain('f_ctm_typo')``
    returned ``None``, and ``None`` on this surface means *"stated at
    every age"* -- the most permissive answer available.  A misspelt
    name therefore did not fail; it silently widened the declared
    domain to everything.

    The shape of that defect recurs in this project: a sentinel the
    happy path legitimately produces (``None`` for "no interval", a
    ``name`` attribute that defaults to ``None``, a ``getattr`` default
    that is also a valid value) is reused to signal the error path, so
    the error path becomes unreachable.  The repair is always the same:
    close the vocabulary, and raise outside it.

    Closing the vocabulary needs no normative text.  It is a statement
    about *which names exist*, not about where any interval lies.
    """


class OutOfDeclaredDomain(NotImplementedError):
    r"""
    A provider was asked for a quantity at an age it does not state.

    This is **not** an error in the caller's input: an age of one day is
    a perfectly legitimate age.  It is the provider declaring the edge
    of its own source.  EN 1992-1-1 §3.1.2(5) states :math:`f_{ck}(t)`
    only for :math:`t \ge 3` d; ACI 209R-92 states :math:`f'_c(t) =
    t/(a + bt)\, f'_c(28)` from :math:`t \to 0^{+}` and has no such
    edge; a tabulated provider is defined exactly over its table.  The
    edge is a property of the *source*, never of GenSec, which is why
    it is raised by the provider and carries the provider's own
    citation.

    Being a :class:`NotImplementedError` places it in the scope-boundary
    half of the exception taxonomy: the request is well formed and
    GenSec simply has no authority to answer it.

    Attributes
    ----------
    provider : str
        Name of the provider that declined.
    quantity : str
        Name of the quantity that was asked for.
    t : float
        Age at which it was asked [days].
    domain : tuple of float
        The provider's declared validity interval
        ``(t_min, t_max)`` [days].
    source : str
        The provider's citation for that interval.

    See Also
    --------
    RheologicalModel.domain : declared interval, per quantity.
    DeclaredRule : how the engineer supplies a rule of their own.
    """

    def __init__(self, provider, quantity, t, domain, source):
        self.provider = str(provider)
        self.quantity = str(quantity)
        self.t = float(t)
        self.domain = (float(domain[0]), float(domain[1]))
        self.source = str(source)
        lo, hi = self.domain
        hi_txt = "inf" if hi == float("inf") else f"{hi:g}"
        super().__init__(
            f"{self.provider} does not state {self.quantity!r} at "
            f"t = {self.t:g} d.  Its declared validity is "
            f"[{lo:g}, {hi_txt}] d ({self.source}).  Outside it this "
            f"provider has no value to give and GenSec will not invent "
            f"one.  Declare a rule of your own — "
            f"declared={{{self.quantity!r}: DeclaredRule(value=..., "
            f"domain=(...), source='...')}} — citing whatever you are "
            f"taking it from, or keep the analysis inside the declared "
            f"range."
        )


class NormativeChoiceNotDeclared(NotImplementedError):
    r"""
    A quantity was asked of a material that never named a code.

    GenSec is normative-agnostic by construction: EN 1992-1-1 and
    NTC 2018 are the shipped defaults, not the mechanism.  A material
    declared with nothing but its strength has therefore declared an
    **input**, not a code, and a quantity that different codes derive
    differently from that input has no code-free answer.

    The conventional secant modulus of concrete is the standing
    example.  From one :math:`f_{ck}`:

    .. math::

        E_{cm} = 22\,000\left(\frac{f_{ck} + 8}{10}\right)^{0.3}
        \qquad \text{EN 1992-1-1:2004 §3.1.3, Table 3.1}

    .. math::

        E_{c} = 4700\sqrt{f'_{c}}
        \qquad \text{ACI 318-19 §19.2.2.1, normalweight}

    Measured on the same input, these disagree by 33.9 % at
    :math:`f_{ck} = 25`, 18.5 % at 40 and 7.4 % at 60 MPa.  There is no
    reading of "the modulus of a C40" that is true under both.

    Why this must be an exception rather than a default
    ---------------------------------------------------
    A default here would not merely be a guess; it would be an
    **unoverridable** guess.  The resolver
    (:class:`~gensec.materials.base.ReferenceModulusRule`) asks the
    material *before* applying the build's rule, precisely so that a
    material which knows its own code wins over a generic one.  A
    material that answers when it does not know therefore beats the
    rule the engineer installed: measured, an ACI rule asked for
    ``Concrete(fck=40)`` returned the Eurocode's 35 220.5 MPa and not
    ACI's 29 725.4 MPa.  A default that cannot be overridden is not a
    default.

    Being a :class:`NotImplementedError` places it in the
    scope-boundary half of the exception taxonomy, beside
    :class:`OutOfDeclaredDomain` and
    ``LoadBetweenJacksNotModelled``: the request is well formed and
    GenSec has no authority to answer it.  It also means the resolver's
    ``except (NotImplementedError, AttributeError)`` catches it, so a
    Eurocode build falls through to the Eurocode rule and sees no
    change at all.

    Attributes
    ----------
    subject : str
        What was asked, e.g. ``'reference modulus'``.
    material : str
        Class name of the material that declined.
    remedy : str
        The correction lever, stated rather than implied.

    See Also
    --------
    OutOfDeclaredDomain : the same taxonomy, for an age outside a
        provider's declared interval.
    ReferenceModulusRule : the replaceable rule the resolver applies.
    """

    def __init__(self, message, subject="reference modulus",
                 material="Material", remedy=None):
        self.subject = str(subject)
        self.material = str(material)
        self.remedy = str(remedy) if remedy is not None else (
            "declare the material with a code-specific type, or "
            "install the rule for the code you are verifying to.")
        super().__init__(str(message))


class DeclaredRule:
    r"""
    An engineer-supplied rule for one quantity over one age interval.

    The escape hatch from :class:`OutOfDeclaredDomain`, and the only
    one.  It lives on :class:`RheologicalModel` rather than on any
    particular provider, so that a rule taken from a national annex, a
    later code, a research paper or a lab campaign enters the same way
    whichever provider it is extending.

    A rule **must** cite its source.  A number whose provenance is not
    recorded is indistinguishable from a number GenSec invented, and
    the whole point of the mechanism is that GenSec never invents one.
    The citation travels into :meth:`RheologicalModel.content_token`,
    so two analyses that differ only in a declared rule are correctly
    two different analyses.

    Parameters
    ----------
    value : float or callable
        The quantity, either constant over the interval or as
        ``f(t) -> float`` with *t* in days.
    source : str
        Non-empty citation: the document, clause, dataset or judgement
        the value comes from.  Recorded verbatim.
    domain : tuple of float, optional
        Ages ``(t_min, t_max)`` [days] over which the rule applies.
        Default: every age.  A rule may deliberately be narrower than
        the gap it fills.

    Raises
    ------
    ValueError
        If *source* is empty, if *domain* is not ordered, or if *value*
        is neither a real number nor callable.

    Examples
    --------
    Extending EN 1992-1-1 below the three-day floor of §3.1.2(5):

    >>> DeclaredRule(value=1.0, domain=(0.0, 3.0),
    ...              source="project judgement, outside EN 1992-1-1 "
    ...                     "§3.1.2(5)")           # doctest: +ELLIPSIS
    DeclaredRule(value=1.0, domain=(0.0, 3.0), source='project ...')

    Notes
    -----
    Precedence is the same as everywhere else in GenSec: an explicit
    declaration wins over the provider's own value wherever the two
    overlap, and a declaration that covers no age used by the analysis
    is reported rather than ignored.  An override that can be silently
    disregarded is a silent error.
    """

    __slots__ = ("value", "source", "domain")

    def __init__(self, value, source, domain=None):
        src = "" if source is None else str(source).strip()
        if not src:
            raise ValueError(
                "DeclaredRule: source is mandatory and must be "
                "non-empty.  Record where the value comes from — a "
                "clause, a paper, a dataset, or an explicit statement "
                "that it is your own engineering judgement.  A number "
                "without provenance cannot be told apart from one "
                "GenSec invented."
            )
        if callable(value):
            val = value
        else:
            try:
                val = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"DeclaredRule: value must be a real number or a "
                    f"callable f(t) -> float, got {value!r}."
                ) from None
        dom = (-float("inf"), float("inf")) if domain is None else (
            float(domain[0]), float(domain[1]))
        if not dom[0] <= dom[1]:
            raise ValueError(
                f"DeclaredRule: domain must be ordered, got {dom!r}."
            )
        object.__setattr__(self, "value", val)
        object.__setattr__(self, "source", src)
        object.__setattr__(self, "domain", dom)

    def __setattr__(self, name, value):        # frozen after __init__
        raise AttributeError(
            "DeclaredRule is immutable; build a new one instead."
        )

    def covers(self, t):
        r"""
        Whether this rule applies at age *t* [days].

        Parameters
        ----------
        t : float

        Returns
        -------
        bool
        """
        return self.domain[0] <= float(t) <= self.domain[1]

    def at(self, t):
        r"""
        The declared value at age *t* [days].

        Parameters
        ----------
        t : float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the rule does not cover *t*, or if a callable rule
            returns a non-finite value.
        """
        if not self.covers(t):
            raise ValueError(
                f"DeclaredRule({self.source!r}) does not cover "
                f"t = {float(t):g} d; its domain is {self.domain!r}."
            )
        out = float(self.value(float(t))) if callable(self.value) \
            else float(self.value)
        if not np.isfinite(out):
            raise ValueError(
                f"DeclaredRule({self.source!r}) returned a non-finite "
                f"value at t = {float(t):g} d."
            )
        return out

    def content_parts(self):
        r"""
        Byte fragments folded into a provider's content token.

        A callable rule cannot be digested by value, so its identity is
        taken from its qualified name; two different callables sharing
        a name are the caller's ambiguity to resolve, and the mandatory
        :attr:`source` is what actually distinguishes them in practice.

        Returns
        -------
        tuple of bytes
        """
        if callable(self.value):
            tag = getattr(self.value, "__qualname__",
                          type(self.value).__name__)
            val = f"callable:{tag}"
        else:
            val = self.value
        return (encode_value(val), encode_value(self.source),
                encode_value(self.domain[0]), encode_value(self.domain[1]))

    def __eq__(self, other):
        if not isinstance(other, DeclaredRule):
            return NotImplemented
        return (self.value == other.value
                and self.source == other.source
                and self.domain == other.domain)

    def __hash__(self):
        val = (self.value if not callable(self.value)
               else getattr(self.value, "__qualname__", "callable"))
        return hash((val, self.source, self.domain))

    def __repr__(self):
        val = (self.value if not callable(self.value)
               else getattr(self.value, "__qualname__", "callable"))
        return (f"DeclaredRule(value={val!r}, domain={self.domain!r}, "
                f"source={self.source!r})")


class ReferenceModulus(NamedTuple):
    r"""
    A conventional service modulus, with its provenance.

    Attributes
    ----------
    value : float
        The modulus [MPa].
    reference_age : float or None
        The age [days] the value belongs to; ``None`` for a quantity
        with no age (a steel's :math:`E_s`).
    source : str
        The clause or document that states it.  A modulus whose
        provenance is not recorded cannot be told apart from one
        GenSec invented.
    """

    value: float
    reference_age: object
    source: str


class ReferenceModulusRule(ABC):
    r"""
    A code's conventional service modulus, used **only** where the
    model declares no time axis for the zone.

    The rule this class exists for, stated once so it is not re-derived
    at every call site:

    * a datum is a service fact and needs a service modulus;
    * when a zone declares a rheological provider, that provider owns
      the modulus and the *age* is mandatory -- prestress transfer
      happens at 3--7 days and a silent 28-day substitute there is the
      exact defect Phase 12 exists to remove;
    * when the model declares **no** time axis at all, there is no age
      to get wrong.  The model has one instant, and every code names a
      conventional secant modulus for it.  Under EN 1992-1-1 that is
      :math:`E_{cm}` at 28 days (Table 3.1); under another code it is
      another number at another reference age, which is why this is an
      abstract rule with a pluggable implementation and not a constant
      in the resolver.

    A rule **must** cite its source, for the same reason
    :class:`DeclaredRule` must: a modulus whose provenance is not
    recorded cannot be told apart from one GenSec invented.  The
    difference between this and a silent default is not the number --
    it is that the number arrives named, cited and replaceable.

    Parameters
    ----------
    source : str
        Non-empty citation: the clause or document that states the
        conventional modulus and its reference age.
    reference_age : float or None, optional
        The age [days] the conventional modulus is stated at, recorded
        for reporting.  ``None`` for a code that states the modulus
        without naming an age.

    Raises
    ------
    ValueError
        If *source* is empty.

    See Also
    --------
    DeclaredRule : the engineer-supplied escape hatch, same discipline.
    gensec.solver.sls.DatumPolicy : the only consumer.

    Notes
    -----
    Scope is deliberately narrow.  This rule is **not** consulted when
    the model has a time axis: a zone with no provider inside a
    time-dependent analysis is an incomplete declaration, not a zone
    without an age (finding F-A).  Widening it there would re-open the
    branch Phase 12 closed.
    """

    def __init__(self, source, reference_age=None):
        src = "" if source is None else str(source).strip()
        if not src:
            raise ValueError(
                "ReferenceModulusRule: source is mandatory and must be "
                "non-empty.  A conventional modulus that does not say "
                "which document states it is a silent default wearing "
                "the clothes of a declared one."
            )
        self.source = src
        self.reference_age = (None if reference_age is None
                              else float(reference_age))

    @abstractmethod
    def E_c(self, material):
        r"""
        Conventional service modulus of *material* [MPa], or ``None``.

        Parameters
        ----------
        material : Material
            The bulk material of the zone being resolved.

        Returns
        -------
        float or None
            ``None`` means *"this rule does not describe this
            material"* -- the slot then falls through to the resolver's
            own refusal, which is the correct outcome and not this
            rule's business to pre-empt.
        """

    def __repr__(self):
        return (f"{type(self).__name__}("
                f"reference_age={self.reference_age!r}, "
                f"source={self.source!r})")


#: Immutable empty mapping shared by every provider that declares no
#: rules, so that the common case allocates nothing.
_NO_DECLARATIONS = {}


class RheologicalModel(ABC):
    r"""
    Normative-agnostic time-dependent constitutive provider.

    Supplies the four material functions the age-adjusted viscoelastic
    section container (:mod:`gensec.solver.losses`) needs.  **All**
    normative content — EN 1992-1-1, NTC 2018, ACI 209R, *fib* MC2010,
    B4, a national annex, a lab campaign — lives behind this interface.
    The container consumes only these four functions and derives
    everything else, including the ageing coefficient :math:`\chi`,
    from them.

    The four functions
    ------------------
    ========================  ====================================
    :meth:`J`                 creep compliance [1/MPa]
    :meth:`eps_imposed`       stress-independent strain [-]
    :meth:`relaxation`        intrinsic steel relaxation [MPa]
    :meth:`E_c`               concrete modulus at age *t* [MPa]
    ========================  ====================================

    Units and signs, fixed at this boundary
    ---------------------------------------
    - Time in **days**, length in **mm**, stress in **MPa**, force in
      **N**.  A provider authored in other units (ACI 209R in psi and
      inches) converts *internally*: unit leakage is the **provider's**
      responsibility, never the container's.
    - Strain is **signed** in GenSec's structural convention:
      shortening is **negative**.  Shrinkage therefore returns a
      *negative* :meth:`eps_imposed` — not the positive magnitude that
      EN 1992-1-1 §3.1.4 tabulates.  This is the sign mapping the
      Phase-5 bulk kernel deliberately left open
      (``7_1-GENSEC_PHASE5_BULK_KERNEL_PATCH.md`` §9); it is fixed
      here, and the container flips it once, explicitly, when it forms
      the bulk offset (:math:`\varepsilon_{\text{bulk}} =
      -\varepsilon_{\mathrm{imp}}`, because the kernel *adds* the
      offset while the physics *subtracts* the eigenstrain).
    - :meth:`relaxation` likewise returns the **signed** stress change
      of the tendon: a decay, hence :math:`\le 0`.

    Geometry
    --------
    The container hands geometry as the pair :math:`(A_c, u)` — the
    concrete area [mm²] and the perimeter exposed to drying [mm] — and
    **never** as a code-specific nominal size.  Each provider forms its
    own measure: EN 1992-1-1 the notional size :math:`h_0 = 2A_c/u`,
    ACI 209R the volume/surface ratio :math:`V/S = A_c/u` (a factor of
    two apart, and in different units).  Binding is **late**
    (:meth:`with_geometry`) because in a composite section every
    concrete zone has its own exposed perimeter, hence its own notional
    size, while the *material* parameters are shared.

    Notes
    -----
    The creep coefficient the container works with is **derived** from
    the compliance, never asked of the provider:

    .. math::

        \varphi(t, t') \;=\; E_c(t')\, J(t, t') \;-\; 1

    (:meth:`phi`).  This is an identity for any provider.  For an EC2
    provider at :math:`t' = 28` d it reproduces the Eurocode
    :math:`\varphi(t,t_0)` exactly; for a B4 provider it returns
    whatever B4's multi-term compliance implies.  A norm that has no
    creep coefficient still has one *implied* by its compliance, and
    that is the number the AAEM equation needs.

    See Also
    --------
    aging_coefficient : :math:`\chi(t,t_0)` computed **from** ``J``.
    relaxation_function : :math:`R(t,t_0)` computed **from** ``J``.
    gensec.materials.rheology : the shipped providers.
    """

    #: Human-readable identifier.
    name: str = ""
    #: Concrete area of the bound zone [mm²] (``None`` until bound).
    A_c = None
    #: Perimeter of the bound zone exposed to drying [mm].
    u = None
    #: Characteristic strength of the bound prestressing steel [MPa].
    f_pk = None

    #: Attribute names that constitute this provider's declared
    #: content, in the order they are folded into
    #: :meth:`content_token`.  A provider that leaves this empty has
    #: no token, by its own declaration.
    CONTENT_FIELDS: tuple = ()

    #: Age interval over which this provider states each quantity, as
    #: ``{name: (t_min, t_max)}`` in days.  A quantity absent from the
    #: mapping is stated at **every** age.  This is *data* — a national
    #: annex may move an interval without the logic changing — and it
    #: is the provider's own, never GenSec's.
    DOMAIN: dict = {}

    #: Closed vocabulary of quantity names this provider speaks.
    #:
    #: Every name accepted by :meth:`domain`, :meth:`domain_source`,
    #: :meth:`states` and :meth:`_guarded` must appear here.  A name
    #: outside it raises :class:`UnknownQuantity` instead of returning
    #: the permissive ``None`` (N-8).
    #:
    #: The base set is the container's own invariant surface -- the
    #: four material functions the age-adjusted solver consumes.  A
    #: normative subclass extends it with the quantities that norm
    #: names; it never shrinks it, because the container calls these
    #: four whatever the provider is.
    QUANTITIES: frozenset = frozenset({
        "J", "eps_imposed", "relaxation", "E_c", "linearity_limit",
    })

    #: Citation for each entry of :attr:`DOMAIN`, as ``{name: str}``.
    #: Every declared interval must be attributable to a document.
    DOMAIN_SOURCE: dict = {}

    #: Engineer-supplied rules, as ``{quantity: DeclaredRule}``.  Empty
    #: unless :meth:`with_declared` was used.
    declared: dict = _NO_DECLARATIONS

    # ------------------------------------------------------------------
    #  Declared validity domain
    # ------------------------------------------------------------------

    def domain(self, quantity):
        r"""
        Age interval over which this provider states *quantity*.

        Instance-level intervals (a tabulated provider's grid, which
        differs per object) take precedence over the class-level
        :attr:`DOMAIN`.

        Parameters
        ----------
        quantity : str
            Name of the quantity, e.g. ``'linearity_limit'``.

        Returns
        -------
        tuple of float or None
            ``(t_min, t_max)`` in days, or ``None`` when the provider
            states the quantity at every age.

        See Also
        --------
        states : whether a given age is inside that interval.
        """
        self._check_quantity(quantity)
        own = getattr(self, "_domain", None)
        if own and quantity in own:
            return tuple(float(v) for v in own[quantity])
        if quantity in type(self).DOMAIN:
            return tuple(float(v) for v in type(self).DOMAIN[quantity])
        return None

    def _check_quantity(self, quantity):
        r"""
        Reject a quantity name outside this provider's vocabulary.

        Parameters
        ----------
        quantity : str

        Raises
        ------
        UnknownQuantity
            When *quantity* is neither in :attr:`QUANTITIES`, nor in
            :attr:`DOMAIN`, nor in this instance's own interval map.
        """
        cls = type(self)
        if quantity in cls.QUANTITIES or quantity in cls.DOMAIN:
            return
        # A provider that *exposes* a quantity speaks its name, whether
        # or not it declared an interval for it.  Without this a
        # third-party provider would have to repeat every method name in
        # QUANTITIES to be usable, and a vocabulary that is expensive to
        # extend is a vocabulary people route around.  A typo is still
        # rejected: it is neither declared nor exposed.
        if hasattr(cls, quantity):
            return
        own = getattr(self, "_domain", None)
        if own and quantity in own:
            return
        known = sorted(set(cls.QUANTITIES) | set(cls.DOMAIN)
                       | set(own or ())
                       | {n for n in dir(cls) if not n.startswith("_")})
        raise UnknownQuantity(
            f"{type(self).__name__} does not speak the quantity "
            f"{quantity!r}.  Known: {known}.  Returning 'no declared "
            f"interval' for an unknown name would state the quantity "
            f"at every age, which is the most permissive answer "
            f"available (N-8).")

    @classmethod
    def check_vocabulary(cls):
        r"""
        Internal-consistency check on this provider's declarations.

        Cheap, so it runs from the test suite on every provider rather
        than behind a flag.  Three conditions, each of which has an
        instance in this tree's history:

        1. every :attr:`DOMAIN` key is in :attr:`QUANTITIES` -- a
           declared interval for a name the provider does not speak is
           unreachable through :meth:`domain`;
        2. every :attr:`DOMAIN_SOURCE` key is a :attr:`DOMAIN` key -- a
           citation without an interval cites nothing;
        3. the names in :attr:`DOMAIN` that have **no attribute** on the
           class are reported.  This is not an error: a provider may
           legitimately declare where a norm states a quantity that it
           computes internally and does not surface.  It is reported
           because in this tree three such names
           (``f_ck``, ``f_cm``, ``f_ctm``) were published by the dump as
           interrogable and were not, and nobody could see it (N-8).

        Returns
        -------
        dict
            ``{"orphan_domain", "orphan_source", "declared_no_surface"}``
            each a sorted list of names.

        Raises
        ------
        AssertionError
            On conditions 1 or 2.
        """
        orphan_domain = sorted(set(cls.DOMAIN) - set(cls.QUANTITIES))
        orphan_source = sorted(set(cls.DOMAIN_SOURCE) - set(cls.DOMAIN))
        assert not orphan_domain, (
            f"{cls.__name__}.DOMAIN declares intervals for names outside "
            f"QUANTITIES: {orphan_domain}")
        assert not orphan_source, (
            f"{cls.__name__}.DOMAIN_SOURCE cites names that have no "
            f"declared interval: {orphan_source}")
        no_surface = sorted(n for n in cls.DOMAIN if not hasattr(cls, n))
        return {"orphan_domain": orphan_domain,
                "orphan_source": orphan_source,
                "declared_no_surface": no_surface}

    def domain_source(self, quantity):
        r"""
        Citation for this provider's declared interval for *quantity*.

        Parameters
        ----------
        quantity : str

        Returns
        -------
        str
            The citation, or a plain statement that none is declared.
        """
        self._check_quantity(quantity)
        own = getattr(self, "_domain_source", None)
        if own and quantity in own:
            return str(own[quantity])
        return str(type(self).DOMAIN_SOURCE.get(
            quantity, "no citation declared"))

    def states(self, quantity, t):
        r"""
        Whether this provider can give *quantity* at age *t* [days].

        True when an engineer-declared rule covers *t*, or when *t*
        falls inside the provider's own interval, or when the provider
        declares no interval at all.

        Parameters
        ----------
        quantity : str
        t : float

        Returns
        -------
        bool
        """
        self._check_quantity(quantity)
        rule = self.declared.get(quantity)
        if rule is not None and rule.covers(t):
            return True
        dom = self.domain(quantity)
        return True if dom is None else dom[0] <= float(t) <= dom[1]

    def with_declared(self, **rules):
        r"""
        Return a **copy** carrying engineer-supplied rules.

        Late binding, like :meth:`with_geometry`: the provider object
        is never mutated, so a rule declared for one zone cannot leak
        into another that happens to share the material.

        Parameters
        ----------
        **rules : DeclaredRule
            One rule per quantity name.

        Returns
        -------
        RheologicalModel
            A copy whose :attr:`declared` carries *rules* merged over
            any already present.

        Raises
        ------
        TypeError
            If a value is not a :class:`DeclaredRule`.

        Examples
        --------
        >>> m = provider.with_declared(          # doctest: +SKIP
        ...     linearity_limit=DeclaredRule(
        ...         value=1.0, domain=(0.0, 3.0),
        ...         source="project judgement, outside EN 1992-1-1 "
        ...                "§3.1.2(5)"))
        """
        for name, rule in rules.items():
            if not isinstance(rule, DeclaredRule):
                raise TypeError(
                    f"with_declared: {name!r} must be a DeclaredRule, "
                    f"got {type(rule).__name__}.  The wrapper is what "
                    f"carries the mandatory source citation."
                )
        out = copy.copy(self)
        merged = dict(self.declared)
        merged.update(rules)
        out.declared = merged
        return out

    def _stated(self, quantity, t, compute):
        r"""
        Resolve *quantity* at age *t*: declared, else provider, else raise.

        The precedence is GenSec's single override rule, applied here
        exactly as :func:`~gensec.solver.sls.resolve_moduli` applies it
        to moduli: an explicit declaration wins wherever it overlaps
        the provider, because an override that can be silently
        disregarded is a silent error.

        Parameters
        ----------
        quantity : str
            Name, which must appear in :attr:`DOMAIN` or in
            :attr:`declared` for the guard to have anything to check.
        t : float
            Age [days].
        compute : callable
            ``f(t) -> float``, the provider's own evaluation.  Called
            only when no declaration covers *t* and *t* is inside the
            provider's interval.

        Returns
        -------
        float

        Raises
        ------
        OutOfDeclaredDomain
            When *t* lies outside the provider's interval and no
            declared rule covers it.
        """
        rule = self.declared.get(quantity)
        if rule is not None and rule.covers(t):
            return rule.at(t)
        dom = self.domain(quantity)
        if dom is not None and not dom[0] <= float(t) <= dom[1]:
            raise OutOfDeclaredDomain(
                provider=self.name or type(self).__name__,
                quantity=quantity, t=t, domain=dom,
                source=self.domain_source(quantity))
        return float(compute(float(t)))

    def content_token(self) -> bytes:
        r"""
        Reproducible digest of this provider's declared content.

        The time-dependent half of a material is a material parameter
        set like any other: an EC2 provider at ``RH = 50 %`` and the
        same provider at ``RH = 80 %`` are different laws and must not
        share an identity.  Because providers are late-bound to a zone
        (:meth:`with_geometry`) and to a tendon
        (:meth:`with_prestressing_steel`), the bound geometry
        :math:`(A_c, u)` and :math:`f_{pk}` are content too — an
        unbound provider and the same provider bound to a 420 mm
        notional size produce different tokens, as they must, since
        they produce different creep.

        Nothing consumes this token inside the solver today: the
        capacity domain is a ULS surface and creep does not move it.
        It exists so that the time-dependent half of a material is
        addressable by content when a construction timeline is
        serialised, rather than a second identity scheme being
        invented at that point.

        Returns
        -------
        bytes
            Digest of length :data:`DIGEST_SIZE`.

        Raises
        ------
        NotImplementedError
            If :attr:`CONTENT_FIELDS` is empty and the method is not
            overridden.
        """
        cls = type(self)
        if not cls.CONTENT_FIELDS:
            raise NotImplementedError(
                f"{cls.__module__}.{cls.__qualname__} declares no "
                f"CONTENT_FIELDS, so its content token cannot be "
                f"derived.  List the attributes that define the "
                f"provider's time-dependent law, or override "
                f"content_token()."
            )
        parts = [_class_token(cls)]
        for attr in cls.CONTENT_FIELDS:
            if attr in CONTENT_EXCLUDED_NAMES:
                continue
            parts.append(encode_value(attr))
            parts.append(encode_value(getattr(self, attr)))
        # Engineer-declared rules are content: a provider extended
        # below EN 1992-1-1's three-day floor is a different law from
        # the same provider left unextended, and the two must not share
        # an identity.  Sorted so the token does not depend on the
        # order the rules were declared in.
        for name in sorted(self.declared):
            parts.append(encode_value(f"declared:{name}"))
            parts.extend(self.declared[name].content_parts())
        return digest_bytes(*parts)

    # ------------------------------------------------------------------
    #  The four material functions
    # ------------------------------------------------------------------

    @abstractmethod
    def J(self, t, t_prime):
        r"""
        Creep compliance :math:`J(t, t')` [1/MPa] — total
        (instantaneous + delayed) strain at age *t* per unit stress
        sustained since *t'*.

        The instantaneous response is *encoded here*, not supplied
        separately: :math:`E_c(t') = 1 / J(t', t')`.

        An EN 1992-1-1 provider returns
        :math:`1/E_{cm}(t') + \varphi(t,t')/E_{cm}(28)`; an ACI 209R
        provider :math:`[1 + \varphi(t,t')]/E_c(t')`; a B4 provider a
        genuine multi-term compliance.  **The container never assumes
        the** :math:`[1+\varphi]/E_c` **form.**

        Parameters
        ----------
        t : float
            Age at observation [days], ``>= t_prime``.
        t_prime : float
            Age at loading [days].

        Returns
        -------
        float
            Compliance [1/MPa].
        """

    @abstractmethod
    def eps_imposed(self, t, t_s):
        r"""
        Stress-independent imposed strain
        :math:`\varepsilon_{\mathrm{imp}}(t, t_s)` [-], **signed**
        (negative for shrinkage).

        The container sees only the *total*; the split — EC2's drying +
        autogenous, ACI 209R's single curve, B4's explicit autogenous
        term — is internal to the provider.

        Parameters
        ----------
        t : float
            Age at observation [days].
        t_s : float
            Age at the end of curing / start of drying [days].

        Returns
        -------
        float
            Imposed strain [-].
        """

    @abstractmethod
    def relaxation(self, t, mu):
        r"""
        **Intrinsic** steel relaxation :math:`\Delta\sigma_{pr}(t,\mu)`
        [MPa], **signed** (negative).

        Stress decay of a tendon held at **constant strain**, with
        :math:`\mu = \sigma_{pi}/f_{pk}` the initial stress ratio.

        *Intrinsic* is load-bearing: the reduction caused by the tendon
        shortening with the concrete — the lump factor 0.8 of
        EN 1992-1-1 (5.46) — is the **container's** coupled equilibrium
        and is applied there
        (:attr:`~gensec.solver.losses.LossModel.relaxation_reduction`).
        It must not be baked in here.

        Parameters
        ----------
        t : float
            Duration under load [days] — not an age: relaxation is
            measured from stressing.
        mu : float
            :math:`\sigma_{pi}/f_{pk}` [-].

        Returns
        -------
        float
            Stress change [MPa], negative.
        """

    @abstractmethod
    def E_c(self, t):
        r"""
        Concrete modulus at age *t* [MPa].

        Consistent with :meth:`J` by the identity
        :math:`E_c(t) = 1/J(t,t)`; implementations may return the
        closed form directly, and the validator asserts the identity.

        Parameters
        ----------
        t : float
            Age [days].

        Returns
        -------
        float
            Modulus [MPa].
        """

    @abstractmethod
    def service_law(self, t):
        r"""
        Service constitutive law of the concrete at age *t*.

        **Rheology is normative, and so is the law it hands back.**  A
        Eurocode provider does not answer this question the way an
        ACI 318-19 provider does, and neither is a house convention:
        each returns the law its own document writes, built from the
        constants its own document defines.  The container asks for
        *the material at this age* and never learns which clause
        produced it, which is what keeps GenSec agnostic while every
        number in it stays attributable.

        This is the fifth abstract function, and it is abstract for the
        same reason the other four are: a provider that cannot say what
        its concrete *does* under strain is not a provider.  A
        rheology suspended over a normative vacuum has nothing to
        return here and must **refuse by name** rather than borrow
        another code's answer -- borrowing is exactly the silent
        normative choice this protocol exists to prevent.

        Two properties the returned material must have, and which no
        implementation may quietly drop:

        * every constant is evaluated **at** *t*, not at 28 days.  A
          non-linear law ages in more than one constant -- stiffness,
          strength and the strain at peak all move -- which is why this
          returns a material and not a modulus;
        * the tensile strength it stops at is the one **the provider's
          own code** nominates.  It is not a parameter of this call and
          it is not a default: it is a derivation owned by a declared
          normative document, in the sense of the code declaration
          (:class:`~gensec.materials.codes.NormativeCode`).

        Parameters
        ----------
        t : float
            Age [days].

        Returns
        -------
        Material
            A material whose ``stress_array`` / ``tangent_array`` answer
            a strain at age *t*.

        Raises
        ------
        NotImplementedError
            If the provider declares no normative document that owns a
            constitutive law -- a compliance table, for instance, is a
            code that is only data and nominates no strength.
        ValueError
            If *t* falls outside the domain the provider's source
            claims.
        """

    # ------------------------------------------------------------------
    #  Derived — provider-agnostic, never overridden by a norm
    # ------------------------------------------------------------------

    def phi(self, t, t_prime):
        r"""
        Generalized creep coefficient
        :math:`\varphi(t,t') = E_c(t')\,J(t,t') - 1` [-], derived from
        :meth:`J`.

        Parameters
        ----------
        t, t_prime : float
            Ages [days].

        Returns
        -------
        float
        """
        return self.E_c(t_prime) * self.J(t, t_prime) - 1.0

    def linearity_limit(self, t):
        r"""
        Compressive stress magnitude [MPa, positive] above which the
        provider's compliance ceases to be stress-independent, or
        ``None`` if the provider does not declare one.

        Linear viscoelasticity — the entire :math:`J`/:math:`\chi`/AAEM
        machinery — is valid only below this stress.  EN 1992-1-1
        §3.1.4(4) puts it at :math:`0.45\, f_{ck}(t_0)`, beyond which a
        *non-linear* creep coefficient is required.  The container
        checks the concrete stress against this value and **raises**
        rather than apply linear creep outside its range: a silent
        extrapolation here is a silent mismodel.

        Parameters
        ----------
        t : float
            Age at loading [days].

        Returns
        -------
        float or None
            Limit stress [MPa, positive magnitude].
        """
        return None

    # ------------------------------------------------------------------
    #  Late binding — geometry (per zone) and steel (per tendon)
    # ------------------------------------------------------------------

    def with_geometry(self, A_c, u):
        r"""
        Return a **copy** bound to the drying geometry
        :math:`(A_c, u)`.

        The receiver is left unchanged: one provider instance is shared
        across the concrete zones that use the same mix, and each zone
        binds its own exposed perimeter.

        Parameters
        ----------
        A_c : float
            Concrete area of the zone [mm²], ``> 0``.
        u : float
            Perimeter of the zone exposed to drying [mm], ``> 0``.

        Returns
        -------
        RheologicalModel

        Raises
        ------
        ValueError
            Non-positive area or perimeter.
        """
        A_c = float(A_c)
        u = float(u)
        if A_c <= 0.0 or u <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.with_geometry: A_c ({A_c}) and u "
                f"({u}) must both be positive.  A zone with no exposed "
                f"perimeter cannot dry — model it with an explicit "
                f"zero-shrinkage provider rather than u = 0."
            )
        out = copy.copy(self)
        out.A_c = A_c
        out.u = u
        return out

    def with_steel(self, f_pk, **kwargs):
        r"""
        Return a **copy** bound to a prestressing steel.

        Parameters
        ----------
        f_pk : float
            Characteristic tensile strength of the tendon [MPa].
        **kwargs
            Provider-specific overrides (relaxation class, ρ₁₀₀₀, …).

        Returns
        -------
        RheologicalModel
        """
        out = copy.copy(self)
        out.f_pk = float(f_pk)
        for k, v in kwargs.items():
            if v is not None:
                setattr(out, k, v)
        return out

    def _require_geometry(self, what):
        r"""Fail loud on an unbound drying geometry."""
        if self.A_c is None or self.u is None:
            raise ValueError(
                f"{type(self).__name__}.{what}: the drying geometry "
                f"(A_c, u) is unbound.  The container binds it per "
                f"concrete zone with with_geometry(A_c, u); a provider "
                f"used bare must be given the geometry at construction."
            )

    def _require_steel(self, what):
        r"""Fail loud on an unbound prestressing steel."""
        if self.f_pk is None:
            raise ValueError(
                f"{type(self).__name__}.{what}: f_pk is unbound.  The "
                f"container binds the prestressing steel per tendon with "
                f"with_steel(f_pk); a provider used bare must be given "
                f"it explicitly."
            )


def relaxation_function(model, t, t0, n_steps=100):
    r"""
    Relaxation function :math:`R(t, t_0)` [MPa] from the provider's
    compliance, by numerical inversion of the Volterra equation.

    :math:`R(t,t_0)` is the stress at age *t* produced by a **unit
    strain** imposed at :math:`t_0` and held constant — the solution of

    .. math::

        \int_{t_0}^{t} J(t, \tau)\,
        \frac{\partial R(\tau, t_0)}{\partial \tau}\,\mathrm{d}\tau
        \;+\; J(t, t_0)\, R(t_0, t_0) \;=\; 1 ,
        \qquad R(t_0,t_0) = \frac{1}{J(t_0,t_0)} = E_c(t_0) .

    Discretised with the classical step-by-step scheme (Bažant 1972):
    a **logarithmic** grid :math:`t_0 = \tau_0 < \dots < \tau_n = t`,
    each stress increment applied at the geometric mid-step
    :math:`\bar\tau_k = \sqrt{\tau_{k-1}\tau_k}`, the unit-strain
    condition enforced at every node,

    .. math::

        \sigma_0\, J(\tau_i, t_0)
        + \sum_{k=1}^{i} \Delta\sigma_k\, J(\tau_i, \bar\tau_k) = 1 ,

    which is lower-triangular and solves forward in a single pass:

    .. math::

        \Delta\sigma_i = \frac{1 - \sigma_0 J(\tau_i, t_0)
            - \sum_{k<i} \Delta\sigma_k\, J(\tau_i, \bar\tau_k)}
            {J(\tau_i, \bar\tau_i)} .

    Parameters
    ----------
    model : RheologicalModel
        The provider.  Only :meth:`~RheologicalModel.J` is used.
    t : float
        Age at observation [days].
    t0 : float
        Age at which the unit strain is imposed [days].
    n_steps : int, optional
        Log-grid resolution.  Default 100.

    Returns
    -------
    float
        :math:`R(t,t_0)` [MPa].

    Raises
    ------
    ValueError
        ``t < t0``, or ``n_steps < 1``.

    Notes
    -----
    Pure mechanics: **no normative constant appears in this function.**
    It is the machine that makes :func:`aging_coefficient` — hence the
    whole container — normative-agnostic.
    """
    t = float(t)
    t0 = float(t0)
    if t < t0:
        raise ValueError(
            f"relaxation_function: t ({t}) must be >= t0 ({t0})."
        )
    if int(n_steps) < 1:
        raise ValueError("relaxation_function: n_steps must be >= 1.")
    sigma_0 = 1.0 / model.J(t0, t0)
    if t == t0:
        return sigma_0

    n = int(n_steps)
    frac = np.logspace(-4.0, 0.0, n + 1)
    frac = (frac - frac[0]) / (frac[-1] - frac[0])
    nodes = t0 + frac * (t - t0)
    nodes[0] = t0

    d_sigma = np.zeros(n, dtype=float)
    tau_bar = np.sqrt(nodes[:-1] * nodes[1:])
    for i in range(1, n + 1):
        ti = nodes[i]
        acc = sigma_0 * model.J(ti, t0)
        for k in range(1, i):
            acc += d_sigma[k - 1] * model.J(ti, tau_bar[k - 1])
        d_sigma[i - 1] = (1.0 - acc) / model.J(ti, tau_bar[i - 1])
    return float(sigma_0 + d_sigma.sum())


def aging_coefficient(model, t, t0, n_steps=100):
    r"""
    Age-adjusted **ageing coefficient** :math:`\chi(t, t_0)` [-],
    computed *from* the provider's compliance — never supplied as an
    input.

    Obtained by relaxing a unit imposed strain on
    :meth:`~RheologicalModel.J` (:func:`relaxation_function`) and
    inverting the AAEM identity:

    .. math::

        \chi(t,t_0) \;=\; \frac{E_c(t_0)}{E_c(t_0) - R(t,t_0)}
                    \;-\; \frac{1}{\varphi(t,t_0)} ,

    with :math:`\varphi(t,t_0) = E_c(t_0) J(t,t_0) - 1`
    (:meth:`~RheologicalModel.phi`).

    **This is the cardinal test of normative agnosticism.**  The
    :math:`0.8` that EN 1992-1-1 (5.46) writes twice is *what EC2's own
    compliance produces*, not a datum: an EC2 provider at
    :math:`t_0 = 28` d, :math:`t \to \infty` returns
    :math:`\chi \approx 0.80` (0.76–0.86 over
    :math:`f_{ck} \in [25,60]`, :math:`h_0 \in [170,570]` mm).  An
    ACI 209R provider returns ACI's own value (:math:`\approx 0.84` on
    the same section); a B4 provider, B4's.  A container that took
    :math:`\chi` as an *argument* would have cemented the Eurocode
    approximation into what is meant to be pure mechanics.

    Parameters
    ----------
    model : RheologicalModel
        The provider.  Only ``J`` (through ``E_c`` and ``phi``) is used.
    t : float
        Age at observation [days].
    t0 : float
        Age at loading [days].
    n_steps : int, optional
        Log-grid resolution of the Volterra inversion.  Default 100,
        for which :math:`\chi` is converged to :math:`1.4\times10^{-4}`
        (verified by the grid-convergence check in
        ``run_phase5_c5_validation.py``).

    Returns
    -------
    float
        Ageing coefficient :math:`\chi` [-], typically in
        :math:`[0.5, 1.0]`.

    Raises
    ------
    ValueError
        ``t <= t0`` (:math:`\varphi \to 0` and the :math:`1/\varphi`
        term diverges), or a provider reporting no creep at all.
    """
    t = float(t)
    t0 = float(t0)
    if t <= t0:
        raise ValueError(
            f"aging_coefficient: t ({t}) must be > t0 ({t0}); chi is "
            f"undefined at zero elapsed time (phi -> 0, so the 1/phi "
            f"term diverges)."
        )
    E0 = model.E_c(t0)
    phi = model.phi(t, t0)
    if abs(phi) < 1e-12:
        raise ValueError(
            f"aging_coefficient: the provider reports a vanishing creep "
            f"coefficient phi({t}, {t0}) = {phi:.3e}.  chi is undefined "
            f"for a non-creeping material — use the elastic modulus "
            f"directly."
        )
    R = relaxation_function(model, t, t0, n_steps=n_steps)
    return float(E0 / (E0 - R) - 1.0 / phi)
