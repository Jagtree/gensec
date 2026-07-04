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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ==================================================================
#  Generic container
# ==================================================================


@dataclass(frozen=True)
class StressLimits:
    r"""
    Normative-agnostic set of SLS stress thresholds for one check.

    All stress values are **positive magnitudes** in MPa; the engine
    applies them with the GenSec sign convention (compression
    negative, tension positive):

    - concrete compression check:
      :math:`|\min_i \sigma_{c,i}| \le \sigma_{c,\mathrm{comp}}`;
    - uncracked-basis validity:
      :math:`\max_i \sigma_{c,i} \le f_{ct,\mathrm{eff}}`
      (a *basis* threshold, not a resistance check — its violation
      flags the linear-elastic uncracked solution as invalid);
    - reinforcement tension check:
      :math:`\max_i \sigma_{s,i} \le \sigma_{s,\mathrm{tens}}`;
    - prestressing-steel check:
      :math:`\max_i \sigma_{p,i} \le \sigma_{p,\max}`;
    - decompression: concrete stress non-tensile at probe points
      located ``c_dec`` from each bonded tendon toward the tensile
      side (see :func:`gensec.solver.sls._decompression_probe`).

    Every threshold is optional: ``None`` disables the corresponding
    check (the engine reports it as ``skipped``, never silently).

    Parameters
    ----------
    name : str
        Identifier of the limit set (e.g. ``"EC2 characteristic"``).
    sigma_c_comp : float or None, optional
        Concrete compressive stress limit [MPa], positive magnitude.
    fct_eff : float or None, optional
        Effective concrete tensile strength [MPa] used as the
        uncracked-basis validity threshold (EC2 §7.1(2):
        :math:`f_{ct,\mathrm{eff}} = f_{ctm}` or
        :math:`f_{ctm,fl}`).  ``None`` = basis not assessed (the
        engine reports ``basis_checked = False``).
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
        If any provided threshold is non-finite or non-positive, or
        if ``c_dec`` is negative.
    """

    name: str = ""
    sigma_c_comp: Optional[float] = None
    fct_eff: Optional[float] = None
    sigma_s_tens: Optional[float] = None
    sigma_p_max: Optional[float] = None
    decompression: bool = False
    c_dec: float = 25.0

    def __post_init__(self):
        for attr in ("sigma_c_comp", "fct_eff",
                     "sigma_s_tens", "sigma_p_max"):
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


def ec2_stress_limits(kind, *, fck=None, fyk=None, fpk=None,
                      fct_eff=None, decompression=False,
                      **overrides):
    r"""
    Build a :class:`StressLimits` set from EC2 recommended values.

    Parameters
    ----------
    kind : {'transfer', 'characteristic', 'quasi_permanent'}
        Which EC2 limit family to build:

        ``'transfer'``
            :math:`\sigma_c \le k_t\, f_{ck}(t)` (EC2 §5.10.2.2(5),
            recommended :math:`k_t = 0.6`; may be raised to 0.7 for
            pretensioned members if justified — supply
            ``k_transfer=0.7``).  ``fck`` here is the strength **at
            the time of transfer**, computed by the caller (e.g. via
            the ``ec2`` bridge attribute with its ``time`` argument)
            — never derived here.
        ``'characteristic'``
            :math:`\sigma_c \le k_1 f_{ck}`,
            :math:`\sigma_s \le k_3 f_{yk}`,
            :math:`\sigma_p \le k_5 f_{pk}` (EC2 §7.2(2), (5)).
        ``'quasi_permanent'``
            :math:`\sigma_c \le k_2 f_{ck}` (EC2 §7.2(3), linear
            creep validity).

    fck : float
        Concrete characteristic strength [MPa] — at transfer time for
        ``kind='transfer'``, at 28 days otherwise.  Required.
    fyk : float, optional
        Reinforcing-steel characteristic yield strength [MPa].
        Consumed by ``'characteristic'``; when omitted, the steel
        check is disabled (``sigma_s_tens=None``).
    fpk : float, optional
        Prestressing-steel characteristic strength [MPa].  Consumed
        by ``'characteristic'``; when omitted, the tendon check is
        disabled.
    fct_eff : float, optional
        Effective tensile strength [MPa] for the uncracked-basis
        validity threshold (typically :math:`f_{ctm}`, possibly
        time-dependent — caller-computed).  ``None`` = basis not
        assessed.
    decompression : bool, optional
        Enable the decompression check.  Default ``False``.
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
        Unknown ``kind``, unknown override key, or missing ``fck``.

    Examples
    --------
    Transfer check for a pretensioned member, strength at 5 days
    computed by the caller through the EC2 bridge:

    >>> lim = ec2_stress_limits('transfer', fck=fck_t5,
    ...                         k_transfer=0.7)   # doctest: +SKIP

    NTC 2018 quasi-permanent limit (same coefficient as EC2 in this
    case, shown as an explicit override for clarity):

    >>> lim = ec2_stress_limits('quasi_permanent', fck=30.0,
    ...                         k2=0.45)
    """
    k = dict(EC2_RECOMMENDED)
    for key, val in overrides.items():
        if key not in k:
            raise ValueError(
                f"Unknown EC2 override {key!r}. "
                f"Valid keys: {sorted(k)}."
            )
        k[key] = float(val)

    if fck is None or not np.isfinite(fck) or fck <= 0.0:
        raise ValueError(
            f"ec2_stress_limits requires an explicit, positive fck "
            f"[MPa]; got {fck!r}.  For 'transfer', pass fck(t) "
            f"computed by the caller (e.g. via the concrete's 'ec2' "
            f"bridge attribute)."
        )

    if kind == "transfer":
        return StressLimits(
            name=f"EC2 transfer (k={k['k_transfer']})",
            sigma_c_comp=k["k_transfer"] * fck,
            fct_eff=fct_eff,
            decompression=decompression,
            c_dec=k["c_dec"],
        )
    if kind == "characteristic":
        return StressLimits(
            name="EC2 characteristic",
            sigma_c_comp=k["k1"] * fck,
            fct_eff=fct_eff,
            sigma_s_tens=(k["k3"] * fyk) if fyk is not None else None,
            sigma_p_max=(k["k5"] * fpk) if fpk is not None else None,
            decompression=decompression,
            c_dec=k["c_dec"],
        )
    if kind == "quasi_permanent":
        return StressLimits(
            name="EC2 quasi-permanent",
            sigma_c_comp=k["k2"] * fck,
            fct_eff=fct_eff,
            decompression=decompression,
            c_dec=k["c_dec"],
        )
    raise ValueError(
        f"Unknown kind {kind!r}. Valid: 'transfer', "
        f"'characteristic', 'quasi_permanent'."
    )
