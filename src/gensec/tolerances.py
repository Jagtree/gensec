# ---------------------------------------------------------------------------
#  GenSec — Copyright (C) 2024-2026 Andrea
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero
#  General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
r"""
Numerical tolerances, in one place.

Scope, stated first because it is what keeps this module
normative-agnostic: **only numerical tolerances live here.**  A limit
that a code *states* -- :math:`0.45\,f_{ck}` for linear
viscoelasticity, a partial factor, a minimum reinforcement ratio -- is
not a tolerance.  It is a declared rule, it belongs to a provider or to
a :class:`~gensec.materials.base.DeclaredRule` with its citation, and
putting it here would create a house in which a norm can be changed
without naming it.  The test is simple: if halving the number changes
what the analysis *means* rather than how precisely it is computed, it
does not belong in this file.

Every tolerance is resolved with the precedence

.. math::

    \text{input document} \;\rightarrow\;
    \text{configuration file} \;\rightarrow\;
    \text{shipped default}

so that a model states its own tolerances where it can, a site or a
harness states the ones no input document can name (validators,
fingerprints, the ``run_*`` drivers), and neither has to edit the
engine.

Most tolerances are **relative**.  The exceptions are the quantities
whose correct value is zero -- a cancellation residue, a centroid
offset -- because a relative tolerance on a quantity that must vanish
is not a tolerance at all.

Examples
--------
>>> t = Tolerances()
>>> t.equilibrium_moment_rel
1e-09
>>> Tolerances.from_mapping({"datum": {"zero_stress_rel": 1e-8}}
...                         ).datum_zero_stress_rel
1e-08

Notes
-----
The resolved set is meant to be recorded in any machine-readable dump
alongside ``zone_rheology`` and the declared rules: a run that does not
say at what tolerance it accepted is not reproducible.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Mapping, Optional

__all__ = ["Tolerances", "TOLERANCE_SCHEMA", "load_tolerance_file",
           "ORIGIN_DOCUMENT", "ORIGIN_DEFAULT", "origin_config_file"]


#: Provenance label for a tolerance written by the input document.
ORIGIN_DOCUMENT: str = "input document"

#: Provenance label for a tolerance nobody wrote.
ORIGIN_DEFAULT: str = "shipped default"


def origin_config_file(path) -> str:
    r"""
    Provenance label for a tolerance read from a configuration file.

    The path is part of the label, not an aside: two machines running
    the same model with two different ``gensec_tolerances.yaml`` files
    produce two different runs, and a dump that recorded only
    ``"configuration file"`` could not tell them apart.

    Parameters
    ----------
    path : str

    Returns
    -------
    str

    Examples
    --------
    >>> origin_config_file("/etc/gensec_tolerances.yaml")
    'configuration file: /etc/gensec_tolerances.yaml'
    """
    return f"configuration file: {path}"


#: The YAML shape, as ``{group: {key: default}}``.  This mapping is the
#: single source for the parser, the validator and the documentation:
#: a tolerance added here becomes settable, checked and documented at
#: once, and one added anywhere else is not settable at all.
TOLERANCE_SCHEMA: Dict[str, Dict[str, float]] = {
    "equilibrium": {
        # Residual of the solved plane against its demand, relative to
        # max(|N|, 1 N) and max(|Mx|, |My|, 1 N*mm).  1e-9 sits two to
        # three orders above float64 on the scales GenSec handles
        # (moments ~1e8 N*mm resolve to ~1e-17), so it is tight enough
        # to see a defect and loose enough not to chase the last bit.
        "force_rel": 1.0e-9,
        "moment_rel": 1.0e-9,
        "max_iter": 100,
    },
    "datum": {
        # "A zone reads zero stress at its own datum" and "a plane is
        # the plane that was drawn" are exact statements; they are
        # given the same 1e-9 because they are the same kind of claim.
        "zero_stress_rel": 1.0e-9,
        "plane_rel": 1.0e-9,
    },
    "prestress": {
        # sigma_p read back after grouting, against the jack.  Nothing
        # mechanical happens at grouting, so this residue is modelling
        # error and the tolerance is deliberately tight.
        "grout_readback_rel": 1.0e-9,
        # A cancellation must *negate the recorded emission*; its
        # correct value is zero, hence absolute, in MPa.
        "cancellation_abs": 1.0e-6,
    },
    "losses": {
        "creep_step_rel": 1.0e-8,
        # chi comes from a numerical integration on a finite grid; one
        # order looser than the algebraic tolerances, on purpose.
        "ageing_chi_rel": 1.0e-6,
    },
    "geometry": {
        "area_rel": 1.0e-12,
        # A centroid offset that must vanish: absolute, in mm.
        "centroid_abs": 1.0e-9,
    },
}


def _flat_defaults() -> Dict[str, float]:
    r"""
    ``{flat_name: default}`` from :data:`TOLERANCE_SCHEMA`.

    Returns
    -------
    dict
    """
    return {f"{group}_{key}": value
            for group, keys in TOLERANCE_SCHEMA.items()
            for key, value in keys.items()}


@dataclasses.dataclass(frozen=True)
class Tolerances:
    r"""
    A resolved tolerance set.

    Field names are the YAML path flattened with an underscore:
    ``equilibrium.force_rel`` is :attr:`equilibrium_force_rel`.  The
    document keeps the grouping because it reads better; the object
    drops it because a consumer wants one name, not a path.

    Parameters
    ----------
    **fields
        Any key of :data:`TOLERANCE_SCHEMA`, flattened.  Every field
        has the shipped default.

    Raises
    ------
    ValueError
        If a tolerance is not a finite positive number.
    """

    equilibrium_force_rel: float = 1.0e-9
    equilibrium_moment_rel: float = 1.0e-9
    equilibrium_max_iter: int = 100
    datum_zero_stress_rel: float = 1.0e-9
    datum_plane_rel: float = 1.0e-9
    prestress_grout_readback_rel: float = 1.0e-9
    prestress_cancellation_abs: float = 1.0e-6
    losses_creep_step_rel: float = 1.0e-8
    losses_ageing_chi_rel: float = 1.0e-6
    geometry_area_rel: float = 1.0e-12
    geometry_centroid_abs: float = 1.0e-9

    #: Where each tolerance came from, as a sorted tuple of
    #: ``(flat_name, origin)`` pairs.  A tuple and not a dict because
    #: the dataclass is frozen and a mapping would be mutable through
    #: the back door; :meth:`provenance_map` gives the readable form.
    #:
    #: Complete by construction: every field of the schema has an entry,
    #: including the ones nobody wrote.  Recording only the overridden
    #: ones would leave the reader to infer *"absent means default"* —
    #: a silent default wearing the clothes of an economy.
    #:
    #: **Outside equality** (``compare=False``), deliberately.  A
    #: tolerance of :math:`10^{-9}` that came from the shipped default
    #: and one an input document wrote explicitly are the *same
    #: tolerance*: the analysis they produce is numerically identical,
    #: to the last bit.  Provenance is the audit trail of how the number
    #: arrived, not part of what it is.  This is the opposite of a
    #: :class:`~gensec.materials.base.DeclaredRule`, whose source *is*
    #: content and does enter the identity token — because there the
    #: engineer's choice of value is the thing being recorded, and two
    #: analyses that differ in it really are two analyses.
    provenance: tuple = dataclasses.field(default=(), compare=False,
                                          repr=False)

    #: Short alias used at the grout site, where the long name would
    #: bury the call.  Kept as a property, not a second field, so the
    #: value has one home.
    @property
    def grout_readback_rel(self) -> float:
        r"""Alias of :attr:`prestress_grout_readback_rel`."""
        return self.prestress_grout_readback_rel

    def __post_init__(self):
        r"""Validate every field (always on, never under ``assert``)."""
        if not self.provenance:
            object.__setattr__(
                self, "provenance",
                tuple(sorted((name, ORIGIN_DEFAULT)
                             for name in _flat_defaults())))
        for field in dataclasses.fields(self):
            if field.name == "provenance":
                continue
            value = getattr(self, field.name)
            if field.name.endswith("max_iter"):
                if int(value) < 1:
                    raise ValueError(
                        f"tolerances: {field.name} must be >= 1, "
                        f"got {value!r}.")
                continue
            v = float(value)
            if not (v > 0.0) or v != v or v in (float("inf"),):
                raise ValueError(
                    f"tolerances: {field.name} must be a finite "
                    f"positive number, got {value!r}.  A tolerance of "
                    f"zero is not a tolerance; a negative one is a "
                    f"typo that would accept everything.")

    @classmethod
    def from_mapping(cls, block: Optional[Mapping[str, Any]],
                     base: Optional["Tolerances"] = None,
                     origin: str = ORIGIN_DOCUMENT) -> "Tolerances":
        r"""
        Build from a nested ``tolerances:`` block.

        Parameters
        ----------
        block : dict or None
            ``{group: {key: value}}``.  ``None`` returns *base*
            unchanged (or the shipped defaults).
        base : Tolerances or None, optional
            The set this block overrides -- the configuration file when
            parsing an input document.
        origin : str, optional
            Provenance label recorded for the keys *block* sets.  Keys
            it does not set keep *base*'s provenance, so a two-step
            resolution (file, then document) leaves each tolerance
            labelled with whichever of the two actually wrote it.

        Returns
        -------
        Tolerances

        Raises
        ------
        ValueError
            On an unknown group or key.  An override that is silently
            ignored is a silent error: the whole point of writing it
            was to change something.

        Examples
        --------
        >>> t = Tolerances.from_mapping({"geometry": {"area_rel": 1e-10}})
        >>> t.provenance_map()["geometry_area_rel"]
        'input document'
        >>> t.provenance_map()["datum_plane_rel"]
        'shipped default'
        """
        current = dataclasses.asdict(base or cls())
        prov = dict(current.pop("provenance", ()) or ())
        if block is None:
            return cls(**current,
                       provenance=tuple(sorted(prov.items())))
        if not isinstance(block, Mapping):
            raise ValueError(
                f"tolerances: expected a mapping of groups, got "
                f"{type(block).__name__}.")
        for group, keys in block.items():
            if group not in TOLERANCE_SCHEMA:
                raise ValueError(
                    f"tolerances: unknown group '{group}'.  Valid: "
                    f"{sorted(TOLERANCE_SCHEMA)}.")
            if not isinstance(keys, Mapping):
                raise ValueError(
                    f"tolerances['{group}']: expected a mapping, got "
                    f"{type(keys).__name__}.")
            for key, value in keys.items():
                if key not in TOLERANCE_SCHEMA[group]:
                    raise ValueError(
                        f"tolerances['{group}']: unknown key '{key}'.  "
                        f"Valid: {sorted(TOLERANCE_SCHEMA[group])}.")
                current[f"{group}_{key}"] = value
                prov[f"{group}_{key}"] = str(origin)
        return cls(**current, provenance=tuple(sorted(prov.items())))

    def provenance_map(self) -> Dict[str, str]:
        r"""
        Provenance as ``{flat_name: origin}``.

        Returns
        -------
        dict

        Examples
        --------
        >>> Tolerances().provenance_map()["equilibrium_force_rel"]
        'shipped default'
        """
        return dict(self.provenance)

    def provenance_document(self) -> Dict[str, str]:
        r"""
        Provenance keyed the way the YAML block is written.

        The dotted path (``equilibrium.force_rel``) rather than the
        flattened field name, because the reader of a dump is holding
        the input document, not the dataclass.

        Returns
        -------
        dict
            ``{"group.key": origin}`` for every tolerance in the
            schema.

        Examples
        --------
        >>> Tolerances().provenance_document()["datum.plane_rel"]
        'shipped default'
        """
        flat = self.provenance_map()
        return {f"{group}.{key}": flat.get(f"{group}_{key}",
                                           ORIGIN_DEFAULT)
                for group, keys in TOLERANCE_SCHEMA.items()
                for key in keys}

    def as_document(self) -> Dict[str, Dict[str, float]]:
        r"""
        The set back in nested form, for a dump or a report.

        Returns
        -------
        dict
        """
        return {group: {key: getattr(self, f"{group}_{key}")
                        for key in keys}
                for group, keys in TOLERANCE_SCHEMA.items()}


def load_tolerance_file(path: Optional[str] = None) -> Tolerances:
    r"""
    Read a ``gensec_tolerances.yaml`` configuration file.

    The fallback house for tolerances that no input document can name:
    those used by the validators, the fingerprint tools and the
    ``run_*`` drivers, which have no model document of their own.  The
    grammar is identical to the ``tolerances:`` block of an input
    document, so there is one schema and not two.

    Parameters
    ----------
    path : str or None, optional
        File to read.  ``None`` looks at the ``GENSEC_TOLERANCES``
        environment variable, then ``gensec_tolerances.yaml`` in the
        working directory.  A missing file is not an error -- the
        shipped defaults are the answer.

    Returns
    -------
    Tolerances
    """
    candidate = path or os.environ.get("GENSEC_TOLERANCES")
    if candidate is None:
        candidate = "gensec_tolerances.yaml"
        if not os.path.exists(candidate):
            return Tolerances()
    if not os.path.exists(candidate):
        raise ValueError(
            f"tolerances: file '{candidate}' does not exist.  A "
            f"configuration that was asked for and not found is not a "
            f"default, it is a missing file.")
    import yaml
    with open(candidate, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    return Tolerances.from_mapping(doc.get("tolerances", doc),
                                   origin=origin_config_file(candidate))
