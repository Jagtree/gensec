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
Fiber integrator and equilibrium solver — biaxial bending.

Supports both single-material and multi-material sections. When
the section has ``mat_indices`` (from :class:`GenericSection` with
``bulk_materials`` zones), the integrator groups fibers by material
to call each constitutive law on its own subset. For single-material
sections, the fast vectorized path is used unchanged.

The strain plane has 3 parameters
:math:`(\varepsilon_0, \chi_x, \chi_y)` and the internal forces
are :math:`(N, M_x, M_y)`.

Uniaxial bending is the special case :math:`\chi_y = 0`.

Performance features
--------------------
- **Batch integration** (:meth:`integrate_batch`): evaluate many
  strain configurations in a single vectorized call, eliminating
  Python-loop overhead in capacity-surface generators.
- **Analytical tangent stiffness** (:meth:`integrate_with_tangent`):
  compute internal forces *and* the 3×3 tangent matrix in one pass,
  halving the cost of each Newton-Raphson iteration compared to a
  finite-difference Jacobian.
"""

import numpy as np


class AxialStateNotAttainable(ValueError):
    r"""
    A pure-axial demand lies outside the section's zero-moment reach.

    Raised when ``solve_equilibrium`` is asked for a state with both
    target moments negligible, the search fails, **and** a measurement
    on the section itself shows that no strain plane satisfies the
    request: the requested axial force is larger, in magnitude, than
    anything the section can carry with :math:`|M_x| \le M_{tol}`.

    Why the distinction is worth an exception
    -----------------------------------------
    Before this class existed the caller got ``converged = False`` and
    nothing else, and the two situations behind it were
    indistinguishable:

    * the state exists and the search did not find it — a **defect**;
    * the state does not exist — a **fact about the section**.

    The second is not a failure of GenSec and cannot be repaired by a
    better search.  Reporting it as a non-convergence is what item
    N-17 records, and the measurement that closed N-17 showed it was
    the second case: on a 1200 x 200 strand-only slab with one
    700 mm^2 strand at ``eps_pe = 0.0060`` and a lever arm of -60 mm,
    the largest axial force reachable with :math:`|M_x| \le 10^3`
    N.mm is 317 016 N, while the interrogated target was
    1 056 673 N — the request misses the attainable set by 70 %, and
    the minimum :math:`|M_x|` anywhere on the locus
    :math:`N = N_{target}` is :math:`6.340041 \times 10^7` N.mm,
    :math:`6.34 \times 10^4` times the tolerance.

    The mechanism is elementary once stated.  An eccentric tendon at
    lever arm :math:`e` carries tension :math:`T`; concrete carries no
    tension, so at high axial load the section is inert everywhere
    except the strand and

    .. math::

        M_x = T\,e ,

    which is zero only if :math:`T` is.  Bringing :math:`M_x` back to
    zero needs a compression resultant :math:`C < 0` with

    .. math::

        |C| \, |l_C| \;=\; |T| \, |e| ,

    and every newton of that compression is subtracted from the axial
    force, so

    .. math::

        N \;=\; T + C \;\le\; T \left( 1 - \frac{|e|}{|l_C|_{max}}
        \right),

    an upper bound strictly below :math:`T_{max}` whenever the tendon
    is eccentric.  The number in ``attainable`` is not this bound: it
    is the measured maximum over the :math:`M_x = 0` locus, which is
    tighter.

    Taxonomy
    --------
    A :class:`ValueError`, deliberately, and **not** a
    :class:`NotImplementedError`.  It is the opposite case from
    :class:`~gensec.solver.posttension.LoadBetweenJacksNotModelled`:
    there GenSec declines a well-posed question it has no authority to
    answer, here GenSec *answers* — negatively, with a certificate.
    The request is a property of the demand, so a caller that catches
    ``ValueError`` to mean "this demand cannot be satisfied by this
    section" should catch this.

    Attributes
    ----------
    N_target : float
        The requested axial force [N].
    attainable : tuple of float
        ``(N_min, N_max)``: the smallest and largest axial force the
        section carries with :math:`|M_x| \le` ``M_tol`` [N].
    state : tuple of float
        ``(eps0, chi_x)`` of the extreme state the bound was measured
        at — the nearest thing to the request that does exist.
    M_tol : float
        Moment tolerance the reach was measured under [N.mm].
    resolution : tuple of int
        ``(n_chi, n_eps)`` of the scan that produced the bound, so the
        certificate carries the coarseness of its own measurement.

    See Also
    --------
    FiberSolver.axial_reach_at_zero_moment : the measurement.
    FiberSolver.solve_equilibrium : the site that raises.
    """

    def __init__(self, N_target, attainable, state, M_tol, resolution):
        self.N_target = float(N_target)
        self.attainable = (float(attainable[0]), float(attainable[1]))
        self.state = (float(state[0]), float(state[1]))
        self.M_tol = float(M_tol)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        near = (self.attainable[1] if self.N_target > 0
                else self.attainable[0])
        miss = abs(self.N_target - near)
        pct = 100.0 * miss / max(abs(self.N_target), 1.0)
        super().__init__(
            f"no strain plane of this section carries N = "
            f"{self.N_target:.1f} N with |Mx| <= {self.M_tol:g} N.mm.  "
            f"The reach at zero moment was measured on the section "
            f"itself and is [{self.attainable[0]:.1f}, "
            f"{self.attainable[1]:.1f}] N; the request misses it by "
            f"{miss:.1f} N ({pct:.2f} % of the target).  The nearest "
            f"state that does exist is eps0 = {self.state[0]:.6g}, "
            f"chi_x = {self.state[1]:.6g}.  This is not a search "
            f"failure: the state does not exist, and no tolerance or "
            f"initial guess will produce it.  Scan resolution: "
            f"{self.resolution[0]} curvatures x {self.resolution[1]} "
            f"strains.")


def _yield_strain_of(material):
    r"""
    Yield strain of a steel-like material, or ``None``.

    Parameters
    ----------
    material : Material

    Returns
    -------
    float or None
        An explicit ``eps_yd`` where the material declares one, else
        ``f_yd / E_s`` where both are declared, else ``None``.  Never
        a substituted value: a capacity factor written against a yield
        strain is meaningless evaluated against an invented one.
    """
    for name in ("eps_yd", "eps_y"):
        value = getattr(material, name, None)
        if value is not None and float(value) > 0.0:
            return float(value)
    modulus = getattr(material, "Es", None) or getattr(material, "Ep", None)
    strength = (getattr(material, "fyd", None)
                or getattr(material, "f_p01d", None))
    if modulus and strength and float(modulus) > 0.0:
        return float(strength) / float(modulus)
    return None


class FiberSolver:
    r"""
    Material-agnostic fiber solver for biaxial bending + axial force.

    Strain plane:

    .. math::

        \varepsilon(x, y) = \varepsilon_0
            + \chi_x \, (y - y_{\text{ref}})
            + \chi_y \, (x - x_{\text{ref}})

    Internal forces:

    .. math::

        N   &= \sum_i \sigma_i \, A_i \\
        M_x &= \sum_i \sigma_i \, A_i \, (y_i - y_{\text{ref}}) \\
        M_y &= \sum_i \sigma_i \, A_i \, (x_i - x_{\text{ref}})

    Convention:

    - :math:`M_x > 0` compresses the bottom edge (:math:`y = 0`),
      tensions the top edge. Right-hand rule: thumb along +x,
      fingers curl from +y toward observer.
    - :math:`M_y > 0` compresses the right edge (:math:`x = x_{\max}`),
      tensions the left edge. Right-hand rule: thumb along +y,
      fingers curl from +x toward observer.

    Parameters
    ----------
    section : GenericSection or RectSection
        Any section exposing ``x_fibers``, ``y_fibers``, ``A_fibers``,
        ``x_rebars``, ``y_rebars``, ``A_rebars``, ``embedded_rebars``,
        ``bulk_material``, ``rebars``, ``x_centroid``, ``y_centroid``.
    x_ref : float or None, optional
        Reference x-coordinate [mm]. Default: x-centroid.
    y_ref : float or None, optional
        Reference y-coordinate [mm]. Default: y-centroid.
    """

    def __init__(self, section, x_ref=None, y_ref=None):
        self.sec = section
        self.x_ref = x_ref if x_ref is not None else section.x_centroid
        self.y_ref = y_ref if y_ref is not None else section.y_centroid
        self._rebar_groups = self._build_rebar_groups()
        #: Zero-moment reach measurements, keyed by the resolution
        #: and tolerance they were taken at (N-17).
        self._axial_reach_cache = {}

        # Pre-compute lever arms (constant for a given section)
        self._ly_bulk = self.sec.y_fibers - self.y_ref
        self._lx_bulk = self.sec.x_fibers - self.x_ref
        self._ly_rebar = self.sec.y_rebars - self.y_ref
        self._lx_rebar = self.sec.x_rebars - self.x_ref

        # Tendon lever arms and locked-in initial strains.  Sections
        # without tendons (legacy or non-prestressed) expose empty
        # arrays from ``_setup_tendons``; ``getattr`` keeps the solver
        # tolerant of sections built before the prestress phase.
        self._y_tendon = getattr(self.sec, 'y_tendons',
                                 np.empty(0, dtype=float))
        self._x_tendon = getattr(self.sec, 'x_tendons',
                                 np.empty(0, dtype=float))
        self._ly_tendon = self._y_tendon - self.y_ref
        self._lx_tendon = self._x_tendon - self.x_ref
        self._eps_init_tendon = getattr(
            self.sec, 'eps_init_tendons', np.empty(0, dtype=float))
        self._has_tendons = len(self._y_tendon) > 0
        self._tendon_groups = (self._build_tendon_groups()
                               if self._has_tendons else [])

        # Build bulk material groups for multi-material support
        self._bulk_groups = self._build_bulk_groups()
        self._is_multi_material = len(self._bulk_groups) > 1

        # ---- Bulk imposed-strain offset (coazione) ----
        # Uniform locked-in bulk strain (shrinkage / differential
        # thermal), read through getattr so sections built before the
        # prestress phase are bit-identical (offset 0.0 -> argument
        # unchanged everywhere).  The bulk law is evaluated at
        # (eps + offset), exactly as a bonded tendon's law is
        # evaluated at (eps_section + eps_pe).
        self._bulk_eps_init = float(
            getattr(self.sec, 'bulk_eps_init', 0.0))
        # Per-fiber / per-element imposed-strain offset fields
        # (Phase 8): the uniform scalar above plus the per-zone
        # locked-in datum planes carried by the view
        # (``bulk_planes_active``).  ``None`` on the fast path — the
        # consumption sites then add the plain scalar, keeping the
        # legacy numeric stream unchanged (never ``eb + zeros(n)``).
        self._build_offset_fields()

    def _build_rebar_groups(self):
        r"""
        Group rebar layers by the pair
        ``(rebar material, bulk-zone material)``.

        For multi-material sections (a confined core inside an
        unconfined cover, for example), an embedded rebar must
        subtract the bulk stress evaluated with the constitutive
        law of the **zone the rebar physically occupies**, not the
        primary ``bulk_material``.  Splitting the rebar groups by
        this pair lets the integrator vectorise the lookup with no
        per-rebar branching at evaluation time.

        For single-material sections (no zones, or every rebar in
        zone ``0``) this collapses to the legacy "group by rebar
        material" behaviour, with the primary ``bulk_material``
        attached to every group.

        Returns
        -------
        list of tuple
            Each tuple is
            ``(rebar_material, bulk_zone_material,
            ndarray_of_rebar_indices)``.
        """
        sec = self.sec
        # Index-aligned list of candidate bulk-zone materials.
        all_bulk_mats = (sec.get_all_bulk_materials()
                         if hasattr(sec, 'get_all_bulk_materials')
                         else [sec.bulk_material])
        # Per-rebar zone index.  Sections without
        # ``mat_indices_rebar`` (legacy ``RectSection``) place every
        # rebar in zone 0.
        mat_idx_r = getattr(sec, 'mat_indices_rebar', None)
        if mat_idx_r is None:
            mat_idx_r = np.zeros(len(sec.rebars), dtype=int)

        groups = {}
        for i, r in enumerate(sec.rebars):
            key = (id(r.material), int(mat_idx_r[i]))
            if key not in groups:
                bulk_mat = all_bulk_mats[mat_idx_r[i]]
                groups[key] = (r.material, bulk_mat, [])
            groups[key][2].append(i)
        return [(rm, bm, np.array(ix))
                for rm, bm, ix in groups.values()]

    def _build_tendon_groups(self):
        r"""
        Group tendons by the pair
        ``(tendon material, bulk-zone material)``.

        Identical zoning logic to :meth:`_build_rebar_groups`: a
        bonded tendon inside a confined core must subtract the bulk
        stress evaluated with the constitutive law of the zone it
        physically occupies.  Splitting by ``(tendon mat, zone mat)``
        vectorises the lookup with no per-tendon branching.

        Unlike rebars, the tendon's own law is evaluated at the
        **offset** total strain (section strain + locked-in initial
        strain), while the displaced-bulk subtraction uses the
        **section** strain — the two arguments differ.  The grouping
        carries the per-tendon initial-strain offset alongside the
        indices so the integrator can apply it group-wise.

        Returns
        -------
        list of tuple
            Each tuple is
            ``(tendon_material, bulk_zone_material,
            ndarray_of_tendon_indices, ndarray_of_eps_init)``.
        """
        sec = self.sec
        all_bulk_mats = (sec.get_all_bulk_materials()
                         if hasattr(sec, 'get_all_bulk_materials')
                         else [sec.bulk_material])
        mat_idx_t = getattr(sec, 'mat_indices_tendon', None)
        if mat_idx_t is None:
            mat_idx_t = np.zeros(len(sec.tendons), dtype=int)

        groups = {}
        for i, t in enumerate(sec.tendons):
            key = (id(t.material), int(mat_idx_t[i]))
            if key not in groups:
                bulk_mat = all_bulk_mats[mat_idx_t[i]]
                groups[key] = (t.material, bulk_mat, [])
            groups[key][2].append(i)

        out = []
        for rm, bm, ix in groups.values():
            ix_arr = np.array(ix)
            out.append((rm, bm, ix_arr,
                        self._eps_init_tendon[ix_arr]))
        return out

    def _build_bulk_groups(self):
        r"""
        Group bulk fibers by material index.

        For single-material sections (no ``mat_indices`` attribute or
        all indices == 0), returns a single group covering all fibers.

        Returns
        -------
        list of tuple
            Each tuple is ``(Material, ndarray_of_fiber_indices)``.
        """
        sec = self.sec

        # Check if section has multi-material zones
        mat_indices = getattr(sec, 'mat_indices', None)
        if mat_indices is None or np.all(mat_indices == 0):
            # Single material: one group covering all fibers
            idx = np.arange(sec.n_fibers)
            return [(sec.bulk_material, idx)]

        # Multi-material: group by index
        all_mats = sec.get_all_bulk_materials()
        groups = []
        for mi in np.unique(mat_indices):
            idx = np.where(mat_indices == mi)[0]
            groups.append((all_mats[mi], idx))
        return groups

    def _build_offset_fields(self):
        r"""
        Precompute the imposed-strain offset fields (Phase 8).

        The bulk constitutive law of fiber *i* (zone
        :math:`z(i) = \texttt{mat\_indices}[i]`) is evaluated at the
        offset argument
        :math:`\varepsilon_{\mathrm{sec},i} +
        \varepsilon^{\mathrm{off}}_i` with

        .. math::

           \varepsilon^{\mathrm{off}}_i \;=\; \varepsilon_{b,0}
           \; + \; \varepsilon_{0,z(i)}
           \; + \; \chi_{x,z(i)}\,(y_i - y_{\mathrm{ref}})
           \; - \; \chi_{y,z(i)}\,(x_i - x_{\mathrm{ref}})

        — the exact sign convention of :meth:`strain_field` — where
        :math:`\varepsilon_{b,0}` is the legacy uniform scalar and
        :math:`(\varepsilon_{0,z}, \chi_{x,z}, \chi_{y,z})` the
        per-zone locked-in plane from the view attribute
        ``bulk_planes_active`` (attached by
        :func:`~gensec.solver.section_state.materialize_view`).
        Companion fields are evaluated at the rebar and tendon
        coordinates for the displaced-bulk subtraction: the displaced
        concrete of zone *z* carries zone *z*'s plane at the element
        point, with the zone given by the **geometric** containment
        (``mat_indices_rebar`` / ``mat_indices_tendon`` — a staging
        ``parent`` override never touches the physics).

        **Fast path**: with no planes (or all-zero planes) every field
        is ``None`` and the consumption sites add the plain scalar
        ``_bulk_eps_init`` — byte-identity with the pre-Phase-8
        pipeline by construction (the NumPy instruction stream is
        unchanged), not by IEEE luck.

        Raises
        ------
        ValueError
            ``bulk_planes_active`` present with a shape inconsistent
            with the section's zone indices.
        """
        planes = getattr(self.sec, 'bulk_planes_active', None)
        if planes is None or not np.any(
                np.asarray(planes, dtype=float)):
            self._bulk_eps_field = None
            self._rebar_eps_field = None
            self._tendon_eps_field = None
            return
        planes = np.asarray(planes, dtype=float)
        mi = getattr(self.sec, 'mat_indices', None)
        if mi is None:
            mi = np.zeros(int(self.sec.n_fibers), dtype=int)
        mi = np.asarray(mi, dtype=int)
        mir = getattr(self.sec, 'mat_indices_rebar', None)
        if mir is None:
            mir = np.zeros(self._ly_rebar.size, dtype=int)
        mir = np.asarray(mir, dtype=int)
        mit = getattr(self.sec, 'mat_indices_tendon', None)
        if mit is None:
            mit = np.zeros(self._ly_tendon.size, dtype=int)
        mit = np.asarray(mit, dtype=int)
        zmax = max([int(mi.max(initial=0)),
                    int(mir.max(initial=0)),
                    int(mit.max(initial=0))])
        if planes.ndim != 2 or planes.shape[1] != 3 \
                or planes.shape[0] <= zmax:
            raise ValueError(
                f"bulk_planes_active has shape {planes.shape}; "
                f"expected (n_zones >= {zmax + 1}, 3) for this "
                f"section's zone indices."
            )
        eb0 = self._bulk_eps_init
        self._bulk_eps_field = (eb0 + planes[mi, 0]
                                + planes[mi, 1] * self._ly_bulk
                                - planes[mi, 2] * self._lx_bulk)
        self._rebar_eps_field = (eb0 + planes[mir, 0]
                                 + planes[mir, 1] * self._ly_rebar
                                 - planes[mir, 2] * self._lx_rebar)
        self._tendon_eps_field = (eb0 + planes[mit, 0]
                                  + planes[mit, 1] * self._ly_tendon
                                  - planes[mit, 2] * self._lx_tendon)

    def _bulk_offset(self, idx=None):
        r"""
        Imposed-strain offset at bulk fibers.

        Parameters
        ----------
        idx : numpy.ndarray or None, optional
            Fiber subset; ``None`` for all fibers.

        Returns
        -------
        float or numpy.ndarray
            The scalar ``_bulk_eps_init`` on the fast path (fields
            ``None``), else the per-fiber field (sliced).  Either
            broadcasts transparently against 1-D and batch strain
            arrays at every consumption site.
        """
        if self._bulk_eps_field is None:
            return self._bulk_eps_init
        return (self._bulk_eps_field if idx is None
                else self._bulk_eps_field[idx])

    def bulk_strain_offset(self, idx=None):
        r"""
        The imposed-strain offset every bulk constitutive law is
        evaluated at, on top of the section strain.

        .. math::

            \varepsilon^{\mathrm{mech}}_i = \varepsilon_{\mathrm{sec},i}
                + \varepsilon_{b,0}
                + \varepsilon_{0,z(i)}
                + \chi_{x,z(i)}\,\ell_{y,i}
                - \chi_{y,z(i)}\,\ell_{x,i}

        Public because the *criteria* need it, not only the integration:
        cracking, crushing and the ultimate limit are properties of the
        material, and a material reads the mechanical strain.  A scan
        that compares :math:`\varepsilon_{\mathrm{sec}}` against a
        material threshold asks the wrong fibre a right question — the
        defect **F-M3**, measured in ``handoff/14_1`` §5 and repaired in
        Phase 15 by routing every such site through this accessor
        instead of letting each one re-derive the field.

        The rebar and tendon companions are deliberately *not*
        published: a rebar's own law reads the section strain alone (the
        concrete's imposed field is the concrete's), and a tendon's
        reads the section strain plus its own prestrain.  Their offsets
        exist only for the displaced-bulk subtraction, which is an
        integration concern and has no criterion attached.

        Parameters
        ----------
        idx : numpy.ndarray or None, optional
            Fiber subset; ``None`` for all bulk fibers.

        Returns
        -------
        float or numpy.ndarray
            A scalar on the fast path (no planes attached), else the
            per-fiber field.  Either broadcasts transparently against a
            1-D strain array and against an ``(n_configs, n_fibers)``
            batch.

        Examples
        --------
        >>> eb, er = solver.strain_field(eps0, chi_x, chi_y)
        ... # doctest: +SKIP
        >>> eb_mech = eb + solver.bulk_strain_offset()
        ... # doctest: +SKIP
        """
        return self._bulk_offset(idx)

    def _rebar_offset(self, idx):
        r"""
        Displaced-bulk offset at rebar locations (fiber-subset *idx*).

        Returns
        -------
        float or numpy.ndarray
        """
        if self._rebar_eps_field is None:
            return self._bulk_eps_init
        return self._rebar_eps_field[idx]

    def element_limit_groups(self):
        r"""
        Every element group with the strain limits of **its own**
        material and the argument that material's law is evaluated at.

        The one home of the fact *"this element's strain, against this
        element's limit"* (N-2, Phase 16).  Before it existed, the
        curvature scans pooled the bulk and rebar strains into one array
        and compared its minimum against the **concrete** crushing limit
        and its maximum against the pool's largest limit, which on a
        reinforced section is the **steel's** -- so a concrete fibre was
        tested against a limit no concrete has, and the tendons, which
        carry :math:`\varepsilon_{pe}` on top of the section strain and
        are therefore the elements closest to their own limit at every
        curvature, were not tested at all.

        Returns
        -------
        list of tuple
            ``(kind, idx, eps_min, eps_max)`` where *kind* is one of
            ``'bulk'``, ``'rebar'``, ``'tendon'``, *idx* indexes that
            class's strain array, and the two limits are the group
            material's own.

        Notes
        -----
        The **argument** each class is evaluated at is the caller's
        responsibility and is stated here so the two cannot drift:
        bulk at :math:`\varepsilon_{sec} + ` :meth:`bulk_strain_offset`,
        rebar at :math:`\varepsilon_{sec}` bare, tendon at
        :math:`\varepsilon_{sec} + \varepsilon_{init}`.  This is the
        same triple :meth:`strains_within_limits` uses; that method
        answers "is this plane admissible", this one exposes the
        per-group data a scan needs to answer "at which step did it stop
        being".

        Examples
        --------
        >>> groups = solver.element_limit_groups()      # doctest: +SKIP
        >>> {kind for kind, _i, _a, _b in groups}       # doctest: +SKIP
        {'bulk', 'rebar', 'tendon'}
        """
        out = []
        for mat, idx in self._bulk_groups:
            out.append(("bulk", idx, float(mat.eps_min),
                        float(mat.eps_max)))
        for mat, _bulk_mat, idx in self._rebar_groups:
            out.append(("rebar", idx, float(mat.eps_min),
                        float(mat.eps_max)))
        if self._has_tendons:
            for mat, _bulk_mat, idx, _eps_init in self._tendon_groups:
                out.append(("tendon", idx, float(mat.eps_min),
                            float(mat.eps_max)))
        return out

    def _tendon_offset(self, idx):
        r"""
        Displaced-bulk offset at tendon locations (subset *idx*).

        Returns
        -------
        float or numpy.ndarray
        """
        if self._tendon_eps_field is None:
            return self._bulk_eps_init
        return self._tendon_eps_field[idx]

    # ==================================================================
    #  Single-configuration integration
    # ==================================================================

    def strain_field(self, eps0, chi_x, chi_y=0.0):
        r"""
        Compute strains at all fibers.

        .. math::

            \varepsilon_i = \varepsilon_0
                + \chi_x (y_i - y_{\text{ref}})
                - \chi_y (x_i - x_{\text{ref}})

        The minus sign on :math:`\chi_y` ensures that positive
        :math:`M_y` compresses the right edge (:math:`x = x_{\max}`),
        consistent with the right-hand rule around the y-axis.

        Parameters
        ----------
        eps0 : float
            Strain at the reference point.
        chi_x : float
            Curvature about the x-axis [1/mm].
        chi_y : float, optional
            Curvature about the y-axis [1/mm]. Default 0.

        Returns
        -------
        eps_bulk : numpy.ndarray
        eps_rebars : numpy.ndarray
        """
        eb = eps0 + chi_x * self._ly_bulk - chi_y * self._lx_bulk
        er = eps0 + chi_x * self._ly_rebar - chi_y * self._lx_rebar
        return eb, er

    # ==================================================================
    #  Tendon contribution (prestress) — decoupled-argument core
    # ==================================================================

    def _tendon_section_strain(self, eps0, chi_x, chi_y=0.0):
        r"""
        Section strain at each tendon location (no prestrain offset).

        .. math::

            \varepsilon_{\text{sec},i} = \varepsilon_0
                + \chi_x (y_i - y_{\text{ref}})
                - \chi_y (x_i - x_{\text{ref}})

        This is the strain the **displaced bulk** sees.  The tendon's
        own law is evaluated at this value plus the locked-in initial
        strain; see :meth:`_tendon_forces`.

        Parameters
        ----------
        eps0, chi_x : float
        chi_y : float, optional

        Returns
        -------
        numpy.ndarray
            Section strain at tendons, shape ``(n_tendons,)``.
        """
        return (eps0 + chi_x * self._ly_tendon
                - chi_y * self._lx_tendon)

    def _tendon_forces(self, et_sec, want_tangent=False):
        r"""
        Accumulate tendon force/moment contributions (and tangent).

        Implements the hard correctness invariant for bonded
        prestress: the tendon evaluates its own constitutive law at
        the **offset total strain**

        .. math::

            \varepsilon_{\text{tot},i} =
                \varepsilon_{\text{sec},i} + \varepsilon_{\text{init},i}

        while the **displaced bulk** is evaluated at the section
        strain :math:`\varepsilon_{\text{sec},i}` alone.  The net
        embedded-tendon stress is therefore evaluated at two different
        arguments:

        .. math::

            \sigma^{\text{net}}_i =
                \sigma_p(\varepsilon_{\text{sec},i}
                         + \varepsilon_{\text{init},i})
                - \sigma_{\text{bulk}}(\varepsilon_{\text{sec},i})

        Because :math:`\varepsilon_{\text{init},i}` is a constant in
        this phase, the chain rule passes it through unchanged, so the
        net tangent modulus is

        .. math::

            E^{\text{net}}_{t,i} =
                E^{p}_t(\varepsilon_{\text{sec},i}
                        + \varepsilon_{\text{init},i})
                - E^{\text{bulk}}_t(\varepsilon_{\text{sec},i})

        evaluated at the same two arguments; the
        :math:`\boldsymbol{\varphi}\boldsymbol{\varphi}^T` outer
        product structure of the stiffness is unaffected.

        Parameters
        ----------
        et_sec : numpy.ndarray
            Section strain at tendons, shape ``(n_tendons,)`` (1-D) or
            ``(n_configs, n_tendons)`` (batch).
        want_tangent : bool, optional
            If ``True``, also accumulate the 3×3 tangent-stiffness
            contribution.  Only valid for the 1-D case.  Default
            ``False``.

        Returns
        -------
        N, Mx, My : float or numpy.ndarray
            Tendon force/moment contributions.  Scalars for the 1-D
            case, arrays of shape ``(n_configs,)`` for the batch case.
        K : numpy.ndarray, optional
            3×3 tangent-stiffness contribution.  Returned only when
            ``want_tangent`` is ``True``.

        Notes
        -----
        ``embedded`` is honoured exactly as for rebars: a
        non-embedded tendon does not displace bulk, so the bulk
        subtraction is skipped for those entries.
        """
        batch = et_sec.ndim == 2
        embedded = getattr(self.sec, 'embedded_tendons',
                           np.ones(et_sec.shape[-1], dtype=bool))

        if batch:
            N = np.zeros(et_sec.shape[0])
            Mx = np.zeros(et_sec.shape[0])
            My = np.zeros(et_sec.shape[0])
        else:
            N = Mx = My = 0.0
            if want_tangent:
                K = np.zeros((3, 3))

        for mat, bulk_mat, idx, eps_init in self._tendon_groups:
            a = self.sec.A_tendons[idx]
            ly_t = self._ly_tendon[idx]
            lx_t = self._lx_tendon[idx]
            emb = embedded[idx]

            if batch:
                e_sec_g = et_sec[:, idx]
                e_tot_g = e_sec_g + eps_init[None, :]
                s_tendon = mat.stress_array(e_tot_g)
                s_bulk = bulk_mat.stress_array(
                    e_sec_g + self._tendon_offset(idx))
                s_net = s_tendon.copy()
                s_net[:, emb] -= s_bulk[:, emb]
                fa = s_net * a[None, :]
                N += fa.sum(axis=1)
                Mx += (fa * ly_t[None, :]).sum(axis=1)
                My -= (fa * lx_t[None, :]).sum(axis=1)
                continue

            e_sec_g = et_sec[idx]
            e_tot_g = e_sec_g + eps_init
            s_tendon = mat.stress_array(e_tot_g)
            s_bulk = bulk_mat.stress_array(
                e_sec_g + self._tendon_offset(idx))
            s_net = s_tendon.copy()
            s_net[emb] -= s_bulk[emb]
            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * ly_t))
            My -= float(np.sum(fa * lx_t))

            if want_tangent:
                Et_tendon = mat.tangent_array(e_tot_g)
                Et_bulk = bulk_mat.tangent_array(
                    e_sec_g + self._tendon_offset(idx))
                Et_net = Et_tendon.copy()
                Et_net[emb] -= Et_bulk[emb]
                EtA = Et_net * a
                K[0, 0] += float(np.sum(EtA))
                K[0, 1] += float(np.sum(EtA * ly_t))
                K[0, 2] -= float(np.sum(EtA * lx_t))
                K[1, 1] += float(np.sum(EtA * ly_t * ly_t))
                K[1, 2] -= float(np.sum(EtA * ly_t * lx_t))
                K[2, 2] += float(np.sum(EtA * lx_t * lx_t))

        if not batch and want_tangent:
            K[1, 0] = K[0, 1]
            K[2, 0] = K[0, 2]
            K[2, 1] = K[1, 2]
            return N, Mx, My, K
        return N, Mx, My

    def sectional_state(self, eps0, chi_x, chi_y=0.0,
                        action="flexure", confinement="none"):
        r"""
        The solved strain state, in the form a code can factor.

        Builds a
        :class:`~gensec.materials.codes.SectionalState` from a strain
        plane.  It exists because some codes carry their safety format
        on the **capacity** rather than on the materials, and such a
        factor is a function of the strain distribution the section
        reached --- it cannot be evaluated before the solve, and it
        cannot be folded into a material law.

        GenSec already knows everything such a rule needs; what was
        missing was a name for it.

        Parameters
        ----------
        eps0 : float
            Axial strain at the reference point.
        chi_x, chi_y : float
            Curvatures [1/mm].
        action : str
            Which resistance is being factored.  Carried through
            unread: one section can need several factors at once, so
            the action belongs to the question.
        confinement : str
            ``'tied'``, ``'spiral'`` or ``'none'``.

        Returns
        -------
        SectionalState

        Notes
        -----
        ``eps_t`` is the **net** tensile strain: for a tendon the
        locked-in ``eps_init`` is subtracted, because a strain-based
        factor measures how far the section is from a brittle failure
        and a prestrain says nothing about that.  Including it would
        put every prestressed section in the ductile branch and never
        trip the transition.

        ``eps_t`` is ``None`` when the section has no tension element
        in tension.  The code is then required to decline rather than
        pick a branch, which is why this returns ``None`` instead of
        ``0.0``: zero is a strain, absence is not.

        Examples
        --------
        >>> state = solver.sectional_state(1.0e-3, 5.0e-6)   # doctest: +SKIP
        >>> code.capacity_factor(state).value                # doctest: +SKIP
        """
        from ..materials.codes import SectionalState

        def _strain(ly):
            return eps0 + chi_x * np.asarray(ly, dtype=float)

        best_eps, best_yield = None, None

        n_reb = int(np.size(self._ly_rebar))
        if n_reb:
            eps_r = _strain(self._ly_rebar)
            k = int(np.argmax(eps_r))
            if eps_r[k] > 0.0:
                best_eps = float(eps_r[k])
                best_yield = self._yield_strain_of_rebar(k)

        n_ten = int(np.size(self._ly_tendon))
        if n_ten:
            # Net of the locked-in prestrain: see Notes.
            eps_p = _strain(self._ly_tendon)
            k = int(np.argmax(eps_p))
            if eps_p[k] > 0.0 and (best_eps is None or eps_p[k] > best_eps):
                best_eps = float(eps_p[k])
                best_yield = self._yield_strain_of_tendon(k)

        eps_c = None
        y_fib = getattr(self.sec, "y_fibers", None)
        if y_fib is not None and np.size(y_fib):
            eps_f = _strain(np.asarray(y_fib, dtype=float) - self.y_ref)
            low = float(np.min(eps_f))
            if low < 0.0:
                eps_c = low

        return SectionalState(eps_t=best_eps, eps_c=eps_c,
                              eps_yield=best_yield,
                              confinement=str(confinement),
                              action=str(action))

    def _yield_strain_of_rebar(self, index):
        r"""
        Yield strain of rebar *index*, or ``None``.

        Parameters
        ----------
        index : int

        Returns
        -------
        float or None
            ``None`` when the material declares neither an explicit
            yield strain nor the pair needed to derive one.  Declining
            is right: a rule written against ``eps_ty`` cannot be
            evaluated against a guess.
        """
        for mat, _b, idx, *_rest in self._rebar_groups:
            if index in np.atleast_1d(idx):
                return _yield_strain_of(mat)
        return None

    def _yield_strain_of_tendon(self, index):
        r"""
        Yield strain of tendon *index*, or ``None``.

        Parameters
        ----------
        index : int

        Returns
        -------
        float or None
        """
        for mat, _b, idx, *_rest in self._tendon_groups:
            if index in np.atleast_1d(idx):
                return _yield_strain_of(mat)
        return None

    def integrate(self, eps0, chi_x, chi_y=0.0):
        r"""
        Direct problem: strain parameters to internal forces.

        For embedded rebars, the bulk material contribution at the
        rebar location is subtracted to avoid double-counting:

        .. math::

            F_i = \bigl[\sigma_{\text{rebar},i}(\varepsilon_i)
                  - \sigma_{\text{bulk}}(\varepsilon_i)\bigr] \, A_{s,i}

        For multi-material sections, each fiber group is evaluated
        with its own constitutive law.

        Parameters
        ----------
        eps0 : float
        chi_x : float
            Curvature about x-axis [1/mm].
        chi_y : float, optional
            Curvature about y-axis [1/mm]. Default 0.

        Returns
        -------
        N : float
            Axial force [N]. Positive = tension.
        Mx : float
            Bending moment about x-axis [N*mm].
        My : float
            Bending moment about y-axis [N*mm].
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        # ---- Bulk contribution ----
        if not self._is_multi_material:
            # Fast path: single material on all fibers
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
        else:
            # Multi-material: evaluate each group at its own offset
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[idx] = mat.stress_array(
                    eb[idx] + self._bulk_offset(idx))

        fA = sb * self.sec.A_fibers
        N = float(np.sum(fA))
        Mx = float(np.sum(fA * self._ly_bulk))
        My = -float(np.sum(fA * self._lx_bulk))

        # ---- Rebar contribution ----
        # For each (rebar material, bulk-zone material) group, subtract
        # the bulk stress evaluated with the **zone's** constitutive
        # law from the rebar stress for embedded rebars, then accumulate
        # forces and moments.  Splitting by (rebar mat, zone mat) is
        # what makes this correct for multi-material sections.
        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]

            s_net = s_rebar.copy()
            s_net[emb] -= sb_at_rebars[emb]

            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * self._ly_rebar[idx]))
            My -= float(np.sum(fa * self._lx_rebar[idx]))

        # ---- Tendon contribution (prestress) ----
        if self._has_tendons:
            et = self._tendon_section_strain(eps0, chi_x, chi_y)
            Nt, Mxt, Myt = self._tendon_forces(et)
            N += Nt
            Mx += Mxt
            My += Myt

        return N, Mx, My

    # ==================================================================
    #  Analytical tangent stiffness
    # ==================================================================

    def integrate_with_tangent(self, eps0, chi_x, chi_y=0.0):
        r"""
        Internal forces **and** 3×3 tangent stiffness in one pass.

        The tangent stiffness matrix relates infinitesimal changes in
        the strain-plane parameters to changes in internal forces:

        .. math::

            \mathbf{K} =
            \frac{\partial (N,\,M_x,\,M_y)}
                 {\partial (\varepsilon_0,\,\chi_x,\,\chi_y)}
            = \sum_i E_{t,i} \, A_i \;
              \boldsymbol{\varphi}_i \, \boldsymbol{\varphi}_i^T

        where the shape-function vector for fiber *i* is

        .. math::

            \boldsymbol{\varphi}_i =
            \bigl[1,\; (y_i - y_{\text{ref}}),\;
                  -(x_i - x_{\text{ref}})\bigr]^T

        and :math:`E_{t,i} = d\sigma_i / d\varepsilon_i` is the
        tangent modulus at the current strain.

        Parameters
        ----------
        eps0 : float
        chi_x : float
        chi_y : float, optional

        Returns
        -------
        N, Mx, My : float
            Internal forces [N, N·mm].
        K : numpy.ndarray
            3×3 tangent stiffness matrix.
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        # ---- Bulk: stress and tangent ----
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
            Et_b = self.sec.bulk_material.tangent_array(
                eb + self._bulk_offset())
        else:
            sb = np.zeros_like(eb)
            Et_b = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                e_off = eb[idx] + self._bulk_offset(idx)
                sb[idx] = mat.stress_array(e_off)
                Et_b[idx] = mat.tangent_array(e_off)

        A = self.sec.A_fibers
        ly = self._ly_bulk
        lx = self._lx_bulk

        fA = sb * A
        N = float(np.sum(fA))
        Mx = float(np.sum(fA * ly))
        My = -float(np.sum(fA * lx))

        # Tangent stiffness from bulk fibers
        #   phi = [1, ly, -lx]
        #   K[i,j] = sum(Et * A * phi_i * phi_j)
        EtA = Et_b * A
        K = np.empty((3, 3))
        K[0, 0] = float(np.sum(EtA))
        K[0, 1] = float(np.sum(EtA * ly))
        K[0, 2] = -float(np.sum(EtA * lx))
        K[1, 0] = K[0, 1]
        K[1, 1] = float(np.sum(EtA * ly * ly))
        K[1, 2] = -float(np.sum(EtA * ly * lx))
        K[2, 0] = K[0, 2]
        K[2, 1] = K[1, 2]
        K[2, 2] = float(np.sum(EtA * lx * lx))

        # ---- Rebar contribution ----
        # See ``integrate``: each (rebar mat, zone mat) group subtracts
        # the displaced bulk stress and tangent using the zone's law.
        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            Et_rebar = mat.tangent_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))
            Et_bulk_r = bulk_mat.tangent_array(
                er_g + self._rebar_offset(idx))
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]
            ly_r = self._ly_rebar[idx]
            lx_r = self._lx_rebar[idx]

            s_net = s_rebar.copy()
            s_net[emb] -= sb_at_rebars[emb]

            Et_net = Et_rebar.copy()
            Et_net[emb] -= Et_bulk_r[emb]

            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * ly_r))
            My -= float(np.sum(fa * lx_r))

            EtA_r = Et_net * a
            K[0, 0] += float(np.sum(EtA_r))
            K[0, 1] += float(np.sum(EtA_r * ly_r))
            K[0, 2] -= float(np.sum(EtA_r * lx_r))
            K[1, 1] += float(np.sum(EtA_r * ly_r * ly_r))
            K[1, 2] -= float(np.sum(EtA_r * ly_r * lx_r))
            K[2, 2] += float(np.sum(EtA_r * lx_r * lx_r))

        # ---- Tendon contribution (prestress) ----
        # Decoupled-argument force and tangent; see _tendon_forces.
        if self._has_tendons:
            et = self._tendon_section_strain(eps0, chi_x, chi_y)
            Nt, Mxt, Myt, Kt = self._tendon_forces(
                et, want_tangent=True)
            N += Nt
            Mx += Mxt
            My += Myt
            K += Kt

        # Symmetric
        K[1, 0] = K[0, 1]
        K[2, 0] = K[0, 2]
        K[2, 1] = K[1, 2]

        return N, Mx, My, K

    # ==================================================================
    #  Batch integration — many configurations in one NumPy call
    # ==================================================================

    def integrate_batch(self, eps0, chi_x, chi_y):
        r"""
        Evaluate internal forces for many strain configurations at once.

        All inputs are 1-D arrays of the same length *n*.  The
        computation is fully vectorized: strain fields are built as
        2-D arrays of shape ``(n, n_fibers)`` and passed through the
        constitutive laws in one call, eliminating Python-loop
        overhead.

        Parameters
        ----------
        eps0 : numpy.ndarray
            Shape ``(n,)``.  Strains at the reference point.
        chi_x : numpy.ndarray
            Shape ``(n,)``.  Curvatures about the x-axis [1/mm].
        chi_y : numpy.ndarray
            Shape ``(n,)``.  Curvatures about the y-axis [1/mm].

        Returns
        -------
        N : numpy.ndarray
            Shape ``(n,)``.  Axial forces [N].
        Mx : numpy.ndarray
            Shape ``(n,)``.  Bending moments about x [N·mm].
        My : numpy.ndarray
            Shape ``(n,)``.  Bending moments about y [N·mm].
        """
        eps0 = np.asarray(eps0, dtype=np.float64)
        chi_x = np.asarray(chi_x, dtype=np.float64)
        chi_y = np.asarray(chi_y, dtype=np.float64)

        # Bulk strains: (n, n_fibers)
        eb = (eps0[:, None]
              + chi_x[:, None] * self._ly_bulk[None, :]
              - chi_y[:, None] * self._lx_bulk[None, :])

        # Bulk stresses: (n, n_fibers)
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
        else:
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[:, idx] = mat.stress_array(
                    eb[:, idx] + self._bulk_offset(idx))

        fA = sb * self.sec.A_fibers[None, :]        # (n, n_fibers)
        N = fA.sum(axis=1)                           # (n,)
        # einsum avoids allocating a full (n, n_fibers) temporary
        # for each moment component — saves one ~400 MB matrix at
        # peak when chunks are large.
        Mx = np.einsum('ij,j->i', fA, self._ly_bulk)
        My = -np.einsum('ij,j->i', fA, self._lx_bulk)

        # Rebar strains: (n, n_rebars)
        n_rebars = len(self.sec.y_rebars)

        # ---- Tendon contribution (prestress), batch ----
        # Computed before the rebar early-return so a section with
        # tendons but no rebars is handled correctly.
        if self._has_tendons:
            et = (eps0[:, None]
                  + chi_x[:, None] * self._ly_tendon[None, :]
                  - chi_y[:, None] * self._lx_tendon[None, :])
            Nt, Mxt, Myt = self._tendon_forces(et)
            N = N + Nt
            Mx = Mx + Mxt
            My = My + Myt

        if n_rebars == 0:
            return N, Mx, My

        er = (eps0[:, None]
              + chi_x[:, None] * self._ly_rebar[None, :]
              - chi_y[:, None] * self._lx_rebar[None, :])

        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[:, idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]
            ly_r = self._ly_rebar[idx]
            lx_r = self._lx_rebar[idx]

            s_net = s_rebar.copy()
            s_net[:, emb] -= sb_at_rebars[:, emb]

            fa = s_net * a[None, :]                  # (n, len(idx))
            N += fa.sum(axis=1)
            Mx += (fa * ly_r[None, :]).sum(axis=1)
            My -= (fa * lx_r[None, :]).sum(axis=1)

        return N, Mx, My

    # ==================================================================
    #  Numerical Jacobian (kept for validation / fallback)
    # ==================================================================

    def jacobian(self, eps0, chi_x, chi_y=0.0, deps=1e-8):
        r"""
        Numerical 3×3 Jacobian via forward finite differences.

        .. math::

            \mathbf{J} = \frac{\partial(N, M_x, M_y)}
                              {\partial(\varepsilon_0, \chi_x, \chi_y)}

        Parameters
        ----------
        eps0, chi_x, chi_y : float
        deps : float, optional
            Perturbation. Default 1e-8.

        Returns
        -------
        numpy.ndarray
            Shape (3, 3).
        """
        f0 = np.array(self.integrate(eps0, chi_x, chi_y))
        J = np.empty((3, 3))
        for j, (de, dx, dy) in enumerate([
            (deps, 0, 0), (0, deps, 0), (0, 0, deps)
        ]):
            f1 = np.array(self.integrate(
                eps0 + de, chi_x + dx, chi_y + dy))
            J[:, j] = (f1 - f0) / deps
        return J

    def _is_uniaxial(self, axis=None):
        r"""
        Detect if the section is effectively uniaxial.

        A section is uniaxial when one of its in-plane extents is
        degenerate, so the corresponding curvature has no
        integration support and the matching Jacobian column is
        singular.

        Two degenerate cases are recognised:

        - **vertical-degenerate**: every fiber shares the same
          :math:`x` coordinate.  :math:`\chi_y` has no effect, so
          bending is meaningful only about the :math:`x` axis.
        - **horizontal-degenerate**: every fiber shares the same
          :math:`y` coordinate.  :math:`\chi_x` has no effect, so
          bending is meaningful only about the :math:`y` axis.

        Parameters
        ----------
        axis : ``'x'``, ``'y'`` or ``None``, optional
            If specified, the test is restricted to the matching
            degeneracy: ``'x'`` returns ``True`` only for the
            vertical-degenerate case (chi_y singular), ``'y'`` only
            for the horizontal-degenerate case.  Without ``axis``,
            the test returns ``True`` for either degeneracy —
            preserving the legacy single-argument semantics.

        Returns
        -------
        bool
        """
        # Range of lever arms along x (sensitivity to chi_y).
        if len(self._lx_rebar) > 0:
            lx_range = (np.max(np.abs(self._lx_bulk))
                        + np.max(np.abs(self._lx_rebar)))
        else:
            lx_range = np.max(np.abs(self._lx_bulk))

        # Range of lever arms along y (sensitivity to chi_x).
        if len(self._ly_rebar) > 0:
            ly_range = (np.max(np.abs(self._ly_bulk))
                        + np.max(np.abs(self._ly_rebar)))
        else:
            ly_range = np.max(np.abs(self._ly_bulk))

        vertical_deg = lx_range < 1e-6     # bending about x
        horizontal_deg = ly_range < 1e-6   # bending about y

        if axis == 'x':
            return vertical_deg
        if axis == 'y':
            return horizontal_deg
        return vertical_deg or horizontal_deg

    # ==================================================================
    #  Main solver entry point
    # ==================================================================

    #: Default resolution of the zero-moment reach scan, as
    #: ``(n_chi, n_eps)``.  Deliberately modest: the scan runs only on
    #: a failure path, once per solver instance, and its purpose is to
    #: separate "no such state" from "search failed", which a coarse
    #: measurement already does.  The numbers travel inside
    #: :class:`AxialStateNotAttainable` so a reader can see how coarse
    #: the certificate is.
    AXIAL_REACH_RESOLUTION = (61, 250)

    def axial_reach_at_zero_moment(self, M_tol=1.0e3, resolution=None):
        r"""
        The axial force this section can carry with zero moment.

        Measures, rather than bounds, the interval

        .. math::

            \bigl[\, \min N,\ \max N \,\bigr]
            \quad \text{over} \quad
            \{ (\varepsilon_0, \chi_x) : |M_x(\varepsilon_0, \chi_x)|
            \le M_{tol} \}

        by walking a grid of curvatures, locating every sign change of
        :math:`M_x(\varepsilon_0)` along each of them, refining it by
        bisection, and keeping the extreme axial forces among the
        roots that verify.

        Why a scan and not a formula
        ----------------------------
        The set is not convex and its boundary is not smooth: concrete
        carries no tension, so :math:`M_x` has plateaux where the
        section is wholly cracked and jumps where a fibre leaves the
        active range.  A closed-form bound exists (see
        :class:`AxialStateNotAttainable`) but is loose enough to call
        an attainable state unattainable, which is the one error this
        measurement must not make.

        Parameters
        ----------
        M_tol : float, optional
            Moment tolerance the reach is measured under [N.mm].
            Default ``1e3``, matching ``solve_equilibrium``'s own
            ``tol * 1000`` at the default force tolerance.
        resolution : tuple of int, optional
            ``(n_chi, n_eps)``.  Default
            :attr:`AXIAL_REACH_RESOLUTION`.

        Returns
        -------
        dict
            ``N_min``, ``N_max`` : float
                Extremes of the attainable axial force [N].
            ``state_min``, ``state_max`` : tuple of float
                ``(eps0, chi_x)`` achieving each extreme.
            ``roots`` : int
                Number of verified :math:`M_x = 0` roots found.  Zero
                means the scan found nothing and the result must not
                be used as a certificate.
            ``resolution`` : tuple of int
                The grid actually used.
            ``M_tol`` : float
                The moment tolerance the reach was measured under [N.mm].

        Notes
        -----
        The result is cached on the instance under the resolution and
        tolerance it was computed with, because the only caller is a
        failure path that may be reached many times in one diagram and
        the section does not change between calls.

        Examples
        --------
        >>> reach = solver.axial_reach_at_zero_moment()   # doctest: +SKIP
        >>> reach["N_max"]                                # doctest: +SKIP
        317016.36
        """
        n_chi, n_eps = (self.AXIAL_REACH_RESOLUTION if resolution is None
                        else (int(resolution[0]), int(resolution[1])))
        cache_key = (n_chi, n_eps, float(M_tol))
        cached = self._axial_reach_cache.get(cache_key)
        if cached is not None:
            return cached

        # Strain window: the same element-complete range the axial
        # bracket contracts to cover (N-15), widened by 1 % so a root
        # sitting exactly on a limit is inside the sampled interval.
        eps_lo = float(self.sec.bulk_material.eps_min)
        eps_hi = 0.0
        for r in self.sec.rebars:
            eps_lo = min(eps_lo, float(r.material.eps_min))
            eps_hi = max(eps_hi, float(r.material.eps_max))
        for mat, _bulk, _idx, eps_init in self._tendon_groups:
            eps_hi = max(eps_hi,
                         float(mat.eps_max) - float(np.min(eps_init)))
            eps_lo = min(eps_lo,
                         float(mat.eps_min) - float(np.max(eps_init)))
        eps_lo *= 1.01
        eps_hi = eps_hi * 1.01 if eps_hi > 0.0 else 1.0e-3

        # Curvatures: logarithmic on both signs plus zero, because the
        # interesting roots sit over four decades of curvature and a
        # linear grid would spend most of its points where nothing
        # happens.
        half = max(1, (n_chi - 1) // 2)
        mags = np.logspace(-8.0, -2.0, half)
        curvatures = np.concatenate([-mags[::-1], [0.0], mags])

        grid = np.linspace(eps_lo, eps_hi, n_eps)
        zeros = np.zeros_like(grid)
        N_min = float("inf")
        N_max = float("-inf")
        state_min = state_max = (0.0, 0.0)
        roots = 0
        N_roots = []

        my_peak = 0.0
        for chi in curvatures:
            Nv, Mxv, Myv = self.integrate_batch(
                grid, np.full_like(grid, chi), zeros)
            # Largest |My| anywhere on the scanned plane.  It is what
            # decides whether the plane is the whole search space or
            # only a slice of it -- measured, not inferred from the
            # geometry declaration.
            my_peak = max(my_peak, float(np.max(np.abs(Myv))))
            crossings = np.where(np.diff(np.sign(Mxv)) != 0)[0]
            for i in crossings:
                a, b, fa = grid[i], grid[i + 1], Mxv[i]
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    _, fm, _ = self.integrate(mid, chi, 0.0)
                    if fa * fm <= 0.0:
                        b = mid
                    else:
                        a, fa = mid, fm
                eps_root = 0.5 * (a + b)
                N_root, Mx_root, My_root = self.integrate(
                    eps_root, chi, 0.0)
                if abs(Mx_root) > M_tol:
                    continue            # a jump, not a root
                # BOTH moments, not one.  The request this measurement
                # answers is ``Mx = My = 0``; a state that zeroes Mx
                # alone is not a state that satisfies it.  Measured on
                # examples/Pannello_187x30.yaml, whose element layout
                # is asymmetric about the y-axis: the Mx = 0 extremum
                # there carries My = 4.736e+08 N.mm, and counting it
                # made the reach look 2 042 789 N wide and produced a
                # 'search_failed_inside_reach' verdict — a claim that
                # a defect exists, issued by a half-blind instrument.
                if abs(My_root) > M_tol:
                    continue
                roots += 1
                N_roots.append(float(N_root))
                if N_root < N_min:
                    N_min, state_min = float(N_root), (eps_root, float(chi))
                if N_root > N_max:
                    N_max, state_max = float(N_root), (eps_root, float(chi))

        if roots == 0:
            N_min = N_max = 0.0
        result = {
            "N_min": N_min, "N_max": N_max,
            "state_min": state_min, "state_max": state_max,
            "roots": roots, "N_roots": tuple(sorted(N_roots)),
            "my_peak": my_peak,
            "on_plane_only": my_peak > float(M_tol),
            "resolution": (n_chi, n_eps),
            "M_tol": float(M_tol),
        }
        self._axial_reach_cache[cache_key] = result
        return result

    def _refuse_unattainable_axial(self, sol, N_target, M_tol):
        r"""
        Turn a mute pure-axial non-convergence into a named outcome.

        Called on exactly one route: both target moments were
        negligible, the whole solve chain failed, and the caller is
        about to receive ``converged = False`` with no reason.  This
        method measures the section's zero-moment reach and either

        * labels it ``axial_state_not_attainable``, when the request
          is outside the reach — the state does not exist and no
          search will find it; or
        * labels it ``search_failed_inside_reach``, when the request
          is *inside* the reach — the state exists and the search
          missed it, which is a defect and is named as one rather
          than hidden.

        **It does not raise**, and that is a deliberate choice with a
        cost attached.  ``solve_equilibrium`` is called inside loops
        that generate interaction domains and moment-curvature
        curves, and those loops treat a non-converged point as a
        point to skip.  Turning the return into an exception would
        abort a whole diagram on one unattainable point, which is a
        capability removed, not a capability added.  The refusal is
        therefore carried **in the return contract**, and
        :meth:`require_equilibrium` is the opt-in that promotes it to
        :class:`AxialStateNotAttainable` for a caller that wants a
        verdict rather than a curve.

        Parameters
        ----------
        sol : dict
            The non-converged result about to be returned.
        N_target : float
            Requested axial force [N].
        M_tol : float
            Moment tolerance in force [N.mm].

        Returns
        -------
        dict
            *sol*, with these keys added:

            ``reason`` : str
                One of ``'axial_state_not_attainable'``,
                ``'search_failed_inside_reach'``,
                ``'reach_not_measurable'``.
            ``refusal`` : str
                The reason in prose, with the numbers in it.
            ``residual_rel`` : float
                Relative force residual of the state returned.
            ``attainable_N`` : tuple of float
                The measured zero-moment reach ``(N_min, N_max)``.
            ``attainable_state`` : tuple of float, optional
                ``(eps0, chi_x)`` of the nearest state that exists;
                present only when the request is outside the reach.
            ``reach_resolution`` : tuple of int
                Coarseness of the scan behind the two above.

        Notes
        -----
        The verdict never travels alone.  ``residual_rel`` is written
        next to it because ``converged`` is meaningless above certain
        residual magnitudes -- the same reason the timeline dump
        carries the relative residual and the tolerance alongside the
        solver flag instead of publishing the flag by itself.

        A reach measurement that finds **no** root is not evidence of
        anything, and this method says so: it labels the outcome
        ``reach_not_measurable`` and refuses to raise on it.  A
        certificate produced by a blind instrument is the failure mode
        this project spent Phase 19 removing.
        """
        scale = max(abs(N_target), 1.0)
        sol["residual_rel"] = abs(
            float(sol.get("N", float("nan"))) - N_target) / scale
        # The tolerance the reach was measured under travels with the
        # verdict.  Re-deriving it in ``require_equilibrium`` from the
        # default argument would be a second definition of one fact.
        sol["M_tol"] = float(M_tol)
        reach = self.axial_reach_at_zero_moment(M_tol=M_tol)
        sol["attainable_N"] = (reach["N_min"], reach["N_max"])
        sol["reach_resolution"] = reach["resolution"]
        if reach["roots"] == 0:
            sol["reason"] = "reach_not_measurable"
            sol["refusal"] = (
                "the zero-moment reach of this section could not be "
                "measured: the scan found no verified Mx = 0 root, so "
                "nothing here is evidence either way.")
            return sol
        outside = (N_target > reach["N_max"] or N_target < reach["N_min"])
        if outside and reach["on_plane_only"]:
            # The scan holds chi_y = 0.  On a section that is not
            # degenerate about x the solver may search chi_y as well,
            # so a target outside the scanned plane is NOT proven
            # unattainable: the measurement is a lower bound on the
            # reach, not the reach.  Report the bound and decline the
            # certificate.  Claiming "no such state" from a plane one
            # did not visit is the error this whole item is about.
            sol["reason"] = "axial_reach_not_certified"
            sol["refusal"] = (
                f"N = {N_target:.1f} N is outside the zero-moment "
                f"reach measured on the chi_y = 0 plane, "
                f"[{reach['N_min']:.1f}, {reach['N_max']:.1f}] N. "
                f"That plane is not the whole search space here: the "
                f"largest |My| on it is {reach['my_peak']:.4g} N.mm, "
                f"above the {M_tol:g} N.mm tolerance, so the section "
                f"is not symmetric about the reference y-axis and "
                f"states with chi_y != 0 exist that the scan never "
                f"visited.  The bound is a LOWER bound on the reach "
                f"and is not a certificate of absence.")
            return sol
        if outside:
            state = (reach["state_max"] if N_target > reach["N_max"]
                     else reach["state_min"])
            sol["reason"] = "axial_state_not_attainable"
            sol["attainable_state"] = state
            sol["refusal"] = str(AxialStateNotAttainable(
                N_target, (reach["N_min"], reach["N_max"]),
                state, M_tol, reach["resolution"]))
            return sol
        # Inside the range is not the same as attainable: the reach is
        # sampled, not solved, and the attainable set has gaps where a
        # fibre leaves the active range.  Only a verified root CLOSE to
        # the target licenses the word "defect".
        gap = min(abs(n - N_target) for n in reach["N_roots"])
        if gap <= 0.01 * max(abs(N_target), 1.0):
            sol["reason"] = "search_failed_inside_reach"
            sol["refusal"] = (
                f"N = {N_target:.1f} N is inside this section's "
                f"measured zero-moment reach [{reach['N_min']:.1f}, "
                f"{reach['N_max']:.1f}] N and a verified state sits "
                f"{gap:.1f} N away from it, so a state exists and the "
                f"search did not find it.  This is a defect, not a "
                f"boundary.")
            return sol
        sol["reason"] = "reach_inconclusive"
        sol["refusal"] = (
            f"N = {N_target:.1f} N lies between the extremes of the "
            f"measured zero-moment reach [{reach['N_min']:.1f}, "
            f"{reach['N_max']:.1f}] N, but the nearest verified state "
            f"is {gap:.1f} N away and the attainable set has gaps.  "
            f"The scan, at {reach['resolution'][0]} curvatures x "
            f"{reach['resolution'][1]} strains, cannot decide whether "
            f"the state exists; it is not evidence either way.")
        return sol

    def require_equilibrium(self, N_target, Mx_target, My_target=0.0,
                            **kwargs):
        r"""
        :meth:`solve_equilibrium`, but a refusal is raised.

        For the caller that wants a **verdict** rather than a point on
        a curve.  Behaves exactly like :meth:`solve_equilibrium` and
        returns its dict, except that a result labelled
        ``axial_state_not_attainable`` is turned into
        :class:`AxialStateNotAttainable`.

        Why the two are separate methods
        --------------------------------
        Interaction domains and moment-curvature curves call the
        solver in loops and treat non-convergence as a point to skip.
        If the solver raised, one unattainable point would abort a
        whole diagram: a capability removed.  Keeping the raise out of
        the default path and offering it here adds the strict
        behaviour without taking the lenient one away.

        Parameters
        ----------
        N_target, Mx_target, My_target : float
            As :meth:`solve_equilibrium`.
        **kwargs
            Forwarded verbatim.

        Returns
        -------
        dict
            The solution, as :meth:`solve_equilibrium`.

        Raises
        ------
        AxialStateNotAttainable
            When the request is a pure-axial one that the section's
            measured zero-moment reach does not contain.

        See Also
        --------
        solve_equilibrium : the lenient form, which labels instead.
        axial_reach_at_zero_moment : the measurement behind the label.
        """
        sol = self.solve_equilibrium(N_target, Mx_target, My_target,
                                     **kwargs)
        if sol.get("reason") == "axial_state_not_attainable":
            raise AxialStateNotAttainable(
                N_target, sol["attainable_N"], sol["attainable_state"],
                sol.get("M_tol", kwargs.get("tol", 1e-3) * 1000.0),
                sol["reach_resolution"])
        return sol

    def solve_equilibrium(self, N_target, Mx_target, My_target=0.0,
                          eps0_init=0.0, chi_x_init=1e-6,
                          chi_y_init=0.0,
                          tol=1e-3, max_iter=50):
        r"""
        Inverse problem: find strain plane for target (N, Mx, My).

        Newton-Raphson with **analytical tangent stiffness** and
        backtracking line search.  Automatically reduces to a 2×2
        system when the section is uniaxial, and further to a
        **1×1 bisection** when both target moments are negligible
        (pure axial load).

        The pure-axial branch exploits the monotonicity of
        :math:`N(\varepsilon_0)` at :math:`\chi_x = \chi_y = 0`:

        .. math::

            \frac{\partial N}{\partial \varepsilon_0}\bigg|_{\chi=0}
            = \sum_i \frac{\partial \sigma_i}
              {\partial \varepsilon}\,A_i \;\ge\; 0

        which guarantees unique bracketing and convergence of the
        bisection search.

        Parameters
        ----------
        N_target : float
            Target axial force [N].
        Mx_target : float
            Target moment about x-axis [N*mm].
        My_target : float, optional
            Target moment about y-axis [N*mm]. Default 0.
        eps0_init, chi_x_init, chi_y_init : float, optional
            Initial guesses.
        tol : float, optional
            Force tolerance [N]; moment tolerance is ``tol * 1000``.
        max_iter : int, optional

        Returns
        -------
        dict
            Keys: ``eps0``, ``chi_x``, ``chi_y``, ``N``, ``Mx``,
            ``My``, ``converged``, ``iterations``.
        """
        M_tol = tol * 1000  # moment tolerance [N·mm]

        # ----------------------------------------------------------
        #  Fast path: pure axial load (both moments negligible)
        # ----------------------------------------------------------
        if abs(Mx_target) < M_tol and abs(My_target) < M_tol:
            sol = self._solve_pure_axial(N_target, tol, max_iter)
            if (sol["converged"]
                    and abs(sol["Mx"]) < M_tol
                    and abs(sol["My"]) < M_tol):
                return sol
            # Fall through: a converged pure-axial (chi=0) state can
            # still violate moment equilibrium when the section carries
            # eccentric *internal* self-stress (e.g. an eccentric bonded
            # tendon at prestress transfer). chi=0 cannot balance that
            # moment, so the full uniaxial/biaxial solve is required.
            #
            # N-15, the contract half.  This fall-through is a
            # DECLARED route for a fast path that legitimately does
            # not apply -- it is not, and must not be used as, a
            # rescue for a fast path that is WRONG.  Until Phase 19
            # it silently absorbed the missing tendon branch of
            # ``_solve_pure_axial``: a strand-only section whose
            # tensile target had no bracket came back non-converged
            # here and was re-solved by ``_solve_uniaxial``, so the
            # caller saw a right answer produced for the wrong
            # reason and no test could see the defect from outside.
            # The bracket is fixed at its own site; what is fixed
            # here is the standing of this branch, which is now a
            # stated contract with a named scope instead of an
            # accidental screen.  ``_solve_pure_axial`` is
            # independently answerable for covering every element
            # class, and is tested directly, not through this route.

        # N-17.  Everything below this point may fail; when the
        # request was a PURE AXIAL one, a bare ``converged = False``
        # cannot say which of two very different things happened, so
        # the corridor is marked here and every exit through it is
        # given a reason before it reaches the caller.
        pure_axial = (abs(Mx_target) < M_tol and abs(My_target) < M_tol)

        # ----------------------------------------------------------
        #  Uniaxial or near-uniaxial
        # ----------------------------------------------------------
        # Vertically-degenerate section: only chi_x is meaningful,
        # solve about the x-axis.
        if self._is_uniaxial(axis='x'):
            sol = self._solve_uniaxial(
                N_target, Mx_target, eps0_init, chi_x_init,
                tol, max_iter)
            if pure_axial and not sol["converged"]:
                return self._refuse_unattainable_axial(
                    sol, N_target, M_tol)
            return sol

        # Horizontally-degenerate section: only chi_y is meaningful,
        # solve about the y-axis.
        if self._is_uniaxial(axis='y'):
            sol = self._solve_uniaxial_y(
                N_target, My_target, eps0_init, chi_y_init,
                tol, max_iter)
            if pure_axial and not sol["converged"]:
                return self._refuse_unattainable_axial(
                    sol, N_target, M_tol)
            return sol

        # Near-uniaxial fast paths.  When one of the target moments
        # is negligible, try the corresponding 2-unknown solve in
        # the dominant plane and accept it only if the spurious
        # moment stays inside tolerance.  Both branches are present
        # for symmetry — previously only the My-near-zero branch
        # existed, penalising demands dominated by My.
        if abs(My_target) < M_tol:
            sol = self._solve_uniaxial(
                N_target, Mx_target, eps0_init, chi_x_init,
                tol, max_iter)
            if sol["converged"] and abs(sol["My"]) < M_tol:
                return sol

        if abs(Mx_target) < M_tol:
            sol = self._solve_uniaxial_y(
                N_target, My_target, eps0_init, chi_y_init,
                tol, max_iter)
            if sol["converged"] and abs(sol["Mx"]) < M_tol:
                return sol

        sol = self._solve_biaxial(
            N_target, Mx_target, My_target,
            eps0_init, chi_x_init, chi_y_init,
            tol, max_iter)
        if pure_axial and not sol["converged"]:
            return self._refuse_unattainable_axial(sol, N_target, M_tol)
        return sol

    # ------------------------------------------------------------------
    #  Pure axial solver  (1-unknown bisection)
    # ------------------------------------------------------------------

    def _solve_pure_axial(self, N_target, tol, max_iter):
        r"""
        Find :math:`\varepsilon_0` for pure axial load
        (:math:`\chi_x = \chi_y = 0`).

        The function :math:`N(\varepsilon_0)` at zero curvature is
        **not globally monotone** because the concrete constitutive
        law returns zero stress for strains beyond
        :math:`\varepsilon_{cu2}`, creating a non-monotone drop at
        the compressive end.  However, :math:`N` *is* monotonically
        non-decreasing on the sub-interval
        :math:`[\varepsilon_{c2},\, \varepsilon_{su}]` where all
        constitutive laws are active, and the vast majority of
        practical targets lie in this range.

        Strategy:

        1.  Scan :math:`N(\varepsilon_0)` on a fine grid spanning
            the full material strain range — using
            :meth:`integrate_batch` for speed.
        2.  Find consecutive grid points where :math:`N` crosses
            :math:`N_{\text{target}}`.  Among all crossings, pick
            the one in the monotone working region (closest to the
            centroid strain).
        3.  Bisect within that local bracket.

        Parameters
        ----------
        N_target : float
            Target axial force [N].
        tol : float
            Convergence tolerance on N [N].
        max_iter : int
            Maximum bisection iterations.

        Returns
        -------
        dict

        Notes
        -----
        **Scan-range contract (N-15).**  The grid of step 1 must span
        every section strain at which the section can be in axial
        equilibrium, and it is built from **all three** element
        classes -- bulk, ordinary reinforcement and tendons -- with
        each tendon's own limits shifted by its locked-in strain,
        :math:`\varepsilon_{\mathrm{sec}} = \varepsilon_p
        - \varepsilon_{\mathrm{init}}`.  A range assembled from a
        subset of the classes leaves part of the section's real
        capacity with no bracket, and the routine then reports
        ``converged=False`` for a load the section carries.

        This routine is answerable for that on its own.  Its caller,
        :meth:`solve_equilibrium`, routes a non-converged pure-axial
        result onward to the uniaxial or biaxial solve, but that
        route exists for the case where a zero-curvature state
        genuinely cannot satisfy moment equilibrium -- **not** as a
        rescue for an incorrect bracket here.  Correctness of this
        routine is therefore tested by calling it directly.
        """
        sec = self.sec

        # --- Build the full strain scan range ---
        eps_lo = sec.bulk_material.eps_min
        eps_hi = 0.0
        for r in sec.rebars:
            eps_lo = min(eps_lo, r.material.eps_min)
            eps_hi = max(eps_hi, r.material.eps_max)
        # --- Tendons (N-15) ---
        # The scan range is a CONTRACT: it must span every strain the
        # section can be in equilibrium at, or a target inside the
        # section's real capacity has no bracket and the routine
        # returns non-converged for a load the section carries.
        # Before this it walked bulk and rebar only, so a tendon was
        # absent from the range even though it is present in the
        # integral the range is scanned with -- the same two-of-three
        # assembly as N-14 (strain envelope) and N-16 (scan hull),
        # here on the axial bracket.
        #
        # A tendon's own limits enter the SECTION strain range shifted
        # by the locked-in strain its law is evaluated on top of:
        #
        #     eps_p = eps_sec + eps_init  =>  eps_sec = eps_p - eps_init
        #
        # On a section that also carries ordinary reinforcement the
        # rebar limit is normally the wider one and this is inert; on
        # a strand-only section -- a hollow-core or double-tee slab,
        # the same geometry N-14 names -- ``eps_hi`` stayed at exactly
        # 0.0 and the whole tensile branch of the scan did not exist.
        # Measured on a 1200 x 200 slab with one 700 mm^2 strand at
        # eps_pe = 0.0060: peak N reachable by the old range 819.0 kN
        # against a true axial tensile capacity of 1112.3 kN, i.e.
        # **26.37 % of the real capacity had no bracket**, and targets
        # at 90 / 95 / 99 % of it all returned non-converged.
        for _mat, _bulk_mat, _idx, eps_init in self._tendon_groups:
            e0_min = float(np.min(eps_init))
            e0_max = float(np.max(eps_init))
            eps_lo = min(eps_lo, float(_mat.eps_min) - e0_max)
            eps_hi = max(eps_hi, float(_mat.eps_max) - e0_min)
        eps_lo *= 1.01
        eps_hi *= 1.01

        # --- Batch scan ---
        n_scan = 120
        eps_vals = np.linspace(eps_lo, eps_hi, n_scan)
        chi_zero = np.zeros(n_scan)
        N_vals, _, _ = self.integrate_batch(eps_vals, chi_zero, chi_zero)

        # --- Find all crossings of N_target ---
        crossings = []
        for k in range(n_scan - 1):
            if ((N_vals[k] - N_target) * (N_vals[k + 1] - N_target) <= 0
                    and abs(N_vals[k + 1] - N_vals[k]) > 0.1):
                crossings.append(k)

        if not crossings:
            return self._fail_result()

        # Pick the best crossing: prefer the one closest to eps0=0
        # (i.e. in the working monotone region, not in the
        # non-physical tail beyond eps_cu2).
        best_k = min(crossings,
                     key=lambda k: abs(eps_vals[k] + eps_vals[k + 1]))

        a = eps_vals[best_k]
        b = eps_vals[best_k + 1]
        Na = N_vals[best_k]

        # --- Bisection within the local bracket ---
        for i in range(max_iter):
            mid = 0.5 * (a + b)
            N_mid, Mx_mid, My_mid = self.integrate(mid, 0.0, 0.0)
            if abs(N_mid - N_target) < tol:
                return {"eps0": mid, "chi_x": 0.0, "chi_y": 0.0,
                        "N": N_mid, "Mx": Mx_mid, "My": My_mid,
                        "converged": True, "iterations": i + 1}
            if (N_mid - N_target) * (Na - N_target) <= 0:
                b = mid
            else:
                a = mid
                Na = N_mid

        mid = 0.5 * (a + b)
        N_f, Mx_f, My_f = self.integrate(mid, 0.0, 0.0)
        converged = abs(N_f - N_target) < tol
        return {"eps0": mid, "chi_x": 0.0, "chi_y": 0.0,
                "N": N_f, "Mx": Mx_f, "My": My_f,
                "converged": converged, "iterations": max_iter}

    @staticmethod
    def _fail_result():
        """Return a non-converged result placeholder."""
        return {"eps0": 0.0, "chi_x": 0.0, "chi_y": 0.0,
                "N": 0.0, "Mx": 0.0, "My": 0.0,
                "converged": False, "iterations": 0}

    # ------------------------------------------------------------------
    #  Internal: elastic initial guess
    # ------------------------------------------------------------------

    def _elastic_initial_guess(self, N_target, Mx_target, My_target=0.0):
        r"""
        Estimate initial (eps0, chi_x, chi_y) from linear-elastic theory.

        Uses the **analytical** tangent stiffness at the origin,
        which is the elastic stiffness matrix.

        Returns
        -------
        eps0, chi_x, chi_y : float
        
        """
        ### TODO: we should use ideal gross section properties here, 
        ### to get a more accurate initial guess. 
        try:
            _, _, _, K0 = self.integrate_with_tangent(0.0, 0.0, 0.0)
            target = np.array([N_target, Mx_target, My_target])
            x0 = np.linalg.solve(K0, target)
            x0[0] = np.clip(x0[0], -0.003, 0.003)
            x0[1] = np.clip(x0[1], -1e-4, 1e-4)
            x0[2] = np.clip(x0[2], -1e-4, 1e-4)
            return float(x0[0]), float(x0[1]), float(x0[2])
        except np.linalg.LinAlgError:
            sec = self.sec
            A_ideal_gross = getattr(sec, 'ideal_gross_area',
                                    sec.B * sec.H)
            # Independent moments of inertia for each bending axis.
            I_x_approx = (A_ideal_gross * sec.H ** 2 / 12
                          if sec.H > 0 else 0.0)
            I_y_approx = (A_ideal_gross * sec.B ** 2 / 12
                          if sec.B > 0 else 0.0)
            ### TODO: substitute 30000 with real bulk base modulus of the section
            E_eff = 30000.0
            eps0_est = N_target / (A_ideal_gross * E_eff)
            chi_x_est = (Mx_target / (E_eff * I_x_approx)
                         if I_x_approx > 0 else 1e-6)
            chi_y_est = (My_target / (E_eff * I_y_approx)
                         if I_y_approx > 0 else 0.0)
            return (float(eps0_est),
                    float(chi_x_est),
                    float(chi_y_est))

    # ------------------------------------------------------------------
    #  Uniaxial solver (analytical Jacobian)
    # ------------------------------------------------------------------

    def _solve_uniaxial(self, N_target, Mx_target,
                        eps0_init, chi_x_init, tol, max_iter):
        r"""
        2-unknown solver (:math:`\chi_y` fixed at 0).

        Strategy:

        1. Try elastic initial guess → Newton-Raphson.
        2. If that fails, warm-start from a pure-axial solve at
           :math:`N = N_{\text{target}}` with a small perturbation
           in :math:`\chi_x` to break the singularity.
        3. Multi-start grid of :math:`\chi_x` values.

        Parameters
        ----------
        N_target, Mx_target : float
        eps0_init, chi_x_init : float
        tol : float
        max_iter : int

        Returns
        -------
        dict
        """
        # --- Attempt 1: user-supplied or elastic guess ---
        if eps0_init == 0.0 and chi_x_init == 1e-6:
            e0, c0, _ = self._elastic_initial_guess(
                N_target, Mx_target, 0.0)
        else:
            e0, c0 = eps0_init, chi_x_init

        sol = self._nr_uniaxial(N_target, Mx_target, e0, c0,
                                tol, max_iter)
        if sol["converged"]:
            return sol

        # --- Attempt 2: warm-start from pure-axial bisection ---
        sol_axial = self._solve_pure_axial(N_target, tol, max_iter)
        if sol_axial["converged"]:
            e0_ax = sol_axial["eps0"]
            # Estimate chi_x from the sign of Mx_target
            chi_sign = 1.0 if Mx_target >= 0 else -1.0
            for chi_mag in [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]:
                sol2 = self._nr_uniaxial(
                    N_target, Mx_target,
                    e0_ax, chi_sign * chi_mag,
                    tol, max_iter)
                if sol2["converged"]:
                    return sol2

        # --- Attempt 3: multi-start grid ---
        emb = self.sec.bulk_material.eps_min
        for chi_sign in [1.0, -1.0]:
            for chi_mag in [1e-5, 5e-6, 2e-5, 3e-5, 5e-5]:
                chi_try = chi_sign * chi_mag
                try:
                    N0, _, _ = self.integrate(0.0, chi_try, 0.0)
                    N1, _, _ = self.integrate(1e-6, chi_try, 0.0)
                    dNde = (N1 - N0) / 1e-6
                    if abs(dNde) > 1:
                        eps0_try = (N_target - N0) / dNde
                        eps0_try = np.clip(eps0_try, emb, -emb)
                    else:
                        eps0_try = emb / 2
                except Exception:
                    eps0_try = emb / 2

                sol = self._nr_uniaxial(N_target, Mx_target,
                                        eps0_try, chi_try,
                                        tol, max_iter // 2)
                if sol["converged"]:
                    return sol

        return self._nr_uniaxial(N_target, Mx_target, e0, c0,
                                 tol, max_iter)

    def _nr_uniaxial(self, N_target, Mx_target, eps0, chi_x,
                     tol, max_iter):
        r"""
        Core Newton-Raphson for uniaxial case.

        Uses the **analytical** 2×2 sub-block of the tangent
        stiffness matrix instead of finite-difference perturbations.
        """
        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(eps0, chi_x, 0.0)
            r = np.array([N - N_target, Mx - Mx_target])

            if abs(r[0]) < tol and abs(r[1]) < tol * 1000:
                return {
                    "eps0": eps0, "chi_x": chi_x, "chi_y": 0.0,
                    "N": N, "Mx": Mx, "My": My,
                    "converged": True, "iterations": i + 1,
                }

            # Extract 2×2 sub-block [dN/de, dN/dc; dMx/de, dMx/dc]
            J = K[:2, :2]
            det = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            if det < 1e-20:
                eps0 += 1e-5
                chi_x += 1e-7
                continue

            try:
                d = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                eps0 += 1e-5
                chi_x += 1e-7
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                e_new = eps0 + alpha * d[0]
                c_new = chi_x + alpha * d[1]
                Nn, Mxn, _ = self.integrate(e_new, c_new, 0.0)
                rn = np.array([Nn - N_target, Mxn - Mx_target])
                if np.linalg.norm(rn) < r_norm:
                    break
                alpha *= 0.5
            eps0 += alpha * d[0]
            chi_x += alpha * d[1]

        N, Mx, My = self.integrate(eps0, chi_x, 0.0)
        return {
            "eps0": eps0, "chi_x": chi_x, "chi_y": 0.0,
            "N": N, "Mx": Mx, "My": My,
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Uniaxial solver about the y-axis  (mirror of _solve_uniaxial)
    # ------------------------------------------------------------------

    def _solve_uniaxial_y(self, N_target, My_target,
                          eps0_init, chi_y_init, tol, max_iter):
        r"""
        2-unknown solver with :math:`\chi_x` fixed at 0.

        Mirror image of :meth:`_solve_uniaxial`, applicable when
        the bending is dominant about the :math:`y` axis (target
        :math:`M_x \approx 0`) or when the section is degenerate
        horizontally (:meth:`_is_uniaxial(axis='y')`).

        Strategy is identical to the x-axis variant: elastic guess
        → Newton, axial warm-start with sign-aware curvature, and
        a multi-start grid as last resort.

        Parameters
        ----------
        N_target, My_target : float
        eps0_init, chi_y_init : float
        tol : float
        max_iter : int

        Returns
        -------
        dict
            ``eps0``, ``chi_x`` (always 0), ``chi_y``, ``N``, ``Mx``,
            ``My``, ``converged``, ``iterations``.
        """
        # --- Attempt 1: user-supplied or elastic guess ---
        if eps0_init == 0.0 and chi_y_init == 0.0:
            e0, _, c0 = self._elastic_initial_guess(
                N_target, 0.0, My_target)
            if c0 == 0.0:
                # Elastic-fallback path may return chi_y = 0; nudge.
                c0 = 1e-6 if My_target >= 0 else -1e-6
        else:
            e0, c0 = eps0_init, chi_y_init

        sol = self._nr_uniaxial_y(N_target, My_target, e0, c0,
                                  tol, max_iter)
        if sol["converged"]:
            return sol

        # --- Attempt 2: warm-start from pure-axial bisection ---
        sol_axial = self._solve_pure_axial(N_target, tol, max_iter)
        if sol_axial["converged"]:
            e0_ax = sol_axial["eps0"]
            chi_sign = 1.0 if My_target >= 0 else -1.0
            for chi_mag in [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]:
                sol2 = self._nr_uniaxial_y(
                    N_target, My_target,
                    e0_ax, chi_sign * chi_mag,
                    tol, max_iter)
                if sol2["converged"]:
                    return sol2

        # --- Attempt 3: multi-start grid ---
        emb = self.sec.bulk_material.eps_min
        for chi_sign in [1.0, -1.0]:
            for chi_mag in [1e-5, 5e-6, 2e-5, 3e-5, 5e-5]:
                chi_try = chi_sign * chi_mag
                try:
                    N0, _, _ = self.integrate(0.0, 0.0, chi_try)
                    N1, _, _ = self.integrate(1e-6, 0.0, chi_try)
                    dNde = (N1 - N0) / 1e-6
                    if abs(dNde) > 1:
                        eps0_try = (N_target - N0) / dNde
                        eps0_try = np.clip(eps0_try, emb, -emb)
                    else:
                        eps0_try = emb / 2
                except Exception:
                    eps0_try = emb / 2

                sol = self._nr_uniaxial_y(N_target, My_target,
                                          eps0_try, chi_try,
                                          tol, max_iter // 2)
                if sol["converged"]:
                    return sol

        return self._nr_uniaxial_y(N_target, My_target, e0, c0,
                                   tol, max_iter)

    def _nr_uniaxial_y(self, N_target, My_target, eps0, chi_y,
                       tol, max_iter):
        r"""
        Core Newton-Raphson for uniaxial bending about the y-axis.

        Uses the **analytical** 2×2 sub-block of the tangent
        stiffness matrix corresponding to
        :math:`(\varepsilon_0, \chi_y)`, i.e. rows/cols 0 and 2 of
        the full 3×3 tangent.  Mirror image of :meth:`_nr_uniaxial`.
        """
        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(eps0, 0.0, chi_y)
            r = np.array([N - N_target, My - My_target])

            if abs(r[0]) < tol and abs(r[1]) < tol * 1000:
                return {
                    "eps0": eps0, "chi_x": 0.0, "chi_y": chi_y,
                    "N": N, "Mx": Mx, "My": My,
                    "converged": True, "iterations": i + 1,
                }

            # Extract 2×2 sub-block [dN/de, dN/dcy; dMy/de, dMy/dcy]
            J = np.array([[K[0, 0], K[0, 2]],
                          [K[2, 0], K[2, 2]]])
            det = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            if det < 1e-20:
                eps0 += 1e-5
                chi_y += 1e-7
                continue

            try:
                d = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                eps0 += 1e-5
                chi_y += 1e-7
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                e_new = eps0 + alpha * d[0]
                c_new = chi_y + alpha * d[1]
                Nn, _, Myn = self.integrate(e_new, 0.0, c_new)
                rn = np.array([Nn - N_target, Myn - My_target])
                if np.linalg.norm(rn) < r_norm:
                    break
                alpha *= 0.5
            eps0 += alpha * d[0]
            chi_y += alpha * d[1]

        N, Mx, My = self.integrate(eps0, 0.0, chi_y)
        return {
            "eps0": eps0, "chi_x": 0.0, "chi_y": chi_y,
            "N": N, "Mx": Mx, "My": My,
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Biaxial solver (analytical Jacobian)
    # ------------------------------------------------------------------

    def _solve_biaxial(self, N_target, Mx_target, My_target,
                       eps0_init, chi_x_init, chi_y_init,
                       tol, max_iter):
        """3-unknown solver with multi-start strategy."""
        if (eps0_init == 0.0 and chi_x_init == 1e-6
                and chi_y_init == 0.0):
            e0, cx0, cy0 = self._elastic_initial_guess(
                N_target, Mx_target, My_target)
        else:
            e0, cx0, cy0 = eps0_init, chi_x_init, chi_y_init

        sol = self._nr_biaxial(N_target, Mx_target, My_target,
                               e0, cx0, cy0, tol, max_iter)
        if sol["converged"]:
            return sol

        # Warm-start from uniaxial
        sol_uni = self._solve_uniaxial(
            N_target, Mx_target, 0.0, 1e-6, tol, max_iter // 2)
        if sol_uni["converged"]:
            sol2 = self._nr_biaxial(
                N_target, Mx_target, My_target,
                sol_uni["eps0"], sol_uni["chi_x"], 1e-7,
                tol, max_iter)
            if sol2["converged"]:
                return sol2

        # Grid of chi_y
        for cy_try in [1e-6, -1e-6, 5e-6, -5e-6, 1e-5, -1e-5]:
            sol3 = self._nr_biaxial(
                N_target, Mx_target, My_target,
                e0, cx0, cy_try, tol, max_iter // 2)
            if sol3["converged"]:
                return sol3

        return sol

    def _nr_biaxial(self, N_target, Mx_target, My_target,
                    eps0, chi_x, chi_y, tol, max_iter):
        r"""
        Core Newton-Raphson for biaxial case.

        Uses the **analytical** 3×3 tangent stiffness matrix.
        """
        x = np.array([eps0, chi_x, chi_y])
        target = np.array([N_target, Mx_target, My_target])
        tol_vec = np.array([tol, tol * 1000, tol * 1000])

        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(
                x[0], x[1], x[2])
            f = np.array([N, Mx, My])
            r = f - target

            if np.all(np.abs(r) < tol_vec):
                return {
                    "eps0": x[0], "chi_x": x[1], "chi_y": x[2],
                    "N": f[0], "Mx": f[1], "My": f[2],
                    "converged": True, "iterations": i + 1,
                }

            try:
                d = np.linalg.solve(K, -r)
            except np.linalg.LinAlgError:
                x += np.array([1e-5, 1e-7, 1e-7])
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                x_new = x + alpha * d
                f_new = np.array(self.integrate(
                    x_new[0], x_new[1], x_new[2]))
                r_new = f_new - target
                if np.linalg.norm(r_new) < r_norm:
                    break
                alpha *= 0.5
            x = x + alpha * d

        f = np.array(self.integrate(x[0], x[1], x[2]))
        return {
            "eps0": x[0], "chi_x": x[1], "chi_y": x[2],
            "N": f[0], "Mx": f[1], "My": f[2],
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Post-processing
    # ------------------------------------------------------------------

    def get_fiber_results(self, eps0, chi_x, chi_y=0.0):
        r"""
        Full strain/stress state at every fiber.

        For multi-material sections, each bulk fiber is evaluated
        with its own constitutive law.

        Parameters
        ----------
        eps0 : float
        chi_x : float
        chi_y : float, optional

        Returns
        -------
        dict
            ``'bulk'``: sub-dict with ``x``, ``y``, ``eps``,
            ``sigma``, ``dA``.
            ``'rebars'``: sub-dict with ``x``, ``y``, ``eps``,
            ``sigma`` (ideal_gross), ``sigma_net`` (net after bulk
            subtraction), ``A``, ``embedded``.
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        # Bulk stresses
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
        else:
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[idx] = mat.stress_array(
                    eb[idx] + self._bulk_offset(idx))

        # Rebar stresses (zone-aware grouping).
        sr_ideal_gross = np.zeros_like(er)
        sb_at_rebars = np.zeros_like(er)
        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            sr_ideal_gross[idx] = mat.stress_array(er_g)
            sb_at_rebars[idx] = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))

        # Net rebar stress (subtract zone-correct bulk at rebar location)
        sr_net = sr_ideal_gross.copy()
        emb = self.sec.embedded_rebars
        sr_net[emb] -= sb_at_rebars[emb]

        # ---- Tendon stresses (SLS): decoupled-argument evaluation ----
        # Each tendon's own law is evaluated at the offset total strain
        # (section strain + locked-in prestrain); the displaced bulk is
        # evaluated at the section strain.  Reported quantities:
        #   eps_sec  — section strain at the tendon
        #   eps_tot  — total strain seen by the tendon law
        #   sigma    — tendon stress sigma_p(eps_tot)
        #   sigma_net— net embedded stress (after bulk subtraction)
        result = {
            "bulk": {
                "x": self.sec.x_fibers.copy(),
                "y": self.sec.y_fibers.copy(),
                "eps": eb.copy(),
                "sigma": sb.copy(),
                "dA": self.sec.A_fibers.copy(),
            },
            "rebars": {
                "x": self.sec.x_rebars.copy(),
                "y": self.sec.y_rebars.copy(),
                "eps": er.copy(),
                "sigma": sr_ideal_gross.copy(),
                "sigma_net": sr_net.copy(),
                "A": self.sec.A_rebars.copy(),
                "embedded": self.sec.embedded_rebars.copy(),
            },
        }

        if self._has_tendons:
            et_sec = self._tendon_section_strain(eps0, chi_x, chi_y)
            et_tot = np.zeros_like(et_sec)
            sp = np.zeros_like(et_sec)
            sb_at_tendons = np.zeros_like(et_sec)
            for mat, bulk_mat, idx, eps_init in self._tendon_groups:
                e_sec_g = et_sec[idx]
                e_tot_g = e_sec_g + eps_init
                et_tot[idx] = e_tot_g
                sp[idx] = mat.stress_array(e_tot_g)
                sb_at_tendons[idx] = bulk_mat.stress_array(
                    e_sec_g + self._tendon_offset(idx))
            sp_net = sp.copy()
            emb_t = self.sec.embedded_tendons
            sp_net[emb_t] -= sb_at_tendons[emb_t]
            result["tendons"] = {
                "x": self.sec.x_tendons.copy(),
                "y": self.sec.y_tendons.copy(),
                "eps_sec": et_sec.copy(),
                "eps_init": self._eps_init_tendon.copy(),
                "eps_tot": et_tot.copy(),
                "sigma": sp.copy(),
                "sigma_net": sp_net.copy(),
                "A": self.sec.A_tendons.copy(),
                "embedded": self.sec.embedded_tendons.copy(),
            }

        return result

    def strains_within_limits(self, eps0, chi_x, chi_y=0.0):
        r"""
        Check whether all fiber strains are within material limits.

        A strain field is *admissible* when every fiber's strain
        lies within the range :math:`[\varepsilon_{\min},\,
        \varepsilon_{\max}]` defined by the fiber's constitutive
        law.

        This is the criterion that distinguishes interior points
        of the resistance domain from exterior points: the domain
        boundary corresponds to configurations where at least one
        fiber is *at* its limit.

        Parameters
        ----------
        eps0 : float
            Strain at the reference point.
        chi_x : float
            Curvature about the x-axis [1/mm].
        chi_y : float, optional
            Curvature about the y-axis [1/mm].

        Returns
        -------
        bool
            ``True`` if all fibers satisfy
            :math:`\varepsilon_{\min} \le \varepsilon_i
            \le \varepsilon_{\max}`.
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        for mat, idx in self._bulk_groups:
            e_group = eb[idx] + self._bulk_offset(idx)
            if np.any(e_group < mat.eps_min) or \
               np.any(e_group > mat.eps_max):
                return False

        for mat, _bulk_mat, idx in self._rebar_groups:
            e_group = er[idx]
            if np.any(e_group < mat.eps_min) or \
               np.any(e_group > mat.eps_max):
                return False

        # Tendons: limits apply to the TOTAL strain (section strain +
        # locked-in prestrain), since that is the argument of the
        # tendon's constitutive law.
        if self._has_tendons:
            et_sec = self._tendon_section_strain(eps0, chi_x, chi_y)
            for mat, _bulk_mat, idx, eps_init in self._tendon_groups:
                e_tot = et_sec[idx] + eps_init
                if np.any(e_tot < mat.eps_min) or \
                   np.any(e_tot > mat.eps_max):
                    return False

        return True
