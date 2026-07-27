# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.  Licensed under the GNU Affero General
# Public License v3 or later.  See LICENSE.
# ---------------------------------------------------------------------------
r"""
Phase-5 / C5 — rheology subsystem and time-dependent prestress losses.

The regression net under ``run_phase5_c5_validation.py``.  Where the
validator *demonstrates* (printing every number so an engineer can check
it), this suite *pins*: each invariant that, if it silently broke, would
leave the library returning a plausible and wrong answer.
"""

import dataclasses
import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from gensec.geometry.fiber import Tendon
from gensec.geometry.geometry import GenericSection
from gensec.materials.base import (
    RheologicalModel, aging_coefficient, relaxation_function,
)
from gensec.materials.ec2_bridge import concrete_from_class, prestress_from_ec2
from gensec.materials.rheology import (
    EC2RheologicalModel, TabulatedRheologicalModel,
)

# The ACI provider is a FALSIFICATION FIXTURE, not a library module: GenSec
# has no ACI material, so a rheology for it would promise a design capability
# that does not exist.  It lives at the repository root, outside the package.
import os                                                        # noqa: E402
import sys                                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from aci209_falsification import ACIRheologicalModel              # noqa: E402
from gensec.solver.losses import (
    EC2_LUMP_CHI, LossModel, compute_interval_losses, ec2_5106_closed_form,
    expand_losses,
)
from gensec.solver.section_state import QUANT_EPS, StagedDomainManager

INF = 25550.0                       # 70 years [days]
AC, U = 600 * 1400.0, 2 * 2000.0    # h0 = 420 mm


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def ec2():
    return EC2RheologicalModel(fck=35, cement_class="N", RH=70).with_geometry(
        A_c=AC, u=U)


@pytest.fixture
def ec2_steel(ec2):
    return ec2.with_steel(1860.0, relaxation_class=2, rho_1000=2.5)


@pytest.fixture
def pt_beam():
    """A 600x1400 post-tensioned beam, one bonded tendon at y = 200."""
    b, h = 600.0, 1400.0
    conc = concrete_from_class("C35/45", ls="S")
    ps = prestress_from_ec2(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                            Ep=195000.0)
    ps.ec2.relaxation_class, ps.ec2.rho_1000 = 2, 2.5
    t = Tendon(y=200.0, x=b / 2, Ap=2000.0, material=ps,
               eps_pe=1200.0 / 195000.0, bonded=True, embedded=True,
               name="T1")
    sec = GenericSection(polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
                         bulk_material=conc, rebars=[], tendons=[t],
                         mesh_size=30.0)
    st = StagedDomainManager(sec, biaxial=False).initial_state()
    return sec, st


# ======================================================================
#  The cardinal test: chi is an OUTPUT, never an input
# ======================================================================

class TestAgeingCoefficient:

    def test_chi_emerges_at_0_80_from_ec2s_own_compliance(self, ec2):
        """EN 1992-1-1 (5.46) writes 0.8 twice.  If the container is
        genuinely normative-agnostic, that 0.8 is what EC2's compliance
        PRODUCES.  This single assertion is the architecture."""
        chi = aging_coefficient(ec2, INF, 28.0, n_steps=200)
        assert 0.78 <= chi <= 0.83, chi

    @pytest.mark.parametrize("fck", [25, 45, 60])
    def test_chi_moves_with_the_concrete(self, fck):
        """A hard-coded 0.8 would not move.  This one does."""
        m = EC2RheologicalModel(fck=fck, RH=70).with_geometry(A_c=AC, u=U)
        chi = aging_coefficient(m, INF, 28.0, n_steps=200)
        assert 0.74 <= chi <= 0.87
        assert abs(chi - EC2_LUMP_CHI) > 1e-4     # not the lump value

    def test_chi_is_converged_at_the_default_grid(self, ec2):
        ref = aging_coefficient(ec2, INF, 28.0, n_steps=800)
        got = aging_coefficient(ec2, INF, 28.0, n_steps=100)
        assert abs(got - ref) < 5e-4

    def test_relaxation_function_starts_at_the_modulus(self, ec2):
        """R(t0, t0) = E_c(t0) = 1/J(t0,t0): the Volterra kernel's own
        initial condition."""
        R0 = relaxation_function(ec2, 28.0, 28.0)
        assert R0 == pytest.approx(ec2.E_c(28.0), rel=1e-12)

    def test_relaxation_decays(self, ec2):
        assert (relaxation_function(ec2, INF, 28.0)
                < relaxation_function(ec2, 100.0, 28.0)
                < ec2.E_c(28.0))

    def test_chi_undefined_at_zero_elapsed_time(self, ec2):
        with pytest.raises(ValueError, match="undefined at zero"):
            aging_coefficient(ec2, 28.0, 28.0)


# ======================================================================
#  Providers
# ======================================================================

class TestEC2Provider:

    def test_the_abc_identity_E_c_equals_one_over_J(self, ec2):
        for t in (7.0, 28.0, 365.0, INF):
            assert ec2.E_c(t) == pytest.approx(1.0 / ec2.J(t, t), rel=1e-12)

    def test_phi_gen_equals_phi_ec2_at_28_days(self, ec2):
        """The generalized (from-J) creep coefficient and the Eurocode's
        own coincide at t0 = 28 d.  Away from it they differ by the
        modulus ratio -- by design, and the creep STRAIN is still EC2's."""
        assert ec2.phi(INF, 28.0) == pytest.approx(ec2.phi_ec2(INF, 28.0),
                                                   rel=1e-12)

    @pytest.mark.parametrize("t,t0", [(365.0, 7.0), (90.0, 14.0)])
    def test_creep_strain_is_ec2s_at_every_loading_age(self, ec2, t, t0):
        container = ec2.phi(t, t0) / ec2.E_c(t0)
        eurocode = ec2.phi_ec2(t, t0) / ec2.Ecm28
        assert container == pytest.approx(eurocode, rel=1e-12)

    def test_shrinkage_is_signed_negative(self, ec2):
        """A shortening.  The Phase-5 bulk kernel left this sign to the
        provider (7_1 §9); it is fixed here."""
        assert ec2.eps_imposed(INF, 7.0) < 0.0
        assert abs(ec2.eps_imposed(INF, 7.0)) == pytest.approx(3.05e-4,
                                                               rel=0.05)

    def test_relaxation_is_signed_negative_and_grows_with_mu(self, ec2_steel):
        r70 = ec2_steel.relaxation(INF, 0.70)
        r80 = ec2_steel.relaxation(INF, 0.80)
        assert r80 < r70 < 0.0

    def test_relaxation_is_capped_at_500000_hours(self, ec2_steel):
        """EN 1992-1-1 §3.3.2(8): the Eurocode does not extrapolate."""
        a = ec2_steel.relaxation(500000.0 / 24.0, 0.7)
        b = ec2_steel.relaxation(1e9, 0.7)
        assert a == pytest.approx(b, rel=1e-12)

    def test_E_c_IS_fben2s(self):
        """The provider must NOT re-derive EC2's instantaneous properties.
        It once did -- because fben2.ecm was broken (F2) -- and that fork
        left two implementations of the same Eurocode formula with nothing
        enforcing that they agree.  F1/F2 are closed; the fork is gone.
        This test is the enforcement: E_c(t) IS fben2.ecm, at every age."""
        from gensec.materials.ec2_properties import fben2
        m = EC2RheologicalModel(fck=35, cement_class="N").with_geometry(
            A_c=AC, u=U)
        for t in (3.0, 7.0, 28.0, 90.0, 365.0):
            p = fben2(fck=35.0, ls="S", loadtype="slow", TypeConc="N",
                      time=t)
            assert m.E_c(t) == pytest.approx(p.ecm, rel=1e-12)
            assert m.linearity_limit(t) == pytest.approx(0.45 * p.fck,
                                                         rel=1e-12)
        assert m.fcm == pytest.approx(43.0, rel=1e-12)

    def test_nothing_eurocode_is_stored_on_the_provider(self):
        """A value copied at construction is a value frozen at
        construction.  The provider must hold NO Eurocode number."""
        m = EC2RheologicalModel(fck=35, cement_class="N")
        stored = set(vars(m))
        assert not stored & {"fcm", "Ecm28", "s_cem", "_memo", "_p28"}

    def test_linearity_limit_is_045_fck(self, ec2):
        assert ec2.linearity_limit(28.0) == pytest.approx(0.45 * 35.0,
                                                          rel=1e-9)

    def test_unbound_geometry_raises(self):
        with pytest.raises(ValueError, match="unbound"):
            EC2RheologicalModel(fck=35).phi(INF, 28.0)

    def test_unbound_steel_raises(self, ec2):
        with pytest.raises(ValueError, match="f_pk"):
            ec2.relaxation(INF, 0.7)


class TestFalsificationByACI:
    """ACI 209R-92 exists in this library for one reason: to try to break
    the claim of normative agnosticism.  It must pass through the same
    four-function door and come out *different*."""

    @pytest.fixture
    def aci(self):
        return ACIRheologicalModel(fc_28=35, RH=70).with_geometry(A_c=AC, u=U)

    def test_it_is_a_RheologicalModel(self, aci):
        assert isinstance(aci, RheologicalModel)

    def test_its_geometry_measure_is_V_over_S_in_inches(self, aci, ec2):
        assert aci.v_over_s_in == pytest.approx((AC / U) / 25.4, rel=1e-12)
        assert ec2.h0 == pytest.approx(2 * AC / U, rel=1e-12)   # a factor 2 apart

    def test_its_modulus_is_4700_sqrt_fc(self, aci):
        assert aci.E_c(28.0) == pytest.approx(4700.0 * math.sqrt(aci.fc(28.0)),
                                              rel=1e-12)

    def test_chi_differs_from_EC2_through_the_same_container(self, aci, ec2):
        chi_a = aging_coefficient(aci, INF, 28.0, n_steps=200)
        chi_e = aging_coefficient(ec2, INF, 28.0, n_steps=200)
        assert abs(chi_a - chi_e) > 0.01
        assert 0.6 < chi_a < 1.0          # still a physical ageing coefficient

    def test_its_linearity_limit_is_its_own_040_not_EC2s_045(self, aci):
        assert aci.linearity_limit(28.0) == pytest.approx(
            0.40 * aci.fc(28.0), rel=1e-12)

    def test_it_refuses_to_extrapolate_below_40_percent_RH(self):
        with pytest.raises(ValueError, match=r"\[40, 100\]"):
            ACIRheologicalModel(fc_28=35, RH=30)


class TestTabulatedProvider:
    """A norm that is only data must enter through the same door."""

    @pytest.fixture
    def tab(self, ec2):
        tp = np.array([7.0, 28.0, 90.0, 365.0])
        tt = np.array([7.0, 28.0, 90.0, 365.0, 3650.0, INF])
        JT = np.array([[ec2.J(max(t, s), s) for s in tp] for t in tt])
        return TabulatedRheologicalModel(tp, tt, JT)

    def test_E_c_is_the_tables_diagonal(self, tab, ec2):
        assert tab.E_c(28.0) == pytest.approx(ec2.E_c(28.0), rel=5e-3)

    def test_it_refuses_to_extrapolate_by_default(self, tab):
        with pytest.raises(ValueError, match="outside the tabulated range"):
            tab.J(1e6, 28.0)

    def test_a_malformed_table_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            TabulatedRheologicalModel([28.0, 7.0], [7.0, 28.0],
                                      np.ones((2, 2)) * 1e-5)


# ======================================================================
#  The engine
# ======================================================================

class TestSectionalAAEM:

    def _run(self, pt_beam, chi="lump"):
        sec, st = pt_beam
        prov = EC2RheologicalModel(fck=35, cement_class="N", RH=70)
        return sec, st, prov, compute_interval_losses(
            sec, st, models={0: LossModel(provider=prov, chi=chi)},
            demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
            zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
            tendon_ages={"T1": INF - 28.0})

    @pytest.mark.parametrize("chi", ["lump", "from_J"])
    def test_it_reproduces_EN_1992_1_1_expression_5_46(self, pt_beam, chi):
        """Two independent derivations -- a 3x3 system on the age-adjusted
        transformed section, and the Eurocode's scalar closed form on the
        gross concrete -- must land on the same number."""
        sec, st, prov, r = self._run(pt_beam, chi)
        b, h, y_t = 600.0, 1400.0, 200.0
        pb = prov.with_geometry(A_c=b * h, u=2 * (b + h)).with_steel(
            1860.0, relaxation_class=2, rho_1000=2.5)
        cf = ec2_5106_closed_form(
            Ep=195000.0, Ec=pb.E_c(28.0), Ap=2000.0, Ac=b * h,
            Ic=b * h ** 3 / 12, z_cp=h / 2 - y_t,
            sigma_c0=r.sigma_c_tendon["T1"],
            eps_sh=pb.eps_imposed(INF, 7.0) - pb.eps_imposed(28.0, 7.0),
            d_sigma_pr=pb.relaxation(INF - 28.0, r.sigma_p0["T1"] / 1860.0),
            phi=r.phi[0], chi=r.chi[0])
        assert r.d_sigma_p["T1"] == pytest.approx(cf, rel=5e-3)

    def test_the_loss_is_a_loss(self, pt_beam):
        _s, _st, _p, r = self._run(pt_beam)
        assert r.d_sigma_p["T1"] < 0.0
        assert 0.03 < r.loss_fraction["T1"] < 0.20

    def test_the_elastic_shortening_EMERGES_from_equilibrium(self, pt_beam):
        """sigma_p0 is read off the section, not declared: the jacking
        stress minus whatever the concrete's shortening took."""
        _s, _st, _p, r = self._run(pt_beam)
        assert r.sigma_p0["T1"] < 1200.0
        assert r.sigma_p0["T1"] > 1150.0

    def test_the_concrete_is_compressed_at_the_tendon(self, pt_beam):
        _s, _st, _p, r = self._run(pt_beam)
        assert r.sigma_c_tendon["T1"] < 0.0

    def test_the_lump_chi_and_the_computed_chi_barely_differ(self, pt_beam):
        """EC2's 0.8 is an excellent approximation *because it is what
        EC2's own compliance produces*.  That agreement is the free
        consistency check of the whole architecture."""
        _s1, _t1, _p1, a = self._run(pt_beam, "lump")
        _s2, _t2, _p2, b = self._run(pt_beam, "from_J")
        assert a.d_sigma_p["T1"] == pytest.approx(b.d_sigma_p["T1"], rel=2e-3)


class TestEmissionTheorem:
    """eps_init(t) = eps_init(t0) + d_sigma_pr / E_p, and nothing else."""

    def test_the_tendons_prestrain_moves_by_exactly_its_relaxation(
            self, pt_beam):
        sec, st = pt_beam
        prov = EC2RheologicalModel(fck=35, RH=70)
        r = compute_interval_losses(
            sec, st, models={0: LossModel(provider=prov)},
            demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
            zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
            tendon_ages={"T1": INF - 28.0})
        ui = next(iter(r.eps_override))
        emitted = r.eps_override[ui] - float(st.eps_init[ui])
        assert emitted == pytest.approx(r.d_sigma_pr["T1"] / 195000.0,
                                        rel=1e-10)

    def test_creep_and_shrinkage_travel_through_the_plane_not_the_steel(
            self, pt_beam):
        sec, st = pt_beam
        prov = EC2RheologicalModel(fck=35, RH=70)
        r = compute_interval_losses(
            sec, st, models={0: LossModel(provider=prov)},
            demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
            zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
            tendon_ages={"T1": INF - 28.0})
        assert abs(r.d_sigma_p["T1"]) > 2.5 * abs(r.d_sigma_pr["T1"])


class TestNullAndSign:

    @pytest.fixture
    def plain(self):
        b, h = 600.0, 1400.0
        sec = GenericSection(
            polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
            bulk_material=concrete_from_class("C35/45", ls="S"),
            rebars=[], tendons=[], mesh_size=50.0)
        return sec, StagedDomainManager(sec, biaxial=False).initial_state()

    def _free(self, plain):
        sec, st = plain
        prov = EC2RheologicalModel(fck=35, RH=70)
        return compute_interval_losses(
            sec, st, models={0: LossModel(provider=prov)},
            demand=(0.0, 0.0, 0.0), zone_ages_t0={0: 28.0},
            zone_ages_t={0: INF}, zone_curing_ages={0: 7.0}, tendon_ages={})

    def test_unrestrained_shrinkage_produces_no_stress(self, plain):
        r = self._free(plain)
        assert abs(float(r.d_sigma_zone[0][0])) < 1e-8

    def test_unrestrained_shrinkage_produces_no_curvature(self, plain):
        r = self._free(plain)
        assert abs(r.u[1]) < 1e-14 and abs(r.u[2]) < 1e-14

    def test_the_bulk_offset_is_the_NEGATIVE_of_the_eigenstrain(self, plain):
        """The kernel ADDS the offset; the physics SUBTRACTS the
        eigenstrain.  Falsifiable consequence: a fully restrained
        shrinking member goes into TENSION -- which is why composite
        toppings crack."""
        r = self._free(plain)
        assert r.bulk_plane_delta[0][0] > 0.0        # offset positive
        assert r.eps_imp_zone[0][0] < 0.0            # eigenstrain negative
        assert r.bulk_plane_delta[0][0] == pytest.approx(
            -r.eps_imp_zone[0][0], rel=1e-12)


class TestBoltzmannSuperposition:
    """A naive cumulative sum (each step restarting the creep clock)
    over-counts by ~80%.  Only superposition over the stress history
    converges."""

    def test_the_step_by_step_converges_on_the_AAEM(self, pt_beam):
        sec, st = pt_beam
        prov = EC2RheologicalModel(fck=35, RH=70)
        common = dict(demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
                      zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
                      tendon_ages_t0={"T1": 0.0}, interval_days=INF - 28.0)
        _o, tr1, _h, _r = expand_losses(
            sec, st, models={0: LossModel(provider=prov)}, **common)
        base = tr1[0].d_sigma_p["T1"]
        cuts = sorted({round((INF - 28.0) * 10 ** (-3 + 3 * k / 8), 4)
                       for k in range(1, 8)})
        _o, tr, _h, _r = expand_losses(
            sec, st, models={0: LossModel(provider=prov, steps=cuts)},
            **common)
        tot = sum(r.d_sigma_p["T1"] for r in tr)
        # converged, and the gap MEASURES the AAEM's own approximation
        assert tot == pytest.approx(base, rel=0.03)
        assert abs(tot - base) / abs(base) > 0.002

    def test_the_relaxation_is_an_increment_not_a_cumulative(self, pt_beam):
        """Relaxation is a function of the TOTAL time under load.  Summed
        cumulatively across sub-steps it would be charged N times."""
        sec, st = pt_beam
        prov = EC2RheologicalModel(fck=35, RH=70)
        common = dict(demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
                      zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
                      tendon_ages_t0={"T1": 0.0}, interval_days=INF - 28.0)
        _o, tr1, _h, _r = expand_losses(
            sec, st, models={0: LossModel(provider=prov)}, **common)
        _o, tr, _h, _r = expand_losses(
            sec, st, models={0: LossModel(provider=prov,
                                          steps=[100.0, 1000.0, 10000.0])},
            **common)
        one = tr1[0].d_sigma_pr["T1"]
        many = sum(r.d_sigma_pr["T1"] for r in tr)
        assert many == pytest.approx(one, rel=1e-9)


class TestGuards:
    """Each of these, if it did not fire, would leave the library
    returning a plausible and wrong number."""

    def test_a_sub_quantum_step_raises(self, pt_beam):
        sec, st = pt_beam
        prov = EC2RheologicalModel(fck=35, RH=70)
        with pytest.raises(ValueError, match="QUANT_EPS"):
            expand_losses(
                sec, st,
                models={0: LossModel(provider=prov,
                                     steps=[INF - 30.0, INF - 29.5])},
                demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
                zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
                tendon_ages_t0={"T1": 0.0}, interval_days=INF - 28.0)

    def test_non_linear_creep_raises(self, pt_beam):
        sec, st = pt_beam
        prov = EC2RheologicalModel(fck=35, RH=70)
        with pytest.raises(ValueError, match="linear-viscoelasticity"):
            compute_interval_losses(
                sec, st, models={0: LossModel(provider=prov)},
                demand=(-9.0e6, -800e6, 0.0), zone_ages_t0={0: 3.0},
                zone_ages_t={0: INF}, zone_curing_ages={0: 3.0},
                tendon_ages={"T1": INF})

    def test_an_unbonded_tendon_raises(self, pt_beam):
        """A section library does not have the member-average strain an
        unbonded tendon's stress change depends on.  Substituting the
        local value would be a silent mismodel."""
        sec, st = pt_beam
        bonded = np.asarray(st.bonded).copy()
        bonded[-1] = False
        st_ub = dataclasses.replace(st, bonded=bonded)
        prov = EC2RheologicalModel(fck=35, RH=70)
        with pytest.raises(NotImplementedError, match="member-average"):
            compute_interval_losses(
                sec, st_ub, models={0: LossModel(provider=prov)},
                demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
                zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
                tendon_ages={"T1": INF})

    def test_a_malformed_LossModel_raises(self):
        prov = EC2RheologicalModel(fck=35).with_geometry(A_c=AC, u=U)
        with pytest.raises(ValueError, match="chi must be"):
            LossModel(provider=prov, chi="rigorous")
        with pytest.raises(ValueError, match="relaxation_reduction"):
            LossModel(provider=prov, relaxation_reduction=1.5)
        with pytest.raises(ValueError, match="strictly increasing"):
            LossModel(provider=prov, steps=[100.0, 50.0])


# ======================================================================
#  D6 -- the bulk_plane_delta primitive
# ======================================================================

class TestBulkPlaneDelta:

    def test_it_is_additive(self, pt_beam):
        """N sub-steps of an interval must sum to the interval.  An
        absolute setter would let the last step overwrite the history."""
        _sec, st = pt_beam
        a = st.with_bulk_plane_delta({0: (1e-4, 2e-8, 0.0)})
        b = a.with_bulk_plane_delta({0: (3e-4, 1e-8, 0.0)})
        assert b.bulk_planes[0][0] == pytest.approx(4e-4, rel=1e-12)
        assert b.bulk_planes[0][1] == pytest.approx(3e-8, rel=1e-12)

    def test_it_refuses_an_inactive_zone(self, pt_beam):
        """Concrete that has not been cast does not creep."""
        _sec, st = pt_beam
        bad = np.asarray(st.bulk_active).copy()
        bad[0] = False
        st2 = dataclasses.replace(st, bulk_active=bad)
        with pytest.raises(ValueError, match="not active"):
            st2.with_bulk_plane_delta({0: (1e-4, 0.0, 0.0)})

    def test_it_refuses_a_malformed_increment(self, pt_beam):
        _sec, st = pt_beam
        with pytest.raises(ValueError, match="three finite floats"):
            st.with_bulk_plane_delta({0: (1e-4, 0.0)})

    def test_it_enters_the_capacity_hash(self, pt_beam):
        """Two different loss states must never collapse onto one cached
        domain.  bulk_planes is already hashed, so the primitive needs no
        hash change -- assert it, do not assume it."""
        sec, st = pt_beam
        mgr = StagedDomainManager(sec, biaxial=False)
        base = [{"name": "s0", "components": []}]
        loss = [{"name": "s0", "components": [],
                 "section_ops": {"bulk_plane_delta": {0: [3e-4, 0.0, 0.0]}}}]
        _s1, h1, _b, _d = mgr.resolve_stages(base, initially_inactive=[])
        _s2, h2, _b, _d = mgr.resolve_stages(loss, initially_inactive=[])
        assert h1[-1] != h2[-1]


# ======================================================================
#  C5 -- the timeline
# ======================================================================

class TestTimelineC5:

    @pytest.fixture
    def flagship(self):
        from gensec.io_yaml import load_yaml
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in ("examples/example_composite_losses.yaml",
                     "../examples/example_composite_losses.yaml"):
            p = os.path.normpath(os.path.join(here, cand))
            if os.path.exists(p):
                return load_yaml(p)
        pytest.skip("example_composite_losses.yaml not found")

    def test_the_yaml_declares_the_rheology(self, flagship):
        lm = flagship["losses_models"]
        assert set(lm) == {"rheo_precast", "rheo_topping"}
        assert isinstance(lm["rheo_precast"].provider, EC2RheologicalModel)

    def test_the_emission_is_frozen_once(self, flagship):
        """Losses are timeline physics.  Re-deriving them per combination
        would let a combination's *variable* actions leak into a
        *permanent* creep history."""
        from gensec.solver.timeline import ConstructionTimeline
        sec = flagship["section"]
        ddb = {d["name"]: d for d in flagship["demands"]}
        tl = ConstructionTimeline.from_block(
            flagship["construction_history"],
            losses_models=flagship["losses_models"])
        res = tl.resolve(sec, ddb)
        assert len(res.losses_ops) == 3
        for ops in res.losses_ops.values():
            assert "eps_override" in ops and "bulk_plane_delta" in ops

    def test_the_compiler_REPLAYS_and_never_re_derives(self, flagship):
        from gensec.solver.timeline import ConstructionTimeline
        sec = flagship["section"]
        ddb = {d["name"]: d for d in flagship["demands"]}
        tl = ConstructionTimeline.from_block(
            flagship["construction_history"],
            losses_models=flagship["losses_models"])
        res = tl.resolve(sec, ddb)
        combo = [c for c in flagship["combinations"]
                 if c["name"] == "ULS_composite"][0]
        seen = []
        for _pt, stages, _in in tl.compile_combination(combo, res, sec, ddb):
            for stg in stages:
                ops = stg.get("section_ops") or {}
                if "bulk_plane_delta" in ops:
                    seen.append(ops["bulk_plane_delta"])
        # every replayed op is IDENTICAL to the frozen one
        frozen = [o["bulk_plane_delta"] for o in res.losses_ops.values()]
        for s in seen:
            assert any(s == f for f in frozen)

    def test_the_young_topping_creeps_faster_than_the_mature_precast(
            self, flagship):
        from gensec.solver.timeline import ConstructionTimeline
        sec = flagship["section"]
        ddb = {d["name"]: d for d in flagship["demands"]}
        tl = ConstructionTimeline.from_block(
            flagship["construction_history"],
            losses_models=flagship["losses_models"])
        res = tl.resolve(sec, ddb)
        last = res.losses_trace[max(res.losses_trace)][0]
        assert last.phi[1] > last.phi[0]

    def test_a_zone_with_no_declared_age_raises(self, flagship):
        from gensec.solver.timeline import ConstructionTimeline
        block = [{"interval": {"days": 100,
                               "losses": {"base": {"model": "rheo_precast",
                                                   "curing": 3}}}}]
        tl = ConstructionTimeline.from_block(
            block, losses_models=flagship["losses_models"])
        sec = flagship["section"]
        with pytest.raises(ValueError, match="no 'age'"):
            tl.resolve(sec, {})

    def test_an_unknown_model_reference_raises_at_PARSE_time(self, flagship):
        from gensec.solver.timeline import ConstructionTimeline
        with pytest.raises(ValueError, match="is not defined"):
            ConstructionTimeline.from_block(
                [{"interval": {"days": 100,
                               "losses": {"base": {"model": "nope",
                                                   "age": 28, "curing": 3}}}}],
                losses_models=flagship["losses_models"])

    def test_a_bare_model_name_is_refused(self, flagship):
        """The rheology of a zone is meaningless without its age and its
        curing age; guessing them would be a silent normative choice."""
        from gensec.solver.timeline import ConstructionTimeline
        with pytest.raises(ValueError, match="non-empty mapping"):
            ConstructionTimeline.from_block(
                [{"interval": {"days": 100, "losses": "rheo_precast"}}],
                losses_models=flagship["losses_models"])

    def test_an_interval_with_losses_but_no_days_raises(self, flagship):
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block(
            [{"interval": {"losses": {"base": {"model": "rheo_precast",
                                               "age": 28, "curing": 3}}}}],
            losses_models=flagship["losses_models"])
        with pytest.raises(ValueError, match="must declare its length"):
            tl.resolve(flagship["section"], {})


# ======================================================================
#  Regression: no losses_models -> the timeline is byte-identical
# ======================================================================

def test_a_model_without_losses_is_unchanged():
    """The whole rheology machinery must be inert unless asked for."""
    from gensec.solver.timeline import ConstructionTimeline
    tl = ConstructionTimeline.from_block([{"interval": {"days": 100}}])
    assert tl.losses_models == {}


# ======================================================================
#  F-B / F-C(1) -- the [stress, grout] window and the orphan drop
# ======================================================================

class TestStressedNotGrouted:
    r"""
    12_0b F-B.  A post-tensioned tendon that is stressed but not yet
    grouted is ``active=False``: invisible to the D8 unbonded guard
    (``losses.py`` ~824, which tests ``active and not bonded``) and
    invisible to the relaxation walk (``_interval_losses`` ~846, ``if not
    state.active[...]: continue``).  Its relaxation over the window is
    charged nowhere, and it is not recovered later either -- the first
    post-grout interval initialises ``relax_from`` at
    ``T = t_now - t_stress``.
    """

    @pytest.fixture
    def gap(self):
        """The flagship with `grout` moved after the first interval."""
        from gensec.io_yaml import load_yaml
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in ("examples/example_composite_losses_fb_gap.yaml",
                     "../examples/example_composite_losses_fb_gap.yaml"):
            p = os.path.normpath(os.path.join(here, cand))
            if os.path.exists(p):
                return load_yaml(p)
        pytest.skip("example_composite_losses_fb_gap.yaml not found")

    # -- the fixture can exhibit the bug (rule 3) ----------------------

    def test_the_gap_fixture_differs_from_the_flagship_by_ONE_event(self):
        """Rule 3 has a converse: a fixture that changes more than one
        thing cannot attribute the difference either.  These two files
        must be the same timeline with `grout` in a different place."""
        from gensec.io_yaml import load_yaml
        import os
        here = os.path.dirname(os.path.abspath(__file__))

        def _hist(fn):
            for cand in (f"examples/{fn}", f"../examples/{fn}"):
                p = os.path.normpath(os.path.join(here, cand))
                if os.path.exists(p):
                    return load_yaml(p)["construction_history"]
            pytest.skip(f"{fn} not found")

        a = _hist("example_composite_losses.yaml")
        b = _hist("example_composite_losses_fb_gap.yaml")
        assert len(a) == len(b)
        # same multiset of events, different order
        key = lambda e: repr(sorted(e.items()))          # noqa: E731
        assert sorted(map(key, a)) == sorted(map(key, b))
        assert [key(e) for e in a] != [key(e) for e in b]
        # and the ONE displaced event is the grout
        moved = [i for i, (x, y) in enumerate(zip(a, b)) if key(x) != key(y)]
        assert moved, "the two histories are identical -- no gap"
        assert any("grout" in e for e in (a[moved[0]], b[moved[0]]))

    # -- the guard --------------------------------------------------------

    def test_the_flagship_gap_fixture_dies_fail_loud(self, gap):
        """The acceptance case of the whole C5 arc, with its stress and
        grout no longer adjacent.  Before the fix it ran and returned a
        wrong number; now it must refuse."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block(
            gap["construction_history"],
            losses_models=gap["losses_models"])
        ddb = {d["name"]: d for d in gap["demands"]}
        # the window named must be the grout: P1 is post-tensioned
        with pytest.raises(NotImplementedError,
                           match=r"not yet live.*'grout' has not"):
            tl.resolve(gap["section"], ddb)

    def test_it_raises_at_the_FIRST_interval_not_a_later_one(self, gap):
        """The event index in the message must name the interval that is
        crossed, not the one that happens to fail downstream."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block(
            gap["construction_history"],
            losses_models=gap["losses_models"])
        ddb = {d["name"]: d for d in gap["demands"]}
        first = min(ev.index for ev in tl.events if ev.kind == "interval")
        with pytest.raises(NotImplementedError,
                           match=rf"construction_history\[{first}\]"):
            tl.resolve(gap["section"], ddb)

    def test_a_PRE_tensioned_bed_window_raises_too(self):
        """The symmetric window.  A pre-tensioned strand is inactive
        until the concrete is cast around it, so the 30 days it spends
        tensioned on the bed are skipped by exactly the line that skips
        an ungrouted duct.  Measured before this guard existed:
        `interval[1] eps_override = None`, `interval[3] eps_override =
        {0: 0.004609...}` -- the bed window charged to no one."""
        from shapely.geometry import Polygon
        from gensec.geometry.geometry import GenericSection
        from gensec.geometry.fiber import Tendon
        from gensec.materials.ec2_bridge import (
            concrete_from_class, prestress_from_ec2)
        from gensec.materials.rheology import EC2RheologicalModel
        from gensec.solver.losses import LossModel
        from gensec.solver.timeline import ConstructionTimeline
        b, h = 600.0, 1000.0
        conc = concrete_from_class("C35/45", ls="S")
        ps = prestress_from_ec2(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                                Ep=195000.0)
        t = Tendon(y=850.0, x=b / 2, Ap=600.0, material=ps,
                   eps_pe=900.0 / 195000.0, name="P1")
        sec = GenericSection(
            polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
            bulk_material=conc, rebars=[], tendons=[t], mesh_size=50.0,
            bulk_materials=[(Polygon([(0, 800), (b, 800), (b, h), (0, h)]),
                             conc, "deck")])
        lm = {"rheo": LossModel(provider=EC2RheologicalModel(
            fck=35.0, cement_class="N", RH=70.0))}
        base = {"model": "rheo", "age": 28, "curing": 3}
        tl = ConstructionTimeline.from_block([
            {"stress": {"tendons": ["P1"], "sigma_p0": 900.0}},
            {"interval": {"days": 30, "losses": {"base": base}}},
            {"cast": {"zone": "deck", "datum": "auto"}},
            {"point": "svc"}], losses_models=lm)
        with pytest.raises(NotImplementedError, match="not yet live"):
            tl.validate(sec)

    def test_a_pre_tensioned_tendon_AFTER_its_cast_is_fine(self):
        """The predicate must let the ordinary pre-tensioning sequence
        through: once the concrete is cast the strand is live, and no
        grout event will ever exist for it."""
        from shapely.geometry import Polygon
        from gensec.geometry.geometry import GenericSection
        from gensec.geometry.fiber import Tendon
        from gensec.materials.ec2_bridge import (
            concrete_from_class, prestress_from_ec2)
        from gensec.solver.timeline import ConstructionTimeline
        b, h = 600.0, 1000.0
        conc = concrete_from_class("C35/45", ls="S")
        ps = prestress_from_ec2(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                                Ep=195000.0)
        t = Tendon(y=850.0, x=b / 2, Ap=600.0, material=ps,
                   eps_pe=900.0 / 195000.0, name="P1")
        sec = GenericSection(
            polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
            bulk_material=conc, rebars=[], tendons=[t], mesh_size=50.0,
            bulk_materials=[(Polygon([(0, 800), (b, 800), (b, h), (0, h)]),
                             conc, "deck")])
        tl = ConstructionTimeline.from_block([
            {"stress": {"tendons": ["P1"], "sigma_p0": 900.0}},
            {"cast": {"zone": "deck", "datum": "auto"}},
            {"interval": {"days": 30}},
            {"point": "svc"}])
        tl.validate(sec)                              # must not raise

    def test_it_is_refused_before_any_solver_runs(self, gap):
        """F-B is statically decidable, so it must not need `resolve`.
        This is what lets the CLI pre-flight a history in milliseconds
        instead of after a 3D resistance surface."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block(
            gap["construction_history"],
            losses_models=gap["losses_models"])
        with pytest.raises(NotImplementedError, match="not yet live"):
            tl.validate(gap["section"])

    def test_a_BARE_interval_raises_too(self, flagship):
        """The gap is opened by the *clock*, not by the losses block: a
        bare `interval` advances t_now just the same, and the first
        post-grout interval then starts its relaxation increment at
        T = t_now - t_stress.  A guard that only fired on intervals
        carrying `losses` would leave the silence fully reachable."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block([
            {"stress": {"tendons": ["P1"], "sigma_p0": 1300.0}},
            {"interval": {"days": 62}},
            {"grout": {"tendons": ["P1"]}},
            {"point": "after"}])
        # a bare interval: no losses block, same refusal
        with pytest.raises(NotImplementedError, match="not yet live"):
            tl.resolve(flagship["section"], {})

    def test_adjacent_stress_and_grout_still_pass(self, flagship):
        """The guard must not fire on the ordinary sequence -- the whole
        flagship is that sequence."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block(
            flagship["construction_history"],
            losses_models=flagship["losses_models"])
        ddb = {d["name"]: d for d in flagship["demands"]}
        res = tl.resolve(flagship["section"], ddb)
        assert len(res.losses_ops) == 3

    def test_an_interval_BEFORE_any_stress_still_passes(self, flagship):
        """`stressed` is populated by the walk, not up front: a tendon
        stressed *after* an interval is not pending *at* it."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block([
            {"interval": {"days": 30}},
            {"stress": {"tendons": ["P1"], "sigma_p0": 1300.0}},
            {"grout": {"tendons": ["P1"]}},
            {"point": "after"}])
        res = tl.resolve(flagship["section"], {})
        assert res.pre_post == {"P1": "post"}

    # -- what the silence was worth ---------------------------------------

    def test_the_dropped_relaxation_is_not_negligible(self, flagship):
        """Rule 2: the reference does not share the implementation's
        inputs -- the increment is read straight off the provider, not
        off the timeline.  Over the flagship's own 62-day first interval
        the skipped increment is ~1 % of sigma_p0, and it is the steep
        part of the curve: rho(1 d) alone is ~40 % of it."""
        from gensec.solver.losses import _tendon_steel
        sec = flagship["section"]
        st = _tendon_steel(sec.tendons[0])
        prov = flagship["losses_models"]["rheo_precast"].provider
        prov = prov.with_geometry(A_c=600 * 1400.0, u=2 * (600 + 1400.0))
        prov = prov.with_steel(st["f_pk"],
                               relaxation_class=st["relaxation_class"],
                               rho_1000=st["rho_1000"])
        mu0 = 1300.0 / st["f_pk"]
        dropped = prov.relaxation(62.0, mu0) - prov.relaxation(0.0, mu0)
        assert dropped < 0.0                      # a relaxation is a loss
        assert abs(dropped) / 1300.0 > 5e-3       # ~1.04 %, not round-off
        knee = prov.relaxation(1.0, mu0) - prov.relaxation(0.0, mu0)
        assert abs(knee) > 0.30 * abs(dropped)    # the window is the knee


class TestGroutOfAPretensionedTendon:
    r"""
    12_0b F-C(1).  ``_stress_actions`` emits a demand-side action only for
    ``pre_post == "post"``; ``_grout_stage`` drops one for **any** tendon
    with a recorded ``stress_sigma``, ``pre_post`` unchecked.  A history
    that stresses a tendon before its parent zone is cast and then grouts
    it therefore subtracts an action that was never added.
    """

    @pytest.fixture
    def pretensioned(self):
        """A tendon lying inside a CASTABLE zone.  Whether it is pre- or
        post-tensioned is therefore not a property of this section at
        all: it depends on where the `stress` event sits relative to the
        `cast`.  The explicit `parent=` override is deliberately absent
        -- for an embedded tendon the section rejects it, the parent
        being fixed by containment."""
        from gensec.geometry.geometry import GenericSection
        from gensec.geometry.fiber import Tendon
        from gensec.materials.ec2_bridge import (
            concrete_from_class, prestress_from_ec2)
        from shapely.geometry import Polygon
        b, h = 400.0, 800.0
        conc = concrete_from_class("C35/45", ls="S")
        ps = prestress_from_ec2(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                                Ep=195000.0)
        t = Tendon(y=700.0, x=b / 2, Ap=1000.0, material=ps,
                   eps_pe=1200.0 / 195000.0, bonded=True, embedded=True,
                   name="P1")
        return GenericSection(
            polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
            bulk_material=conc, rebars=[], tendons=[t], mesh_size=50.0,
            bulk_materials=[(Polygon([(0, 600), (b, 600), (b, h), (0, h)]),
                             conc, "topping")])

    def test_the_classification_FLIPS_with_the_event_order(
            self, pretensioned):
        """The same section and the same tendon, twice.  Only the order
        of `stress` and `cast` differs -- and that is the entire
        definition of pre- versus post-tensioning.  `Tendon.system`, if
        it were consulted, could contradict this; it is not."""
        from gensec.solver.timeline import (
            ConstructionTimeline, _classify_pre_post)
        before = ConstructionTimeline.from_block([
            {"stress": {"tendons": ["P1"], "sigma_p0": 1300.0}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"point": "p"}])
        after = ConstructionTimeline.from_block([
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"stress": {"tendons": ["P1"], "sigma_p0": 1300.0}},
            {"point": "p"}])
        assert _classify_pre_post(before.events, pretensioned) == {"P1": "pre"}
        assert _classify_pre_post(after.events, pretensioned) == {"P1": "post"}

    def test_resolve_and_validate_classify_IDENTICALLY(self, flagship):
        """Rule 5 made testable: the map `resolve` returns is the map the
        classifier `validate` uses.  Two homes could not be asserted
        equal -- they would only happen to agree."""
        from gensec.solver.timeline import (
            ConstructionTimeline, _classify_pre_post)
        tl = ConstructionTimeline.from_block(
            flagship["construction_history"],
            losses_models=flagship["losses_models"])
        sec = flagship["section"]
        ddb = {d["name"]: d for d in flagship["demands"]}
        assert tl.resolve(sec, ddb).pre_post == _classify_pre_post(
            tl.events, sec)

    def test_grouting_it_is_refused(self, pretensioned):
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block([
            {"stress": {"tendons": ["P1"], "sigma_p0": 1300.0}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"grout": {"tendons": ["P1"]}},
            {"point": "p"}])
        with pytest.raises(ValueError, match="PRE-tensioned"):
            tl.validate(pretensioned)

    def test_it_is_refused_at_VALIDATE_time_before_any_solve(
            self, pretensioned):
        """The rejection must not need the resolution walk: `validate`
        has no solver, and an absurd history should die before anything
        expensive runs."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block([
            {"stress": {"tendons": ["P1"], "sigma_p0": 1300.0}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"grout": {"tendons": ["P1"]}},
            {"point": "p"}])
        with pytest.raises(ValueError, match="orphan NEGATIVE"):
            tl.resolve(pretensioned, {})

    def test_the_unknown_tendon_message_still_wins(self, pretensioned):
        """Ordering: reference errors are reported in their own words,
        not swallowed by the classifier raising on a bad name."""
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block([
            {"grout": {"tendons": ["nope"]}}, {"point": "p"}])
        with pytest.raises(ValueError, match="unknown tendon"):
            tl.validate(pretensioned)

    def test_post_tensioned_grouting_is_untouched(self, flagship):
        from gensec.solver.timeline import ConstructionTimeline
        tl = ConstructionTimeline.from_block(
            flagship["construction_history"],
            losses_models=flagship["losses_models"])
        tl.validate(flagship["section"])          # must not raise


@pytest.fixture
def flagship():
    """Module-level twin of `TestTimelineC5.flagship` (class fixtures do
    not cross class boundaries)."""
    from gensec.io_yaml import load_yaml
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ("examples/example_composite_losses.yaml",
                 "../examples/example_composite_losses.yaml"):
        p = os.path.normpath(os.path.join(here, cand))
        if os.path.exists(p):
            return load_yaml(p)
    pytest.skip("example_composite_losses.yaml not found")
