"""
Numerical validation of ``properties.py``.

Covers:
1. Pure-geometry degeneration (no rebars, E_bulk = E_ref = 1).
2. Homogenized RC section — 4-bar symmetric rectangle.
3. Homogenized RC section — asymmetric (rebars only on top),
   centroid shift predicted analytically.
4. Elastic moduli W_x = I_x / c for rectangle.
5. Plastic modulus Z of a rectangle = B H^2 / 4.
6. Plastic modulus Z of a circle = 4 R^3 / 3.
7. Plastic modulus of a doubly-symmetric T-section (composition
   formula).
8. Plastic modulus of a rotated rectangle, Z_xi recovered from Z_x
   of the unrotated one.
9. Mono-material degeneration: rebars with E equal to bulk give
   exactly the same result as no rebars.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from shapely.geometry import Polygon, Point, box as sbox

from properties import (
    compute_section_properties,
    compute_inertia_ellipse,
    compute_kern_polygon,
    HomogenizedRebar,
)


def _assert_close(got, want, rel=1e-6, abs_=1e-6, label=""):
    if abs(got - want) <= max(abs_, rel * abs(want)):
        return
    raise AssertionError(
        f"{label}: got {got:.6e}, want {want:.6e} "
        f"(diff {got-want:.3e})")


# ==== 1. Pure geometry — rectangle, no rebars ==================

def test_rectangle_pure_geometry():
    B, H = 300.0, 600.0
    p = sbox(0, 0, B, H)
    s = compute_section_properties(p)
    _assert_close(s.area, B * H, label="rect A")
    _assert_close(s.xg, B / 2, label="rect xG")
    _assert_close(s.yg, H / 2, label="rect yG")
    _assert_close(s.Ix, B * H ** 3 / 12, label="rect Ix")
    _assert_close(s.Iy, H * B ** 3 / 12, label="rect Iy")
    _assert_close(s.n_bulk, 1.0, label="rect n_bulk")
    # Elastic moduli
    _assert_close(s.W_x_top, B * H ** 2 / 6, rel=1e-6, label="W_x_top")
    _assert_close(s.W_x_bot, B * H ** 2 / 6, rel=1e-6, label="W_x_bot")
    _assert_close(s.W_y_left, H * B ** 2 / 6, rel=1e-6, label="W_y_left")
    _assert_close(s.W_y_right, H * B ** 2 / 6, rel=1e-6, label="W_y_right")
    # Extreme fiber distances
    _assert_close(s.c_y_top, H / 2, label="c_y_top")
    _assert_close(s.c_y_bot, H / 2, label="c_y_bot")
    _assert_close(s.c_x_left, B / 2, label="c_x_left")
    _assert_close(s.c_x_right, B / 2, label="c_x_right")
    # Plastic modulus: B H^2 / 4
    _assert_close(s.Z_x, B * H ** 2 / 4, rel=1e-4, label="Z_x")
    _assert_close(s.Z_y, H * B ** 2 / 4, rel=1e-4, label="Z_y")
    print("rectangle (pure geometry) OK")


# ==== 2. Homogenized RC — symmetric 4-bar rectangle ============

def test_RC_rectangle_symmetric():
    B, H = 300.0, 500.0
    cov = 40.0
    E_c = 30000.0
    E_s = 200000.0
    As = np.pi * 20.0 ** 2 / 4.0
    n_s = E_s / E_c           # ≈ 6.667
    p = sbox(0.0, 0.0, B, H)
    rebars = [
        HomogenizedRebar(cov, cov, As, E_s),
        HomogenizedRebar(B - cov, cov, As, E_s),
        HomogenizedRebar(cov, H - cov, As, E_s),
        HomogenizedRebar(B - cov, H - cov, As, E_s),
    ]
    s = compute_section_properties(p, rebars=rebars,
                                    E_bulk=E_c, E_ref=E_c)
    # A_id = B H + 4 As (n_s - 1)
    A_id = B * H + 4.0 * As * (n_s - 1.0)
    _assert_close(s.area, A_id, label="RC A_id")
    _assert_close(s.xg, B / 2, label="RC xG (symm)")
    _assert_close(s.yg, H / 2, label="RC yG (symm)")
    # I_x_id = B H^3 / 12 + 4 As (n_s - 1) * (H/2 - cov)^2
    ybar = H / 2 - cov
    Ix_id = B * H ** 3 / 12.0 + 4.0 * As * (n_s - 1.0) * ybar ** 2
    _assert_close(s.Ix, Ix_id, label="RC Ix_id")
    # Extreme fiber is the polygon top/bottom, not the rebars.
    _assert_close(s.c_y_top, H / 2, label="RC c_y_top")
    _assert_close(s.c_y_bot, H / 2, label="RC c_y_bot")
    # W_x = Ix_id / (H/2)
    _assert_close(s.W_x_top, Ix_id / (H / 2), label="RC W_x_top")
    print(f"RC symmetric OK (A_id={A_id:.1f}, Ix_id={Ix_id:.3e})")


# ==== 3. Homogenized RC — asymmetric (rebars only on top) ======

def test_RC_rectangle_asymmetric():
    B, H = 300.0, 500.0
    cov = 40.0
    E_c = 30000.0
    E_s = 200000.0
    As = np.pi * 20.0 ** 2 / 4.0
    n_s = E_s / E_c
    p = sbox(0.0, 0.0, B, H)
    rebars = [
        HomogenizedRebar(cov, H - cov, As, E_s),
        HomogenizedRebar(B - cov, H - cov, As, E_s),
    ]
    s = compute_section_properties(p, rebars=rebars,
                                    E_bulk=E_c, E_ref=E_c)
    A_id = B * H + 2.0 * As * (n_s - 1.0)
    yg_id = (B * H * (H / 2)
             + 2.0 * As * (n_s - 1.0) * (H - cov)) / A_id
    _assert_close(s.area, A_id, label="asym RC A_id")
    _assert_close(s.yg, yg_id, rel=1e-9, label="asym RC yG")
    assert s.yg > H / 2, "Asymmetric rebar must pull yG upward"
    # Centroid must be above the geometric mid-height.
    print(f"RC asymmetric OK (yG={yg_id:.4f} > H/2={H/2})")


# ==== 4. Mono-material degeneration ============================

def test_mono_material_degenerates():
    """
    Rebars with E equal to E_bulk must produce the same output as
    the bare polygon: contribution factor (n_s - n_bulk) = 0.
    """
    B, H = 300.0, 500.0
    E = 200000.0
    p = sbox(0.0, 0.0, B, H)
    rebars = [
        HomogenizedRebar(50, 50, 500.0, E),
        HomogenizedRebar(250, 50, 500.0, E),
    ]
    s1 = compute_section_properties(p, E_bulk=E)
    s2 = compute_section_properties(p, rebars=rebars, E_bulk=E)
    _assert_close(s1.area, s2.area, rel=1e-12, label="mono A")
    _assert_close(s1.Ix, s2.Ix, rel=1e-12, label="mono Ix")
    _assert_close(s1.Z_x, s2.Z_x, rel=1e-9, label="mono Z_x")
    print("mono-material degeneration OK")


# ==== 5. Plastic modulus — rectangle ===========================

def test_plastic_rectangle():
    B, H = 300.0, 600.0
    p = sbox(-B/2, -H/2, B/2, H/2)
    s = compute_section_properties(p)
    _assert_close(s.Z_x, B * H ** 2 / 4, rel=1e-4, label="Z_x rect")
    _assert_close(s.Z_y, H * B ** 2 / 4, rel=1e-4, label="Z_y rect")
    # For rectangle: Z_xi = Z_x because α=0.
    _assert_close(s.Z_xi, B * H ** 2 / 4, rel=1e-4, label="Z_xi rect")
    print("rectangle plastic modulus OK")


# ==== 6. Plastic modulus — circle ==============================

def test_plastic_circle():
    R = 500.0
    p = Point(0, 0).buffer(R, resolution=64)
    s = compute_section_properties(p)
    # Z = 4 R^3 / 3 (analytical for full circle)
    Z_exact = 4.0 * R ** 3 / 3.0
    _assert_close(s.Z_x, Z_exact, rel=3e-3, label="Z circle")
    _assert_close(s.Z_y, Z_exact, rel=3e-3, label="Z circle y")
    print("circle plastic modulus OK")


# ==== 7. Plastic modulus — doubly-symmetric T-section ==========

def test_plastic_tsection_symmetric():
    """
    Symmetric I-like section: flange t_f × b_f on top and bottom,
    web t_w × h_w in the middle.  For this doubly-symmetric case
    Z_x = 2 × (Q_flange + Q_halfweb).
    """
    bf, tf, tw, hw = 400.0, 50.0, 200.0, 400.0
    total_H = hw + 2 * tf
    coords = [
        (-bf/2, -total_H/2),
        ( bf/2, -total_H/2),
        ( bf/2, -hw/2),
        ( tw/2, -hw/2),
        ( tw/2,  hw/2),
        ( bf/2,  hw/2),
        ( bf/2,  total_H/2),
        (-bf/2,  total_H/2),
        (-bf/2,  hw/2),
        (-tw/2,  hw/2),
        (-tw/2, -hw/2),
        (-bf/2, -hw/2),
    ]
    p = Polygon(coords)
    s = compute_section_properties(p)
    # Half section (above x axis): flange centroid at (hw/2 + tf/2),
    # web centroid at (hw/4).  Z_x = 2 * [A_f d_f + A_hw d_hw]
    A_f = bf * tf
    d_f = hw / 2 + tf / 2
    A_hw = tw * (hw / 2)
    d_hw = hw / 4
    Z_x_exact = 2.0 * (A_f * d_f + A_hw * d_hw)
    _assert_close(s.Z_x, Z_x_exact, rel=1e-4, label="Z_x I-section")
    print(f"I-section plastic modulus OK (Z_x={Z_x_exact:.3e})")


# ==== 8. Plastic modulus — rotated rectangle ===================

def test_plastic_rotated_rectangle():
    """
    Rotate a rectangle by 35° and verify Z_xi = Z_x of the
    unrotated rectangle.
    """
    B, H = 200.0, 500.0
    theta = np.deg2rad(35.0)
    c, si = np.cos(theta), np.sin(theta)
    pts = np.array([(-B/2, -H/2), (B/2, -H/2),
                    (B/2, H/2), (-B/2, H/2)])
    R = np.array([[c, -si], [si, c]])
    rot = pts @ R.T
    p = Polygon(rot)
    s = compute_section_properties(p)
    # Z_xi of the rotated rectangle must equal Z_x of the
    # unrotated one.
    _assert_close(np.rad2deg(s.alpha), 35.0, abs_=1e-6,
                  label="rot α")
    _assert_close(s.Z_xi, B * H ** 2 / 4, rel=1e-4,
                  label="Z_xi (= Z_x of unrotated)")
    _assert_close(s.Z_eta, H * B ** 2 / 4, rel=1e-4,
                  label="Z_eta (= Z_y of unrotated)")
    print("rotated-rectangle plastic modulus OK")


# ==== 9. RC plastic modulus with asymmetric rebars =============

def test_plastic_RC_asymmetric():
    """
    Regression: Z_x of a homogenized rectangle with extra rebars
    near the top must exceed Z_x of the bare rectangle, *and* the
    PNA must shift (detectable because Z_y is unaffected since
    rebars are centred on the y-axis).
    """
    B, H = 300.0, 500.0
    cov = 40.0
    As = 314.16
    E_c = 30000.0
    E_s = 200000.0
    p = sbox(0.0, 0.0, B, H)
    # Two rebars centred on x=B/2, at different heights.
    rebars = [
        HomogenizedRebar(B/2, H - cov, As, E_s),
        HomogenizedRebar(B/2, H - cov - 50, As, E_s),
    ]
    s = compute_section_properties(p, rebars=rebars,
                                    E_bulk=E_c, E_ref=E_c)
    Z_x_bare = B * H ** 2 / 4
    assert s.Z_x > Z_x_bare, \
        f"Top rebars must increase Z_x: got {s.Z_x:.1f}, " \
        f"bare {Z_x_bare:.1f}"
    # Rebars on y-axis (x = B/2 = centroid x) — Z_y unchanged.
    Z_y_bare = H * B ** 2 / 4
    _assert_close(s.Z_y, Z_y_bare, rel=2e-3,
                  label="Z_y unaffected by rebars on symmetry line")
    print(f"RC plastic modulus shift OK "
          f"(Z_x={s.Z_x:.2e} > bare {Z_x_bare:.2e})")


# ==== 10. Homogenization reference ≠ bulk ======================

def test_homogenization_alt_reference():
    """
    Take a steel + rebar composite, homogenize to E_ref = E_s / 2
    (arbitrary non-bulk reference).  The homogenized area must
    scale as n_bulk = 2.
    """
    E_bulk = 200000.0
    E_ref = 100000.0
    B, H = 100.0, 100.0
    p = sbox(0.0, 0.0, B, H)
    s = compute_section_properties(p, E_bulk=E_bulk, E_ref=E_ref)
    _assert_close(s.n_bulk, 2.0, label="alt-ref n_bulk")
    _assert_close(s.area, 2.0 * B * H, label="alt-ref A")
    _assert_close(s.Ix, 2.0 * B * H ** 3 / 12,
                  label="alt-ref Ix")
    print("alternative-reference homogenization OK")


if __name__ == "__main__":
    test_rectangle_pure_geometry()
    test_RC_rectangle_symmetric()
    test_RC_rectangle_asymmetric()
    test_mono_material_degenerates()
    test_plastic_rectangle()
    test_plastic_circle()
    test_plastic_tsection_symmetric()
    test_plastic_rotated_rectangle()
    test_plastic_RC_asymmetric()
    test_homogenization_alt_reference()
    print("\nAll property tests passed ✔")
