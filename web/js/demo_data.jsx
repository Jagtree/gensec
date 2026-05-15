/* ==========================================================
 * GenSec GUI — demo data loader
 *
 * Activated by appending ?demo=1 to the URL.  Populates GS_DATA
 * with synthetic but realistic resistance domain / curves so the
 * Plotly plots can be exercised without the Python backend.
 * ========================================================== */
(function () {
  "use strict";
  const params = new URLSearchParams(location.search);
  if (params.get("demo") !== "1") return;

  // Build a stack of Mx-My contours that grow then shrink with N (typical
  // RC interaction surface).  Closed polygons; first point repeated at end.
  const Nlevels = [-4500, -3800, -3000, -2200, -1500, -800, -200, 200, 600, 1000, 1200];
  function ringAt(N_kN) {
    const Nb = -2200; // balance point
    const span = 4500;
    const factor = Math.max(0, 1 - Math.pow((N_kN - Nb) / span, 2) * 1.4);
    const Mx0 = 420 * factor;
    const My0 = 280 * factor;
    const pts = [];
    const n = 96;
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI * 2;
      // super-ellipse (~rectangle with rounded corners) for visual variety
      const c = Math.cos(t), s = Math.sin(t);
      const r = 1 / Math.pow(Math.pow(Math.abs(c), 2.3) + Math.pow(Math.abs(s), 2.3), 1 / 2.3);
      pts.push([Mx0 * r * c, My0 * r * s]);
    }
    pts.push(pts[0]);
    return pts;
  }

  const surface_slices = Nlevels.map(N => ({ N_kN: N, points: ringAt(N) }));
  const mx_my_slices = surface_slices.map(s => ({ N_kN: s.N_kN, points: s.points }));

  // N-M (uniaxial about X)
  const nm_points = [];
  for (let N = -4500; N <= 1200; N += 60) {
    const ring = ringAt(N);
    let Mmax = 0;
    for (const [Mx] of ring) Mmax = Math.max(Mmax, Mx);
    nm_points.push([N, Mmax]);
  }

  // M-χ per N level (5 curves)
  const mchi = [-3000, -1500, 0, 600, 1000].map((N) => {
    const ring = ringAt(N);
    let My = 0;
    for (const [Mx] of ring) My = Math.max(My, Mx);
    const Mu = My * 0.95;
    const chi_y = 25e-6 + 8e-6 * (1 + N / 4000);
    const pts = [];
    pts.push([0, 0]);
    pts.push([chi_y, Mu * 0.85]);
    pts.push([chi_y * 2.4, Mu]);
    pts.push([chi_y * 5, Mu * 0.98]);
    pts.push([chi_y * 6, Mu * 0.7]);
    return { N_kN: N, points: pts };
  });

  // Synthetic AnalysisResult
  const ar = {
    section: {
      B_mm: 400, H_mm: 600, bulk_material: "C30/37",
      n_fibers_x: 16, n_fibers_y: 24,
      rebars: [
        { x_mm: 50,  y_mm: 50,  diameter_mm: 20, material: "B450C", area_mm2: 314 },
        { x_mm: 200, y_mm: 50,  diameter_mm: 20, material: "B450C", area_mm2: 314 },
        { x_mm: 350, y_mm: 50,  diameter_mm: 20, material: "B450C", area_mm2: 314 },
        { x_mm: 50,  y_mm: 550, diameter_mm: 20, material: "B450C", area_mm2: 314 },
        { x_mm: 200, y_mm: 550, diameter_mm: 20, material: "B450C", area_mm2: 314 },
        { x_mm: 350, y_mm: 550, diameter_mm: 20, material: "B450C", area_mm2: 314 },
        { x_mm: 50,  y_mm: 300, diameter_mm: 16, material: "B450C", area_mm2: 201 },
        { x_mm: 350, y_mm: 300, diameter_mm: 16, material: "B450C", area_mm2: 201 },
      ],
    },
    materials: [
      { id: "C30/37", kind: "concrete",
        properties: { class: "C30/37", fcd_MPa: 17.0, fctm_MPa: 2.9, Ecm_MPa: 32837, eps_su: 0.0035 } },
      { id: "B450C", kind: "steel",
        properties: { class: "B450C", fyd_MPa: 391.3, Es_MPa: 200000, eps_su: 0.075 } },
    ],
    demands: [
      { name: "D1-perm",  N_kN: -1800, Mx_kNm: 220, My_kNm: 90 },
      { name: "D2-quake", N_kN: -1200, Mx_kNm: 360, My_kNm: 180 },
      { name: "D3-wind",  N_kN: -2400, Mx_kNm: 180, My_kNm: -120 },
    ],
    combinations: [
      { name: "ULS-1", components: ["D1-perm","D2-quake"], stages: [],
        resolved: { N_kN: -2100, Mx_kNm: 310, My_kNm: 140 } },
    ],
    envelopes: [
      { name: "ENV-A", members: ["D1-perm","D2-quake","D3-wind"], eta_max: 0.84 },
    ],
    verification: {
      rows: [
        { kind: "demand", name: "D1-perm",  N_kN: -1800, Mx_kNm: 220, My_kNm:  90, eta_3D: 0.62, eta_2D: 0.58, eta_path: 0.61, eta_path_2D: 0.57, status: "ok" },
        { kind: "demand", name: "D2-quake", N_kN: -1200, Mx_kNm: 360, My_kNm: 180, eta_3D: 0.91, eta_2D: 0.86, eta_path: 0.93, eta_path_2D: 0.88, status: "warn" },
        { kind: "demand", name: "D3-wind",  N_kN: -2400, Mx_kNm: 180, My_kNm:-120, eta_3D: 0.55, eta_2D: 0.52, eta_path: 0.54, eta_path_2D: 0.51, status: "ok" },
        { kind: "combo",  name: "ULS-1",    N_kN: -2100, Mx_kNm: 310, My_kNm: 140, eta_3D: 0.78, eta_2D: 0.74, eta_path: 0.80, eta_path_2D: 0.75, status: "ok" },
        { kind: "envelope", name: "ENV-A", N_kN: null, Mx_kNm: null, My_kNm: null, eta_3D: 0.91, eta_2D: 0.86, eta_path: 0.93, eta_path_2D: 0.88, status: "warn" },
      ],
    },
    domain: { surface_slices, mx_my_slices, nm_points, mchi },
    meta: { elapsed_ms: 142, cached: false },
  };

  // Wait until GS_STORE exists (data.jsx loaded), then push.
  function tryLoad() {
    if (window.GS_STORE && typeof window.GS_STORE.setFromAnalysis === "function") {
      window.GS_STORE.setFromAnalysis(ar);
      try {
        localStorage.setItem("gensec.yamlText", "# demo synthetic data\n");
        localStorage.setItem("gensec.yamlName", "demo.yaml");
      } catch (_) {}
      console.log("[gensec demo] synthetic data loaded");
    } else {
      setTimeout(tryLoad, 30);
    }
  }
  tryLoad();
})();
