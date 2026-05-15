/* ==========================================================
 * GenSec GUI — data store
 *
 * Two stages:
 *
 * 1) setFromYamlText(text):
 *    Parses the YAML with js-yaml in the BROWSER, populates
 *    SECTION / MATERIALS / DEMANDS / COMBINATIONS / ENVELOPES
 *    from the raw fields, and sets VERIFICATION to an empty
 *    array.  No HTTP call.  Used the moment the user loads a
 *    file or pastes text.
 *
 * 2) setFromAnalysis(ar):
 *    Replaces the store with the AnalysisResult payload
 *    returned by /api/analyze (eta values, full domain).
 * ========================================================== */
(function () {
  "use strict";

  const EMPTY = {
    SECTION: { B: 0, H: 0, bulk_material: "—",
               n_fibers_x: 0, n_fibers_y: 0, rebars: [] },
    MATERIALS: [],
    DEMANDS: [],
    COMBINATIONS: [],
    ENVELOPES: [],
    VERIFICATION: [],
    PROPERTIES: null,
    makeMxMyContour: () => [],
    makeNM: () => [],
    makeMchi: () => [],
    makeSurfaceSlices: () => [],
    _loaded: false,
    _yamlOnly: false,
  };

  const g = (o, k, dflt) =>
    (o && o[k] !== undefined && o[k] !== null) ? o[k] : dflt;

  // ---------- Browser-side YAML parse ----------

  /** Parse YAML text into the GS_DATA shape, ZERO HTTP calls. */
  function fromYamlText(text) {
    let doc;
    try {
      doc = window.jsyaml ? window.jsyaml.load(text) : null;
    } catch (_) {
      doc = null;
    }
    if (!doc || typeof doc !== "object") {
      return Object.assign({}, EMPTY, { _loaded: true, _yamlOnly: true });
    }

    // ---- Section ----
    const sec = doc.section || {};
    let B = Number(sec.B || 0);
    let H = Number(sec.H || 0);

    // Custom polygon: compute bbox so the section preview renders.
    let polygon = null;
    if (sec.shape === "custom" && sec.params && sec.params.exterior) {
      polygon = sec.params.exterior;
      if (Array.isArray(polygon) && polygon.length) {
        const xs = polygon.map(p => Number(p[0]));
        const ys = polygon.map(p => Number(p[1]));
        const xmin = Math.min.apply(null, xs);
        const xmax = Math.max.apply(null, xs);
        const ymin = Math.min.apply(null, ys);
        const ymax = Math.max.apply(null, ys);
        B = B || (xmax - xmin);
        H = H || (ymax - ymin);
      }
    }

    const SECTION = {
      B, H,
      bulk_material: sec.bulk_material || "—",
      n_fibers_x: Number(sec.n_fibers_x || 0) || 1,
      n_fibers_y: Number(sec.n_fibers_y || 0) ||
                  Math.max(1, Math.round(H / Number(sec.mesh_size || 20))),
      polygon: polygon,
      shape: sec.shape || (sec.B && sec.H ? "rect" : null),
      rebars: (sec.rebars || []).map(r => ({
        x: Number(r.x !== undefined ? r.x : (B/2 || 0)),
        y: Number(r.y || 0),
        diameter: Number(r.diameter || 0) || null,
        material: r.material || "—",
        As: Number(r.As || (r.diameter
              ? Math.PI/4 * r.diameter * r.diameter : 0)),
      })),
    };

    // ---- Materials ----
    const matsRaw = doc.materials || {};
    const MATERIALS = Object.entries(matsRaw).map(([id, m]) => ({
      id, kind: (m && m.type) || "—",
      cls: m && (m.class || m.cls) ? (m.class || m.cls) : id,
      fcd: m && m.fck ? Number(m.fck).toFixed(1) + " MPa" : undefined,
      fyd: m && m.fyk ? Number(m.fyk).toFixed(1) + " MPa" : undefined,
      Es:  m && m.Es  ? Number(m.Es).toLocaleString() + " MPa" : undefined,
      eps_su: m && m.eps_su !== undefined ? String(m.eps_su) : undefined,
    }));

    // ---- Demands ----
    const DEMANDS = (doc.demands || []).map(d => ({
      name: String(g(d, "name", "?")),
      N:  Number(g(d, "N_kN",  0)),
      Mx: Number(g(d, "Mx_kNm", g(d, "M_kNm", 0))),
      My: Number(g(d, "My_kNm", 0)),
    }));

    // ---- Combinations (linear resolve of components only) ----
    const demandIdx = Object.fromEntries(DEMANDS.map(d => [d.name, d]));
    const COMBINATIONS = (doc.combinations || []).map(c => {
      const staged = !!c.stages;
      let N = 0, Mx = 0, My = 0;
      let stages = null;
      if (staged) {
        stages = [];
        for (const s of (c.stages || [])) {
          let sN = 0, sMx = 0, sMy = 0;
          for (const comp of (s.components || [])) {
            const d = demandIdx[comp.ref];
            const f = Number(g(comp, "factor", 1.0));
            if (d) { sN += f*d.N; sMx += f*d.Mx; sMy += f*d.My; }
          }
          N += sN; Mx += sMx; My += sMy;
          stages.push({ name: String(s.name||""),
                        cumulative: { N_kN: N, Mx_kNm: Mx, My_kNm: My } });
        }
      } else {
        for (const comp of (c.components || [])) {
          const d = demandIdx[comp.ref];
          const f = Number(g(comp, "factor", 1.0));
          if (d) { N += f*d.N; Mx += f*d.Mx; My += f*d.My; }
        }
      }
      return {
        name: String(c.name),
        staged, stages,
        resolved: { N, Mx, My },
      };
    });

    // ---- Envelopes ----
    const ENVELOPES = (doc.envelopes || []).map(e => ({
      name: String(e.name),
      members: e.members || [],
      eta_max: null,
    }));

    return {
      SECTION, MATERIALS, DEMANDS, COMBINATIONS, ENVELOPES,
      VERIFICATION: [],
      PROPERTIES: null,
      makeMxMyContour: () => [],
      makeNM:          () => [],
      makeMchi:        () => [],
      makeSurfaceSlices: () => [],
      _loaded: true,
      _yamlOnly: true,
    };
  }

  // ---------- AnalysisResult -> GS_DATA (after Run) ----------

  function fromAnalysisResult(ar) {
    const section = ar.section || {};
    const domain = ar.domain || {};

    const SECTION = {
      B: g(section, "B_mm", 0),
      H: g(section, "H_mm", 0),
      bulk_material: g(section, "bulk_material", "—"),
      n_fibers_x: g(section, "n_fibers_x", 0),
      n_fibers_y: g(section, "n_fibers_y", 0),
      rebars: (section.rebars || []).map(r => ({
        x: r.x, y: r.y,
        diameter: r.diameter, material: r.material,
        As: r.As_mm2,
      })),
    };

    const MATERIALS = (ar.materials || []).map(m => ({
      id: m.id, kind: m.kind, cls: m.cls || m.id,
      fcd: m.design_strength_MPa
        ? m.design_strength_MPa.toFixed(2) + " MPa" : undefined,
      Es:  m.modulus_MPa
        ? m.modulus_MPa.toLocaleString() + " MPa" : undefined,
      eps_su: m.eps_ultimate !== null && m.eps_ultimate !== undefined
        ? String(m.eps_ultimate) : undefined,
    }));

    const DEMANDS = (ar.demands || []).map(d => ({
      name: d.name, N: d.N_kN, Mx: d.Mx_kNm, My: d.My_kNm,
    }));

    const COMBINATIONS = (ar.combinations || []).map(c => ({
      name: c.name,
      staged: !!c.staged,
      stages: c.stages || null,
      resolved: {
        N: g(c.resolved, "N_kN", 0),
        Mx: g(c.resolved, "Mx_kNm", 0),
        My: g(c.resolved, "My_kNm", 0),
      },
    }));

    const ENVELOPES = (ar.envelopes || []).map(e => ({
      name: e.name, members: e.members || [], eta_max: e.eta_max,
    }));

    const VERIFICATION = (ar.verification || []).map(r => ({
      kind: r.kind === "combination" ? "combo" : r.kind,
      name: r.name,
      N:  (r.N_kN  !== null && r.N_kN  !== undefined) ? r.N_kN  : "—",
      Mx: (r.Mx_kNm!== null && r.Mx_kNm!== undefined) ? r.Mx_kNm: "—",
      My: (r.My_kNm!== null && r.My_kNm!== undefined) ? r.My_kNm: "—",
      eta3D: r.eta_norm,                     // legacy alias for the table
      eta2D: r.eta_2D,
      etaPath: r.eta_path_norm_ray,
      etaPath2D: r.eta_path_2D,
      eta_norm: r.eta_norm,
      eta_norm_beta: r.eta_norm_beta,
      eta_norm_ray: r.eta_norm_ray,
      eta_governing: r.eta_governing,
      status: r.status || "ok",
      staged: !!r.staged,
    }));

    return {
      SECTION, MATERIALS, DEMANDS, COMBINATIONS, ENVELOPES, VERIFICATION,
      PROPERTIES: ar.properties || null,
      makeMxMyContour: () => [],
      makeNM:          () => [],
      makeMchi:        () => [],
      makeSurfaceSlices: () => [],
      _loaded: true,
      _yamlOnly: false,
      _meta: ar.meta || {},
    };
  }

  // ---------- Store ----------

  const listeners = new Set();
  const GS_STORE = {
    isLoaded() { return !!(window.GS_DATA && window.GS_DATA._loaded); },
    isYamlOnly() {
      return !!(window.GS_DATA && window.GS_DATA._yamlOnly);
    },
    setFromYamlText(text) {
      window.GS_DATA = fromYamlText(text);
      listeners.forEach(fn => { try { fn(); } catch (_) {} });
    },
    setFromAnalysis(ar) {
      window.GS_DATA = fromAnalysisResult(ar);
      listeners.forEach(fn => { try { fn(); } catch (_) {} });
    },
    clear() {
      window.GS_DATA = Object.assign({}, EMPTY);
      listeners.forEach(fn => { try { fn(); } catch (_) {} });
    },
    subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  };

  window.GS_DATA = Object.assign({}, EMPTY);
  window.GS_STORE = GS_STORE;
})();
