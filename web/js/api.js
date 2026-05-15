/* ==========================================================
 * GenSec GUI — HTTP client
 *
 * Wraps fetch().  Only the analyze endpoint is wired in the
 * frontend; everything else is parsed client-side from the
 * YAML buffer (see data.jsx).
 * ========================================================== */
(function () {
  "use strict";

  const urlParams = new URLSearchParams(location.search);
  const BASE = (urlParams.get("api") || "").replace(/\/$/, "");

  async function postJSON(path, body) {
    const r = await fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const text = await r.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) {}
    if (!r.ok) {
      const detail = (data && data.detail) || text || r.statusText;
      const err = new Error(detail);
      err.status = r.status;
      throw err;
    }
    return data;
  }

  async function getJSON(path) {
    const r = await fetch(BASE + path);
    if (!r.ok) throw new Error(r.statusText);
    return r.json();
  }

  window.GensecAPI = {
    health: () => getJSON("/api/health"),
    analyze: (yaml_text) => postJSON("/api/analyze", { yaml_text }),
    contour: (yaml_text, N_kN, n_angles = 144) =>
      postJSON("/api/contour", { yaml_text, N_kN, n_angles }),
    verifyPoint: (yaml_text, N_kN, Mx_kNm, My_kNm = 0) =>
      postJSON("/api/verify-point", { yaml_text, N_kN, Mx_kNm, My_kNm }),
    plot: (kind, yaml_text, options = {}, w = 1200, h = 800, dpi = 150) =>
      postJSON("/api/plot/" + encodeURIComponent(kind),
               { yaml_text, width_px: w, height_px: h, dpi, options }),
  };
})();
