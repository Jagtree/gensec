/* ==========================================================
 * GenSec GUI — main app (backend-connected)
 *
 * v0.3 behaviour change:
 *   - Loading a YAML (file / drop / Load example / Edit YAML / localStorage)
 *     ONLY puts the text in the buffer.  The solver runs only when the
 *     user explicitly clicks "Run".  This protects against accidentally
 *     burning seconds on every drop.
 *   - Visible solving overlay with indeterminate progress bar so the
 *     user knows something is happening, even though the backend does
 *     not report per-step progress.
 * ========================================================== */

const { useState, useEffect, useMemo, useRef, useCallback } = React;

function useSize(ref) {
  const [size, setSize] = useState({ w: 320, h: 240 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const cr = e.contentRect;
        setSize({ w: Math.max(100, cr.width), h: Math.max(100, cr.height) });
      }
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

const DEFAULT_FLAGS = {
  eta_3D: true, eta_2D: true, eta_path: true, eta_path_2D: true,
  delta_N_tol: 0.03,
  generate_mx_my: true, generate_3d_surface: true,
  generate_moment_curvature: true, generate_polar_ductility: true,
  generate_3d_moment_curvature: true,
  n_angles_mx_my: 144,
};

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent_h": 225,
  "dark": false,
  "serif": true,
  "compact": false,
  "grid": true,
  "rail_w": 260
}/*EDITMODE-END*/;

const LS_YAML_TEXT = "gensec.yamlText";
const LS_YAML_NAME = "gensec.yamlName";

// Starter template used by the "New…" topbar button.
const NEW_YAML_TEMPLATE = `# GenSec input — minimal starter
materials:
  C25:
    type: concrete_ec2_gen1
    class: C25/30
    ls: F
    loadtype: slow
    TypeConc: R

  B450C:
    type: steel
    fyk: 450.0
    gamma_s: 1.15
    works_in_compression: true

section:
  B: 300
  H: 500
  bulk_material: C25
  n_fibers_y: 40
  n_fibers_x: 20
  rebars:
    - {x: 40,  y: 40,  diameter: 20, material: B450C}
    - {x: 260, y: 40,  diameter: 20, material: B450C}
    - {x: 40,  y: 460, diameter: 20, material: B450C}
    - {x: 260, y: 460, diameter: 20, material: B450C}

demands:
  - name: G
    N_kN: -1500
    Mx_kNm: 120
    My_kNm: 0

output:
  eta_norm: true
  eta_norm_beta: true
  eta_2D: true
`;

// ---- Export helpers (PNG via Plotly, CSV/JSON regen from GS_DATA) ----
function _findPlotlyDiv(stageEl) {
  if (!stageEl) return null;
  return stageEl.querySelector(".js-plotly-plot");
}

function _download(filename, text, mime) {
  const blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
}

function _toCSV(rows) {
  return rows.map(r => r.map(c => {
    if (c === null || c === undefined) return "";
    const s = String(c);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }).join(",")).join("\n");
}

function _tabPayload(tab, ctx) {
  const D = window.GS_DATA;
  if (tab === "mxmy") {
    const pts = D.makeMxMyContour ? D.makeMxMyContour(ctx.nSlice, 144) : [];
    return { kind: "mxmy", N_kN: ctx.nSlice,
      contour: pts.map(([Mx, My]) => ({ Mx, My })),
      demands: ctx.demandPoints || [] };
  }
  if (tab === "nm") {
    const pts = D.makeNM ? D.makeNM() : [];
    return { kind: "nm", contour: pts.map(([N, M]) => ({ N, M })),
             demands: ctx.demandPoints || [] };
  }
  if (tab === "surface") {
    const sl = D.makeSurfaceSlices ? D.makeSurfaceSlices() : [];
    return { kind: "surface",
      slices: sl.map(s => ({ N_kN: s.N,
        contour: s.contour.map(([Mx, My]) => ({ Mx, My })) })),
      demands: ctx.demandPoints || [] };
  }
  if (tab === "mchi") {
    const ser = D.makeMchi ? D.makeMchi() : [];
    return { kind: "mchi",
      series: ser.map(s => ({ N_kN: s.N,
        points: s.pts.map(([chi, M]) => ({ chi, M })) })) };
  }
  if (tab === "section") return { kind: "section", section: D.SECTION };
  return { kind: tab };
}

function exportPNG(tab, stageEl) {
  const div = _findPlotlyDiv(stageEl);
  if (div && window.Plotly) {
    window.Plotly.downloadImage(div, {
      format: "png", scale: 2, filename: "gensec-" + tab });
    return;
  }
  const svg = stageEl && stageEl.querySelector("svg");
  if (!svg) return;
  const xml = new XMLSerializer().serializeToString(svg);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = svg.clientWidth * 2; c.height = svg.clientHeight * 2;
    const ctx = c.getContext("2d");
    ctx.scale(2, 2);
    ctx.drawImage(img, 0, 0);
    c.toBlob(b => {
      const url = URL.createObjectURL(b);
      const a = document.createElement("a");
      a.href = url; a.download = "gensec-" + tab + ".png";
      a.click(); URL.revokeObjectURL(url);
    });
  };
  img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(xml)));
}

function exportCSV(tab, ctx) {
  const p = _tabPayload(tab, ctx);
  let rows = [];
  if (p.kind === "mxmy") {
    rows.push(["# Mx-My contour at N =", p.N_kN, "kN"]);
    rows.push(["Mx_kNm", "My_kNm"]);
    p.contour.forEach(({ Mx, My }) => rows.push([Mx, My]));
    rows.push([]); rows.push(["# Demands"]);
    rows.push(["name", "N_kN", "Mx_kNm", "My_kNm"]);
    (p.demands || []).forEach(d => rows.push([d.name, d.N, d.Mx, d.My]));
  } else if (p.kind === "nm") {
    rows.push(["# N-M interaction (uniaxial about X)"]);
    rows.push(["N_kN", "M_kNm"]);
    p.contour.forEach(({ N, M }) => rows.push([N, M]));
    rows.push([]); rows.push(["# Demands"]);
    rows.push(["name", "N_kN", "Mx_kNm", "My_kNm"]);
    (p.demands || []).forEach(d => rows.push([d.name, d.N, d.Mx, d.My]));
  } else if (p.kind === "surface") {
    rows.push(["N_kN", "Mx_kNm", "My_kNm"]);
    p.slices.forEach(s => s.contour.forEach(({ Mx, My }) =>
      rows.push([s.N_kN, Mx, My])));
  } else if (p.kind === "mchi") {
    rows.push(["N_kN", "chi_1_per_mm", "M_kNm"]);
    p.series.forEach(s => s.points.forEach(({ chi, M }) =>
      rows.push([s.N_kN, chi, M])));
  } else if (p.kind === "section") {
    const s = p.section || {};
    rows.push(["B_mm", s.B]); rows.push(["H_mm", s.H]);
    rows.push(["n_fibers_x", s.n_fibers_x]);
    rows.push(["n_fibers_y", s.n_fibers_y]);
    rows.push([]); rows.push(["# Rebars"]);
    rows.push(["x_mm", "y_mm", "diameter_mm", "material"]);
    (s.rebars || []).forEach(r =>
      rows.push([r.x, r.y, r.diameter, r.material || ""]));
  }
  _download("gensec-" + tab + ".csv", _toCSV(rows), "text/csv;charset=utf-8");
}

function exportJSON(tab, ctx) {
  const p = _tabPayload(tab, ctx);
  _download("gensec-" + tab + ".json", JSON.stringify(p, null, 2),
            "application/json");
}

// ----- Empty-state / drop-zone panel -----
// State machine:
//   no yamlText        -> drop zone, primary CTA = Open / Load example
//   yamlText + !loaded -> "ready" panel, primary CTA = Run analysis
//   loaded             -> not rendered (real plots take over)
function EmptyState({ hasYaml, yamlName, onLoadFile, onLoadExample,
                     onRun, onInspect, onEdit, loading, error }) {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  return (
    <div style={{
      position: "absolute", inset: 0,
      display: "grid", placeItems: "center",
      background: "var(--paper)",
    }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files && e.dataTransfer.files[0];
          if (f) onLoadFile(f);
        }}
        onClick={() => { if (!hasYaml) fileInputRef.current && fileInputRef.current.click(); }}
        style={{
          width: "min(560px, 86%)", padding: "44px 40px",
          border: `2px ${hasYaml ? "solid" : "dashed"} ${
            dragOver ? "var(--accent)" :
            hasYaml ? "var(--rule-2)" : "var(--rule-2)"}`,
          borderRadius: "12px",
          background: dragOver ? "var(--accent-soft)" : "var(--paper-2)",
          textAlign: "center",
          cursor: hasYaml ? "default" : "pointer",
          transition: "all 0.15s",
        }}
      >
        <input ref={fileInputRef} type="file" accept=".yaml,.yml"
               style={{ display: "none" }}
               onChange={(e) => e.target.files[0] && onLoadFile(e.target.files[0])} />

        {!hasYaml && (
          <>
            <div style={{ fontFamily: "var(--ff-serif)", fontSize: 22,
                          fontWeight: 600, marginBottom: 10 }}>
              Load a GenSec YAML to begin
            </div>
            <div style={{ color: "var(--ink-3)", fontSize: 13, marginBottom: 20 }}>
              Drop a <code style={{ fontFamily: "var(--ff-mono)" }}>.yaml</code>{" "}
              file here, or click to browse.
            </div>
            <button className="btn primary"
                    onClick={(e) => { e.stopPropagation();
                                       fileInputRef.current.click(); }}>
              Open YAML…
            </button>
            <span style={{ margin: "0 12px", color: "var(--ink-3)" }}>or</span>
            <button className="btn"
                    onClick={(e) => { e.stopPropagation(); onLoadExample(); }}>
              Load example
            </button>
          </>
        )}

        {hasYaml && (
          <>
            <div style={{ fontFamily: "var(--ff-serif)", fontSize: 20,
                          fontWeight: 600, marginBottom: 6 }}>
              YAML loaded
            </div>
            <div style={{ color: "var(--ink-3)", fontSize: 13, marginBottom: 6 }}>
              Buffer:
            </div>
            <div style={{ fontFamily: "var(--ff-mono)", fontSize: 13,
                          color: "var(--ink)", marginBottom: 20,
                          padding: "4px 10px", display: "inline-block",
                          background: "var(--paper-3)", borderRadius: 4 }}>
              {yamlName}
            </div>
            <div style={{ marginBottom: 14, color: "var(--ink-3)",
                          fontSize: 12 }}>
              <b>Inspect</b> — parse + section properties (fast, no solver).<br/>
              <b>Run analysis</b> — full solver + verification (seconds).
            </div>
            <div style={{ marginBottom: 18, display: "flex",
                          justifyContent: "center", gap: 10 }}>
              <button className="btn" disabled={loading}
                      style={{ height: 36, padding: "0 22px", fontSize: 13 }}
                      onClick={onInspect}>
                Inspect
              </button>
              <button className="btn primary" disabled={loading}
                      style={{ height: 36, padding: "0 22px", fontSize: 13 }}
                      onClick={onRun}>
                {loading ? "Solving…" : "▸ Run analysis"}
              </button>
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: 8 }}>
              <button className="btn ghost sm" disabled={loading}
                      onClick={onEdit}>Edit YAML</button>
              <button className="btn ghost sm" disabled={loading}
                      onClick={(e) => { e.stopPropagation();
                                         fileInputRef.current.click(); }}>
                Choose different file
              </button>
            </div>
          </>
        )}

        {loading && (
          <SolvingIndicator />
        )}

        {error && (
          <div style={{
            marginTop: 24, padding: "10px 14px",
            background: "var(--fail-soft)", color: "var(--fail)",
            borderRadius: 6, fontFamily: "var(--ff-mono)", fontSize: 11.5,
            textAlign: "left", whiteSpace: "pre-wrap",
          }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

// ----- Indeterminate progress strip -----
function SolvingIndicator() {
  return (
    <div style={{ marginTop: 22 }}>
      <div style={{ color: "var(--ink-2)", fontFamily: "var(--ff-mono)",
                    fontSize: 12, marginBottom: 8 }}>
        Solving — this may take a few seconds…
      </div>
      <div style={{
        height: 4, background: "var(--paper-3)", borderRadius: 2,
        overflow: "hidden", position: "relative", maxWidth: 320,
        marginLeft: "auto", marginRight: "auto",
      }}>
        <div style={{
          position: "absolute", top: 0, left: 0, height: "100%",
          width: "30%", background: "var(--accent)",
          borderRadius: 2,
          animation: "gs-pb 1.2s ease-in-out infinite",
        }}/>
      </div>
      <style>{`
        @keyframes gs-pb {
          0%   { left: -30%; }
          50%  { left: 50%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  );
}

// Overlay used on top of the plot stage while a re-run is in progress.
function SolvingOverlay() {
  return (
    <div style={{
      position: "absolute", inset: 0, zIndex: 30,
      background: "color-mix(in oklab, var(--paper) 75%, transparent)",
      backdropFilter: "blur(2px)",
      display: "grid", placeItems: "center",
    }}>
      <div style={{
        background: "var(--paper)", border: "1px solid var(--rule-2)",
        borderRadius: 10, padding: "20px 28px", boxShadow: "var(--sh-2)",
        textAlign: "center", minWidth: 280,
      }}>
        <div style={{ fontFamily: "var(--ff-serif)", fontSize: 16,
                      fontWeight: 600, marginBottom: 6 }}>
          Running analysis
        </div>
        <SolvingIndicator />
      </div>
    </div>
  );
}

// ----- YAML editor modal -----
function YamlEditor({ open, initialText, initialName, onClose, onSubmit, onLoad }) {
  const [text, setText] = useState(initialText || "");
  const [name, setName] = useState(initialName || "untitled.yaml");
  useEffect(() => {
    if (open) { setText(initialText || "");
                 setName(initialName || "untitled.yaml"); }
  }, [open, initialText, initialName]);

  if (!open) return null;
  return (
    <div style={{
      position: "fixed", inset: 0, background: "oklch(0% 0 0 / 0.35)",
      display: "grid", placeItems: "center", zIndex: 100,
    }} onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: "min(900px, 92vw)", height: "min(80vh, 720px)",
        background: "var(--paper)", border: "1px solid var(--rule-2)",
        borderRadius: 10, boxShadow: "var(--sh-2)",
        display: "flex", flexDirection: "column", overflow: "hidden",
      }}>
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 14px", borderBottom: "1px solid var(--rule)",
          background: "var(--paper-2)",
        }}>
          <h2 style={{ flex: 1 }}>Edit YAML</h2>
          <input value={name} onChange={(e) => setName(e.target.value)}
                 style={{
                   width: 240, height: 26, padding: "0 8px",
                   border: "1px solid var(--rule-2)", borderRadius: 4,
                   background: "var(--paper)",
                   fontFamily: "var(--ff-mono)", fontSize: 12,
                 }} />
          <button className="btn ghost sm" onClick={onClose}>Close</button>
          <button className="btn" onClick={() => onLoad(text, name)}>
            Save buffer
          </button>
          <button className="btn primary" onClick={() => onSubmit(text, name)}>
            Save &amp; run
          </button>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
          style={{
            flex: 1, width: "100%", resize: "none",
            padding: 16, border: "none", outline: "none",
            background: "var(--paper)", color: "var(--ink)",
            fontFamily: "var(--ff-mono)", fontSize: 12.5, lineHeight: 1.55,
            tabSize: 2,
          }}
        />
      </div>
    </div>
  );
}

// ----- Error banner -----
function ErrorBanner({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div style={{
      position: "fixed", top: 52, right: 16, zIndex: 90,
      maxWidth: 420, padding: "10px 14px",
      background: "var(--fail-soft)", color: "var(--fail)",
      border: "1px solid var(--fail)", borderRadius: 6,
      boxShadow: "var(--sh-2)",
      fontFamily: "var(--ff-mono)", fontSize: 11.5,
      display: "flex", alignItems: "flex-start", gap: 8,
    }}>
      <div style={{ flex: 1, whiteSpace: "pre-wrap" }}>{message}</div>
      <button className="btn ghost sm" onClick={onDismiss}
              style={{ color: "var(--fail)" }}>×</button>
    </div>
  );
}

function PropertiesTable({ properties }) {
  const p = properties;
  if (!p) {
    return (
      <div style={{padding:32, textAlign:"center",
                   color:"var(--ink-3)", fontSize:12,
                   fontFamily:"var(--ff-mono)"}}>
        Section properties not available for this YAML.
      </div>
    );
  }
  // Number formatter for mm², mm³, mm⁴, etc.
  const fmt = (v, digits = 1) => {
    if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "—";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e6 || a < 1e-2) return v.toExponential(3);
    return v.toFixed(digits).replace(/\B(?=(\d{3})+(?!\.))/g, " ");
  };
  const deg = (rad) => (rad === null || rad === undefined ? "—"
                        : (rad * 180 / Math.PI).toFixed(2) + "°");
  const Section = ({ title, children }) => (
    <div style={{padding:"12px 16px", borderBottom:"1px solid var(--rule)"}}>
      <div className="overline" style={{marginBottom:8}}>{title}</div>
      <div style={{display:"grid",
                   gridTemplateColumns:"repeat(auto-fit, minmax(220px, 1fr))",
                   gap:"4px 24px"}}>
        {children}
      </div>
    </div>
  );
  const Row = ({ label, value, unit }) => (
    <div style={{display:"flex", alignItems:"baseline",
                 justifyContent:"space-between", gap:8,
                 fontFamily:"var(--ff-mono)", fontSize:11.5,
                 padding:"2px 0"}}>
      <span style={{color:"var(--ink-2)"}}>{label}</span>
      <span>
        <b style={{color:"var(--ink)"}}>{value}</b>
        {unit && <span style={{marginLeft:4, color:"var(--ink-3)",
                                 fontSize:10.5}}>{unit}</span>}
      </span>
    </div>
  );
  return (
    <div style={{overflow:"auto"}}>
      <Section title="Homogenization">
        <Row label="E_ref" value={fmt(p.E_ref_MPa)} unit="MPa"/>
        <Row label="E_bulk" value={fmt(p.E_bulk_MPa)} unit="MPa"/>
        <Row label="n_bulk" value={fmt(p.n_bulk, 3)}/>
        <Row label="convex" value={p.is_convex ? "yes" : "no"}/>
      </Section>
      <Section title="Area & centroid">
        <Row label="A_id" value={fmt(p.area_mm2)} unit="mm²"/>
        <Row label="x_G" value={fmt(p.xg_mm)} unit="mm"/>
        <Row label="y_G" value={fmt(p.yg_mm)} unit="mm"/>
        <Row label="S_x" value={fmt(p.Sx_mm3)} unit="mm³"/>
        <Row label="S_y" value={fmt(p.Sy_mm3)} unit="mm³"/>
      </Section>
      <Section title="Second moments (centroidal, user frame)">
        <Row label="I_x" value={fmt(p.Ix_mm4)} unit="mm⁴"/>
        <Row label="I_y" value={fmt(p.Iy_mm4)} unit="mm⁴"/>
        <Row label="I_xy" value={fmt(p.Ixy_mm4)} unit="mm⁴"/>
        <Row label="I_polar" value={fmt(p.I_polar_mm4)} unit="mm⁴"/>
        <Row label="ρ_x" value={fmt(p.rho_x_mm)} unit="mm"/>
        <Row label="ρ_y" value={fmt(p.rho_y_mm)} unit="mm"/>
      </Section>
      <Section title="Principal axes">
        <Row label="I_ξ" value={fmt(p.I_xi_mm4)} unit="mm⁴"/>
        <Row label="I_η" value={fmt(p.I_eta_mm4)} unit="mm⁴"/>
        <Row label="α" value={deg(p.alpha_rad)}/>
        <Row label="ρ_ξ" value={fmt(p.rho_xi_mm)} unit="mm"/>
        <Row label="ρ_η" value={fmt(p.rho_eta_mm)} unit="mm"/>
      </Section>
      <Section title="Extreme fibre distances">
        <Row label="c_y_top" value={fmt(p.c_y_top_mm)} unit="mm"/>
        <Row label="c_y_bot" value={fmt(p.c_y_bot_mm)} unit="mm"/>
        <Row label="c_x_left" value={fmt(p.c_x_left_mm)} unit="mm"/>
        <Row label="c_x_right" value={fmt(p.c_x_right_mm)} unit="mm"/>
        <Row label="c_ξ_pos" value={fmt(p.c_xi_pos_mm)} unit="mm"/>
        <Row label="c_ξ_neg" value={fmt(p.c_xi_neg_mm)} unit="mm"/>
        <Row label="c_η_pos" value={fmt(p.c_eta_pos_mm)} unit="mm"/>
        <Row label="c_η_neg" value={fmt(p.c_eta_neg_mm)} unit="mm"/>
      </Section>
      <Section title="Elastic section moduli">
        <Row label="W_x_top" value={fmt(p.W_x_top_mm3)} unit="mm³"/>
        <Row label="W_x_bot" value={fmt(p.W_x_bot_mm3)} unit="mm³"/>
        <Row label="W_y_left" value={fmt(p.W_y_left_mm3)} unit="mm³"/>
        <Row label="W_y_right" value={fmt(p.W_y_right_mm3)} unit="mm³"/>
        <Row label="W_ξ_pos" value={fmt(p.W_xi_pos_mm3)} unit="mm³"/>
        <Row label="W_ξ_neg" value={fmt(p.W_xi_neg_mm3)} unit="mm³"/>
        <Row label="W_η_pos" value={fmt(p.W_eta_pos_mm3)} unit="mm³"/>
        <Row label="W_η_neg" value={fmt(p.W_eta_neg_mm3)} unit="mm³"/>
      </Section>
      <Section title="Plastic section moduli">
        <Row label="Z_x" value={fmt(p.Z_x_mm3)} unit="mm³"/>
        <Row label="Z_y" value={fmt(p.Z_y_mm3)} unit="mm³"/>
        <Row label="Z_ξ" value={fmt(p.Z_xi_mm3)} unit="mm³"/>
        <Row label="Z_η" value={fmt(p.Z_eta_mm3)} unit="mm³"/>
      </Section>
      {p.I_t_mm4 !== null && p.I_t_mm4 !== undefined && (
        <Section title="Torsion">
          <Row label="I_t" value={fmt(p.I_t_mm4)} unit="mm⁴"/>
        </Section>
      )}
    </div>
  );
}

// Small floating overlay on the Section view showing the headline
// numbers (A, x_G/y_G, I_x, I_y, alpha).
function PropertiesOverlay({ properties }) {
  if (!properties) return null;
  const p = properties;
  const fmt = (v, digits = 1) => {
    if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "—";
    const a = Math.abs(v);
    if (a >= 1e6 || a < 1e-2) return v.toExponential(2);
    return v.toFixed(digits).replace(/\B(?=(\d{3})+(?!\.))/g, " ");
  };
  return (
    <div style={{
      position:"absolute", left:14, bottom:14,
      background:"color-mix(in oklab, var(--paper) 92%, transparent)",
      backdropFilter:"blur(4px)",
      border:"1px solid var(--rule)",
      borderRadius:"var(--r-2)",
      padding:"10px 12px",
      fontFamily:"var(--ff-mono)", fontSize:11,
      color:"var(--ink-2)",
      display:"grid",
      gridTemplateColumns:"max-content max-content",
      gap:"3px 14px",
    }}>
      <span>A_id</span>     <b style={{color:"var(--ink)"}}>{fmt(p.area_mm2)} mm²</b>
      <span>x_G</span>      <b style={{color:"var(--ink)"}}>{fmt(p.xg_mm)} mm</b>
      <span>y_G</span>      <b style={{color:"var(--ink)"}}>{fmt(p.yg_mm)} mm</b>
      <span>I_x</span>      <b style={{color:"var(--ink)"}}>{fmt(p.Ix_mm4)} mm⁴</b>
      <span>I_y</span>      <b style={{color:"var(--ink)"}}>{fmt(p.Iy_mm4)} mm⁴</b>
      <span>α</span>        <b style={{color:"var(--ink)"}}>{(p.alpha_rad*180/Math.PI).toFixed(2)}°</b>
    </div>
  );
}

function App() {
  // ---- Tabs / selection ----
  const [tab, setTab] = useState("section");
  const [resTab, setResTab] = useState("verify");
  const [selected, setSelected] = useState(null);
  const [flags, setFlags] = useState(DEFAULT_FLAGS);
  const [nSlice, setNSlice] = useState(0);
  const [surfaceMode, setSurfaceMode] = useState("slices");

  // ---- Tweaks ----
  const [tweaks, setTweaks] = useState(TWEAK_DEFAULTS);
  const [tweaksOpen, setTweaksOpen] = useState(false);

  // ---- Backend state ----
  const [yamlText, setYamlText] = useState("");
  const [yamlName, setYamlName] = useState("(no file)");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [dataVersion, setDataVersion] = useState(0);

  useEffect(() =>
    window.GS_STORE.subscribe(() => setDataVersion(v => v + 1)), []);

  // Apply tweaks to CSS.
  useEffect(() => {
    document.documentElement.setAttribute("data-theme",
      tweaks.dark ? "dark" : "light");
    document.documentElement.style.setProperty("--accent-h", tweaks.accent_h);
    document.documentElement.style.setProperty("--rail-w", tweaks.rail_w + "px");
    document.documentElement.style.setProperty("--row-h",
      tweaks.compact ? "24px" : "28px");
    document.documentElement.style.setProperty("--pad",
      tweaks.compact ? "10px" : "14px");
    if (!tweaks.serif)
      document.documentElement.style.setProperty("--ff-serif", 'var(--ff-sans)');
    else
      document.documentElement.style.removeProperty("--ff-serif");
  }, [tweaks]);

  // Edit-mode protocol.
  useEffect(() => {
    const handler = (ev) => {
      const d = ev.data;
      if (!d || typeof d !== "object") return;
      if (d.type === "__activate_edit_mode") setTweaksOpen(true);
      if (d.type === "__deactivate_edit_mode") setTweaksOpen(false);
    };
    window.addEventListener("message", handler);
    window.parent.postMessage({ type: "__edit_mode_available" }, "*");
    return () => window.removeEventListener("message", handler);
  }, []);

  const setTweak = (k, v) => {
    setTweaks((t) => {
      const next = { ...t, [k]: v };
      try {
        window.parent.postMessage({ type: "__edit_mode_set_keys",
                                     edits: { [k]: v } }, "*");
      } catch {}
      return next;
    });
  };
  const setFlag = (k, v) => setFlags((f) => ({ ...f, [k]: v }));

  // ---- Buffer management ----
  //
  // Loading a YAML (file, drop, example, editor save, New template)
  // **only** stores the text in the buffer.  No HTTP call.  No
  // calculation.  The user explicitly chooses Inspect or Run.
  const loadBuffer = useCallback((text, name) => {
    setYamlText(text);
    setYamlName(name || "(buffer)");
    setError(null);
    // Drop any previously-computed data so the UI reflects the
    // new buffer's identity ("this YAML is not solved yet").
    try { window.GS_STORE.clear(); } catch (_) {}
    try {
      localStorage.setItem(LS_YAML_NAME, name || "(buffer)");
    } catch (_) {}
  }, []);

  const handleLoadFile = useCallback(async (file) => {
    const text = await file.text();
    loadBuffer(text, file.name);
  }, [loadBuffer]);

  const handleLoadExample = useCallback(async () => {
    setError(null);
    try {
      const r = await fetch("examples/biaxial_column.yaml");
      if (!r.ok) throw new Error(
        "No example bundled (expected web/examples/biaxial_column.yaml).");
      const text = await r.text();
      loadBuffer(text, "biaxial_column.yaml");
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [loadBuffer]);

  // ---- Inspect: explicit user action.  Parser + section properties.
  // No solver, no resistance domain, no verification.
  const runInspect = useCallback(async () => {
    if (!yamlText || !yamlText.trim()) {
      setError("YAML buffer is empty.  Open, drop, or paste a YAML first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      console.log("[GenSec] /api/inspect requested, yaml length =",
                  yamlText.length);
      const ir = await window.GensecAPI.inspect(yamlText);
      console.log("[GenSec] /api/inspect OK, elapsed_ms =",
                  ir && ir.meta && ir.meta.elapsed_ms);
      window.GS_STORE.setFromInspect(ir);
      setTab("section");
    } catch (e) {
      console.error("[GenSec] /api/inspect FAILED:", e);
      setError("Inspect failed: " + (e.message || String(e)));
    } finally {
      setLoading(false);
    }
  }, [yamlText]);

  // ---- Solver call (only on explicit user action) ----
  const runAnalyze = useCallback(async (text, name) => {
    const t = (text !== undefined && text !== null) ? text : yamlText;
    const n = (name !== undefined && name !== null) ? name : yamlName;
    console.log("[GenSec] runAnalyze invoked, yaml length =",
                (t || "").length, "name =", n);
    if (!t || !t.trim()) {
      setError("YAML is empty. Load a file or paste content into the editor first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await window.GensecAPI.analyze(t);
      console.log("[GenSec] analyze OK, elapsed_ms =",
                  res && res.meta && res.meta.elapsed_ms);
      window.GS_STORE.setFromAnalysis(res);
      if (text !== undefined) setYamlText(t);
      if (name !== undefined) setYamlName(n || "(buffer)");
      const D2 = window.GS_DATA;
      if (D2.DEMANDS.length) {
        setSelected("dem:" + D2.DEMANDS[0].name);
        setNSlice(Math.round(D2.DEMANDS[0].N || 0));
      } else {
        setSelected(null);
      }
    } catch (e) {
      console.error("[GenSec] analyze FAILED:", e);
      setError("Analyze failed: " + (e.message || String(e)));
    } finally {
      setLoading(false);
    }
  }, [yamlText, yamlName]);

  // Auto-restore last YAML on mount, but DO NOT auto-run.
  // No auto-restore on mount.  Every page reload starts from the
  // empty state.  The previous YAML file name is shown as a hint
  // in the topbar, but nothing is loaded automatically — the user
  // explicitly opens it.
  useEffect(() => {
    try {
      const n = localStorage.getItem(LS_YAML_NAME);
      if (n) setYamlName(n + "  (not loaded)");
    } catch (_) {}
  }, []);

  // ---- Derived data ----
  const D = window.GS_DATA;
  // eslint-disable-next-line no-unused-vars
  const _ = dataVersion;

  const isEmpty = !window.GS_STORE.isLoaded();
  const isInspectOnly = window.GS_STORE.isInspectOnly();
  const hasYaml = !!(yamlText && yamlText.trim());

  const activeDemand = useMemo(() => {
    if (!selected) return null;
    if (selected.startsWith("dem:"))
      return D.DEMANDS.find((d) => d.name === selected.slice(4));
    if (selected.startsWith("cmb:")) {
      const c = D.COMBINATIONS.find((c) => c.name === selected.slice(4));
      if (c) return { name: c.name, ...c.resolved };
    }
    return null;
  }, [selected, dataVersion]);

  const activeName = activeDemand?.name;

  const demandPoints = useMemo(() => {
    const base = D.DEMANDS.map((d) =>
      ({ name: d.name, N: d.N, Mx: d.Mx, My: d.My }));
    const combos = D.COMBINATIONS.map((c) =>
      ({ name: c.name, ...c.resolved }));
    return base.concat(combos);
  }, [dataVersion]);

  const stageRef = useRef(null);
  const stageSize = useSize(stageRef);

  const tabs = [
    { id: "surface", label: "3D surface", cap: "N–Mx–My resistance surface" },
    { id: "mxmy",    label: "Mx–My",      cap: "Mx–My interaction contour at fixed N" },
    { id: "nm",      label: "N–M",        cap: "Uniaxial N–M interaction" },
    { id: "mchi",    label: "M–χ",        cap: "Moment-curvature" },
    { id: "polar",   label: "Polar",      cap: "Polar ductility rose" },
    { id: "section", label: "Section",    cap: "Fiber mesh & rebars" },
  ];

  const resTabs = [
    { id: "verify",       label: "Verification",
      count: isInspectOnly ? 0 : D.VERIFICATION.length },
    { id: "properties",   label: "Properties",   count: D.PROPERTIES ? 1 : 0 },
    { id: "demands",      label: "Demands",      count: D.DEMANDS.length },
    { id: "combinations", label: "Combinations", count: D.COMBINATIONS.length },
    { id: "envelopes",    label: "Envelopes",    count: D.ENVELOPES.length },
    { id: "materials",    label: "Materials",    count: D.MATERIALS.length },
  ];

  // Helper: build VerificationRow-like records from parsed YAML
  // (used when inspectOnly == true, so the bottom panel still shows
  // demands / combos / envelopes even before the solver has run).
  function _previewRows(D) {
    const out = [];
    (D.DEMANDS || []).forEach(d => out.push({
      kind: "demand", name: d.name,
      N: d.N, Mx: d.Mx, My: d.My,
      eta3D: null, eta2D: null, etaPath: null, etaPath2D: null,
      status: "ok", staged: false,
    }));
    (D.COMBINATIONS || []).forEach(c => {
      const r = c.resolved || {};
      out.push({
        kind: "combo", name: c.name,
        N: r.N, Mx: r.Mx, My: r.My,
        eta3D: null, eta2D: null, etaPath: null, etaPath2D: null,
        status: "ok", staged: !!c.staged,
      });
    });
    (D.ENVELOPES || []).forEach(e => out.push({
      kind: "envelope", name: e.name,
      N: "—", Mx: "—", My: "—",
      eta3D: e.eta_max, eta2D: null, etaPath: null, etaPath2D: null,
      status: "ok", staged: false,
    }));
    return out;
  }

  const resRows = useMemo(() => {
    const preview = isInspectOnly ? _previewRows(D) : null;
    const all = preview || D.VERIFICATION;
    if (resTab === "verify")       return all;
    if (resTab === "demands")      return all.filter(r => r.kind === "demand");
    if (resTab === "combinations") return all.filter(r => r.kind === "combo");
    if (resTab === "envelopes")    return all.filter(r => r.kind === "envelope");
    return [];
  }, [resTab, dataVersion, isInspectOnly]);

  return (
    <div className="app" data-screen-label="GenSec GUI">
      {/* TOPBAR */}
      <div className="topbar">
        <div className="brand">
          <span className="logo">G</span>
          <h1>GenSec</h1>
          <span className="ver">v0.3</span>
        </div>
        <div className="topbar-sep"/>
        <div className="topbar-file">
          <svg className="ic" viewBox="0 0 16 16" fill="none"
               stroke="currentColor" strokeWidth="1.4">
            <path d="M3 2h7l3 3v9H3z"/><path d="M10 2v3h3"/>
          </svg>
          <span className="path"><b>{yamlName}</b>
            {hasYaml && isEmpty && !loading &&
              <span style={{marginLeft:8, color:"var(--warn)",
                            fontSize:10, fontWeight:600,
                            textTransform:"uppercase",
                            letterSpacing:0.06}}>
                not solved
              </span>}
            {isInspectOnly && !loading &&
              <span style={{marginLeft:8, color:"var(--ink-3)",
                            fontSize:10, fontWeight:600,
                            textTransform:"uppercase",
                            letterSpacing:0.06}}>
                inspected
              </span>}
          </span>
        </div>
        <button className="btn ghost sm" onClick={() => {
          const el = document.createElement("input");
          el.type = "file"; el.accept = ".yaml,.yml";
          el.onchange = () => el.files[0] && handleLoadFile(el.files[0]);
          el.click();
        }}>Open…</button>
        <button className="btn ghost sm" onClick={() => {
          loadBuffer(NEW_YAML_TEMPLATE, "untitled.yaml");
          setEditorOpen(true);
        }}>New…</button>
        <button className="btn ghost sm" onClick={() => setEditorOpen(true)}
                disabled={!yamlText}>Edit YAML…</button>
        <button className="btn ghost sm" onClick={runInspect}
                disabled={!yamlText || loading}
                title="Parse the YAML and compute section properties (fast, no solver).">
          Inspect
        </button>
        <div className="topbar-spacer"/>
        <div style={{display:"flex",alignItems:"center",gap:6,
                     color:"var(--ink-3)", fontSize:11,
                     fontFamily:"var(--ff-mono)"}}>
          <span className="dot" style={{
            width:6, height:6, borderRadius:"50%",
            background: loading ? "var(--warn)" :
                        (error ? "var(--fail)" :
                         (hasYaml && isEmpty ? "var(--ink-3)" : "var(--ok)")),
            display:"inline-block",
          }}/>
          {loading ? "solving…" :
           error ? "error" :
           (hasYaml && isEmpty ? "ready (not solved)" :
            (isEmpty ? "idle" : "solved"))}
        </div>
        <div className="topbar-sep"/>
        <button className="btn ghost icon" title="Theme"
                onClick={()=>setTweak("dark", !tweaks.dark)}>
          {tweaks.dark
            ? <svg className="ic" viewBox="0 0 16 16" fill="currentColor">
                <path d="M6 2a6 6 0 1 0 8 8A7 7 0 0 1 6 2z"/></svg>
            : <svg className="ic" viewBox="0 0 16 16" fill="none"
                   stroke="currentColor" strokeWidth="1.4">
                <circle cx="8" cy="8" r="3"/>
                <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.5 1.5M11.5 11.5L13 13M3 13l1.5-1.5M11.5 4.5L13 3"/>
              </svg>}
        </button>
        <button className="btn primary" onClick={() => runAnalyze()}
                disabled={!yamlText || loading}>
          {loading ? "Solving…" : "▸ Run"}
        </button>
      </div>

      {/* MAIN */}
      <div className="main">
        <LeftRail selected={selected} onSelect={setSelected} />

        <section className="center">
          <div className="viewport">
            <div className="vp-tabs">
              {tabs.map(t => (
                <button key={t.id}
                        className={"vp-tab" + (tab===t.id ? " active":"")}
                        onClick={()=>setTab(t.id)}
                        disabled={isEmpty || (isInspectOnly && t.id !== "section")}
                        title={isInspectOnly && t.id !== "section"
                               ? "Click ▸ Run to compute this view" : undefined}>
                  {t.label}
                </button>
              ))}
              <div className="vp-spacer"/>
              <div className="vp-tools">
                {tab === "surface" && (
                  <div style={{display:"flex", gap:2, marginRight:8,
                               padding:2, background:"var(--paper-3)",
                               borderRadius:4}}>
                    {[["slices","Slices"],["mesh","Surface"],
                      ["wireframe","Wire"]].map(([id,lbl])=>(
                      <button key={id}
                              className={"btn ghost sm" +
                                (surfaceMode===id?" primary":"")}
                              style={surfaceMode===id?{height:22}:
                                {height:22, background:"transparent"}}
                              onClick={()=>setSurfaceMode(id)}>{lbl}</button>
                    ))}
                  </div>
                )}
                <button className="btn ghost sm" disabled={isEmpty}
                        onClick={()=>exportPNG(tab, stageRef.current)}>PNG</button>
                <button className="btn ghost sm" disabled={isEmpty}
                        onClick={()=>exportCSV(tab, {nSlice, demandPoints})}>CSV</button>
                <button className="btn ghost sm" disabled={isEmpty}
                        onClick={()=>exportJSON(tab, {nSlice, demandPoints})}>JSON</button>
              </div>
            </div>

            <div className="vp-stage" ref={stageRef}>
              {isEmpty ? (
                <EmptyState
                  hasYaml={hasYaml}
                  yamlName={yamlName}
                  onLoadFile={handleLoadFile}
                  onLoadExample={handleLoadExample}
                  onRun={() => runAnalyze()}
                  onInspect={runInspect}
                  onEdit={() => setEditorOpen(true)}
                  loading={loading}
                  error={error}
                />
              ) : (
                <>
                  <div className="plot-caption">
                    {tabs.find(t=>t.id===tab).cap}
                  </div>
                  <div className="plot-subcaption">
                    {tab === "mxmy"    && `N = ${nSlice} kN   ·   section ${D.SECTION.B}×${D.SECTION.H}`}
                    {tab === "nm"      && `Uniaxial about X   ·   section ${D.SECTION.B}×${D.SECTION.H}`}
                    {tab === "surface" && `convex hull   ·   ${D.MATERIALS.map(m=>m.id).join(" / ")}`}
                    {tab === "mchi"    && `per N-level   ·   ${D.MATERIALS.map(m=>m.id).join(" / ")}`}
                    {tab === "polar"   && "ductility ratio μ_χ"}
                    {tab === "section" && `${D.SECTION.n_fibers_x}×${D.SECTION.n_fibers_y} fibers   ·   ${D.SECTION.rebars.length} rebars`}
                  </div>

                  {tab === "surface" && <Surface3D width={stageSize.w}
                                                    height={stageSize.h}
                                                    mode={surfaceMode}
                                                    highlightN={nSlice}
                                                    demands={demandPoints}
                                                    activeName={activeName} />}
                  {tab === "mxmy" && (
                    <>
                      <MxMyPlot width={stageSize.w} height={stageSize.h - 56}
                                N_kN={nSlice} demands={demandPoints}
                                activeName={activeName}/>
                      <div className="n-slider">
                        <span className="label">N slice</span>
                        <input type="range" min={-4500} max={1200} step={50}
                               value={nSlice}
                               onChange={(e)=>setNSlice(parseInt(e.target.value))}/>
                        <span className="value">{nSlice} kN</span>
                      </div>
                    </>
                  )}
                  {tab === "nm"      && <NMPlot width={stageSize.w}
                                                height={stageSize.h}
                                                demands={demandPoints}
                                                activeName={activeName}/>}
                  {tab === "mchi"    && <MchiPlot width={stageSize.w}
                                                  height={stageSize.h}
                                                  activeN={nSlice}/>}
                  {tab === "polar"   && <PolarDuctility width={stageSize.w}
                                                        height={stageSize.h}/>}
                  {tab === "section" && <>
                    <SectionPreview width={stageSize.w} height={stageSize.h}/>
                    <PropertiesOverlay properties={D.PROPERTIES} />
                  </>}

                  {(tab === "mxmy" || tab === "nm" || tab === "surface") && (
                    <div className="legend">
                      <div className="row">
                        <span className="sw" style={{background:"var(--accent)"}}/>
                        Resistance domain
                      </div>
                      <div className="row">
                        <span className="sw" style={{background:"var(--ink)",
                                                      borderRadius:"50%"}}/>
                        Demand
                      </div>
                      <div className="row">
                        <span className="sw" style={{background:"var(--fail)",
                                                      borderRadius:"50%"}}/>
                        Selected · η-ray
                      </div>
                    </div>
                  )}

                  {loading && <SolvingOverlay />}
                </>
              )}
            </div>
          </div>

          <div className="results">
            <div className="results-tabs">
              {resTabs.map(t => (
                <button key={t.id}
                        className={"results-tab" +
                          (resTab===t.id ? " active":"")}
                        onClick={()=>setResTab(t.id)}>
                  {t.label}<span className="count">{t.count}</span>
                </button>
              ))}
              <div className="vp-spacer"/>
              <div style={{display:"flex", gap:6, padding: "0 6px"}}>
                <button className="btn ghost sm" disabled>Filter</button>
                <button className="btn ghost sm" disabled>Export summary</button>
              </div>
            </div>
            <div className="results-body">
              {isEmpty ? (
                <div style={{
                  padding: 32, textAlign: "center", color: "var(--ink-3)",
                  fontSize: 12, fontFamily: "var(--ff-mono)",
                }}>
                  {hasYaml
                    ? "YAML loaded — click ▸ Run to compute results."
                    : "Load a YAML to see results."}
                </div>
              ) : (isInspectOnly && resTab === "verify") ? (
                <div style={{
                  padding: 32, textAlign: "center", color: "var(--ink-3)",
                  fontSize: 12, fontFamily: "var(--ff-mono)",
                }}>
                  Click ▸ Run to compute the verification table.<br/>
                  In the meantime, the <b>Properties</b> tab is available.
                </div>
              ) : resTab === "properties" ? (
                <PropertiesTable properties={D.PROPERTIES} />
              ) : resTab === "materials" ? (
                <table className="dt">
                  <thead><tr>
                    <th style={{width:24}}></th><th>ID</th><th>Type</th>
                    <th>Class</th>
                    <th>Design strength</th><th>Modulus</th><th>Ultimate</th>
                  </tr></thead>
                  <tbody>
                    {D.MATERIALS.map(m => (
                      <tr key={m.id}>
                        <td><span className="dot"
                                  style={{background:"var(--series-2)"}}/></td>
                        <td className="mono" style={{fontWeight:500}}>{m.id}</td>
                        <td style={{color:"var(--ink-3)"}}>{m.kind}</td>
                        <td className="mono">{m.cls}</td>
                        <td className="mono">{m.fcd || m.fyd || "—"}</td>
                        <td className="mono">{m.Ecm || m.Es || "—"}</td>
                        <td className="mono">{m.eps_su || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <ResultsTable rows={resRows} selected={selected}
                              onSelect={setSelected} flags={flags}/>
              )}
            </div>
          </div>
        </section>

        <RightDrawer flags={flags} setFlag={setFlag}
                     nSlice={nSlice} setNSlice={setNSlice}
                     onRun={() => runAnalyze()} />
      </div>

      {/* STATUSBAR */}
      <div className="statusbar">
        <span>
          <span className="dot" style={{
            background: loading ? "var(--warn)" :
                        (hasYaml && isEmpty ? "var(--ink-3)" : "var(--ok)"),
          }}/>
          gensec {loading ? "solving" :
                  (hasYaml && isEmpty ? "buffer ready" : "ready")}
        </span>
        <span>{yamlName}</span>
        <span>selected · <b style={{color:"var(--ink-2)"}}>
          {selected || "—"}</b></span>
        <div className="spacer"/>
        <span>
          {D.MATERIALS.length} materials · {D.DEMANDS.length} demands ·{" "}
          {D.COMBINATIONS.length} combinations · {D.ENVELOPES.length} envelopes
        </span>
        {D._meta && D._meta.elapsed_ms !== undefined &&
          <span>solved in {D._meta.elapsed_ms} ms
            {D._meta.cached ? " (cached)" : ""}</span>}
      </div>

      <TweaksPanel open={tweaksOpen} onClose={()=>setTweaksOpen(false)}
                   tweaks={tweaks} setTweak={setTweak} />

      <YamlEditor
        open={editorOpen}
        initialText={yamlText}
        initialName={yamlName}
        onClose={() => setEditorOpen(false)}
        onLoad={(text, name) => { loadBuffer(text, name); }}
        onSubmit={(text, name) => {
          setEditorOpen(false);
          loadBuffer(text, name);
          // chain a run after the buffer update
          runAnalyze(text, name);
        }}
      />

      <ErrorBanner message={!isEmpty && error}
                   onDismiss={() => setError(null)} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
