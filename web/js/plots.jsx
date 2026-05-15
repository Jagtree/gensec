/* ==========================================================
 * GenSec GUI — Plotly-backed plot components
 *
 * Each plot is a thin React wrapper around Plotly.newPlot.
 * The wrapper handles:
 *   - mount  → Plotly.newPlot
 *   - update → Plotly.react (in-place, preserves user camera/zoom)
 *   - resize → Plotly.Plots.resize
 *   - unmount → Plotly.purge
 *
 * SectionPreview stays as SVG (technical drawing) but gains
 * mouse/wheel pan+zoom via PannableSVG.
 *
 * Public components (kept on window for app.jsx):
 *   MxMyPlot, NMPlot, Surface3D, MchiPlot, SectionPreview, PolarDuctility
 *   plus exportCurrentPlot(plotKind, format) for the toolbar buttons.
 * ========================================================== */

const { useEffect, useRef, useState, useCallback, useMemo } = React;

// Track the currently mounted Plotly DOM nodes so the topbar
// PNG/CSV/JSON buttons can grab them without prop-drilling.
const PLOT_REGISTRY = {};
function registerPlot(kind, node) { PLOT_REGISTRY[kind] = node; }
function unregisterPlot(kind, node) { if (PLOT_REGISTRY[kind] === node) delete PLOT_REGISTRY[kind]; }
function getPlotNode(kind) { return PLOT_REGISTRY[kind] || null; }

// ---- Generic Plotly wrapper ----------------------------------
function PlotlyChart({ kind, width, height, build, deps = [], filename, useReact = true }) {
  const ref = useRef(null);
  const builtRef = useRef(false);

  useEffect(() => {
    if (!ref.current || !window.Plotly) return;
    window.GS_PLOTLY.refreshTheme();
    const { traces, layout } = build();
    const config = window.GS_PLOTLY.commonConfig({ filename });
    if (!builtRef.current) {
      window.Plotly.newPlot(ref.current, traces, layout, config).then(() => {
        builtRef.current = true;
        registerPlot(kind, ref.current);
      });
    } else if (useReact) {
      window.Plotly.react(ref.current, traces, layout, config);
    } else {
      window.Plotly.newPlot(ref.current, traces, layout, config);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (ref.current && builtRef.current && window.Plotly) {
      window.Plotly.Plots.resize(ref.current);
    }
  }, [width, height]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      unregisterPlot(kind, el);
      if (el && window.Plotly) {
        try { window.Plotly.purge(el); } catch (_) {}
      }
    };
  }, [kind]);

  return <div ref={ref} style={{ width, height }} />;
}

// ---- Plot toolbar (custom buttons sitting over the plot) -----
function PlotToolbar({ children }) {
  return (
    <div style={{
      position: 'absolute', top: 6, right: 10, zIndex: 5,
      display: 'flex', gap: 4, alignItems: 'center',
      background: 'color-mix(in oklab, var(--paper) 88%, transparent)',
      backdropFilter: 'blur(4px)',
      border: '1px solid var(--rule)',
      borderRadius: 'var(--r-2)',
      padding: '3px 4px',
      pointerEvents: 'auto',
    }}>{children}</div>
  );
}
function ToolBtn({ onClick, title, active, children }) {
  return (
    <button className="btn ghost sm" title={title} onClick={onClick}
      style={{
        height: 22, padding: '0 8px', fontSize: 10.5,
        color: active ? 'var(--accent-ink)' : 'var(--ink-2)',
        background: active ? 'var(--accent-soft)' : 'transparent',
      }}>
      {children}
    </button>
  );
}

// ---- Mx-My ---------------------------------------------------
function MxMyPlot({ width, height, N_kN, demands, activeName }) {
  const build = () => window.GS_PLOTLY.buildMxMy({ N_kN, demands, activeName });
  const ref = useRef(null);
  const reset = () => {
    const node = getPlotNode('mxmy');
    if (node && window.Plotly) window.Plotly.relayout(node, { 'xaxis.autorange': true, 'yaxis.autorange': true });
  };
  return (
    <div ref={ref} style={{ position: 'relative', width, height }}>
      <PlotlyChart kind="mxmy" width={width} height={height} build={build}
        deps={[width, height, N_kN, activeName, demands && demands.length]}
        filename={`gensec-mxmy-N${Math.round(N_kN)}`} />
      <PlotToolbar>
        <ToolBtn title="Reset zoom & pan" onClick={reset}>⟲ Reset</ToolBtn>
      </PlotToolbar>
    </div>
  );
}

// ---- N-M -----------------------------------------------------
function NMPlot({ width, height, demands, activeName }) {
  const build = () => window.GS_PLOTLY.buildNM({ demands, activeName });
  const reset = () => {
    const node = getPlotNode('nm');
    if (node && window.Plotly) window.Plotly.relayout(node, { 'xaxis.autorange': true, 'yaxis.autorange': true });
  };
  return (
    <div style={{ position: 'relative', width, height }}>
      <PlotlyChart kind="nm" width={width} height={height} build={build}
        deps={[width, height, activeName, demands && demands.length]}
        filename="gensec-nm" />
      <PlotToolbar>
        <ToolBtn title="Reset zoom & pan" onClick={reset}>⟲ Reset</ToolBtn>
      </PlotToolbar>
    </div>
  );
}

// ---- 3D resistance surface -----------------------------------
function Surface3D({ width, height, highlightN, demands, activeName, mode: modeProp }) {
  const [internalMode, setInternalMode] = useState('slices');
  const mode = modeProp || internalMode;             // 'slices' | 'mesh' | 'wireframe'
  const setMode = setInternalMode;
  const [proj, setProj] = useState('perspective');    // 'perspective' | 'orthographic'

  const build = () => window.GS_PLOTLY.buildSurface3D({
    mode, highlightN, demands, activeName,
  });

  const resetCam = () => {
    const node = getPlotNode('surface');
    if (node && window.Plotly) {
      window.Plotly.relayout(node, {
        'scene.camera': {
          eye: { x: 1.6, y: 1.6, z: 0.9 },
          up:  { x: 0, y: 0, z: 1 },
          center: { x: 0, y: 0, z: 0 },
        },
      });
    }
  };
  const setView = (eye) => {
    const node = getPlotNode('surface');
    if (node && window.Plotly) {
      window.Plotly.relayout(node, { 'scene.camera.eye': eye, 'scene.camera.up': { x:0, y:0, z:1 } });
    }
  };
  const toggleProj = () => {
    const next = proj === 'perspective' ? 'orthographic' : 'perspective';
    setProj(next);
    const node = getPlotNode('surface');
    if (node && window.Plotly) {
      window.Plotly.relayout(node, { 'scene.camera.projection.type': next });
    }
  };

  return (
    <div style={{ position: 'relative', width, height }}>
      <PlotlyChart kind="surface" width={width} height={height} build={build}
        deps={[width, height, mode, highlightN, activeName, demands && demands.length]}
        filename="gensec-surface3d" />
      <PlotToolbar>
        <ToolBtn title="Stacked N-slice contours" active={mode==='slices'} onClick={()=>setMode('slices')}>Slices</ToolBtn>
        <ToolBtn title="Continuous mesh surface"  active={mode==='mesh'}   onClick={()=>setMode('mesh')}>Mesh</ToolBtn>
        <ToolBtn title="Wireframe rings only"     active={mode==='wireframe'} onClick={()=>setMode('wireframe')}>Wire</ToolBtn>
        <span style={{ width: 1, height: 14, background: 'var(--rule)' }}/>
        <ToolBtn title="View along +X (Mx axis)"  onClick={()=>setView({x:2.4,y:0,z:0})}>X</ToolBtn>
        <ToolBtn title="View along +Y (My axis)"  onClick={()=>setView({x:0,y:2.4,z:0})}>Y</ToolBtn>
        <ToolBtn title="Top-down (N axis)"         onClick={()=>setView({x:0,y:0,z:2.4})}>Z</ToolBtn>
        <ToolBtn title="Isometric"                  onClick={()=>setView({x:1.6,y:1.6,z:0.9})}>Iso</ToolBtn>
        <span style={{ width: 1, height: 14, background: 'var(--rule)' }}/>
        <ToolBtn title="Toggle perspective / orthographic" active={proj==='orthographic'} onClick={toggleProj}>
          {proj === 'orthographic' ? 'Ortho' : 'Persp'}
        </ToolBtn>
        <ToolBtn title="Reset camera" onClick={resetCam}>⟲</ToolBtn>
      </PlotToolbar>
    </div>
  );
}

// ---- M-χ ------------------------------------------------------
function MchiPlot({ width, height, activeN }) {
  const build = () => window.GS_PLOTLY.buildMchi({ activeN });
  const reset = () => {
    const node = getPlotNode('mchi');
    if (node && window.Plotly) window.Plotly.relayout(node, { 'xaxis.autorange': true, 'yaxis.autorange': true });
  };
  return (
    <div style={{ position: 'relative', width, height }}>
      <PlotlyChart kind="mchi" width={width} height={height} build={build}
        deps={[width, height, activeN]}
        filename="gensec-mchi" />
      <PlotToolbar>
        <ToolBtn title="Reset zoom & pan" onClick={reset}>⟲ Reset</ToolBtn>
      </PlotToolbar>
    </div>
  );
}

// ---- Polar ductility -----------------------------------------
function PolarDuctility({ width, height }) {
  const build = () => window.GS_PLOTLY.buildPolar();
  const reset = () => {
    const node = getPlotNode('polar');
    if (node && window.Plotly) window.Plotly.relayout(node, { 'polar.radialaxis.autorange': true });
  };
  return (
    <div style={{ position: 'relative', width, height }}>
      <PlotlyChart kind="polar" width={width} height={height} build={build}
        deps={[width, height]}
        filename="gensec-polar" />
      <PlotToolbar>
        <ToolBtn title="Reset zoom" onClick={reset}>⟲ Reset</ToolBtn>
      </PlotToolbar>
    </div>
  );
}

// ---- Section preview (SVG with pan+zoom) ---------------------
function SectionPreview({ width, height }) {
  const { B, H, rebars, n_fibers_x, n_fibers_y } = window.GS_DATA.SECTION;
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const dragRef = useRef(null);

  const reset = () => setView({ k: 1, tx: 0, ty: 0 });

  const onWheel = (e) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    setView(v => {
      const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
      const k2 = Math.max(0.25, Math.min(20, v.k * factor));
      const tx2 = mx - (mx - v.tx) * (k2 / v.k);
      const ty2 = my - (my - v.ty) * (k2 / v.k);
      return { k: k2, tx: tx2, ty: ty2 };
    });
  };
  const onMouseDown = (e) => {
    dragRef.current = { x0: e.clientX, y0: e.clientY, tx0: view.tx, ty0: view.ty };
  };
  const onMouseMove = (e) => {
    if (!dragRef.current) return;
    const d = dragRef.current;
    setView(v => ({ ...v, tx: d.tx0 + (e.clientX - d.x0), ty: d.ty0 + (e.clientY - d.y0) }));
  };
  const onMouseUp = () => { dragRef.current = null; };

  if (!B || !H) {
    return <div style={{ width, height, display: 'grid', placeItems: 'center', color: 'var(--ink-3)' }}>—</div>;
  }

  const pad = 32;
  const sc = Math.min((width - pad*2) / B, (height - pad*2) / H);
  const w = B * sc, h = H * sc;
  const x0 = (width - w) / 2, y0 = (height - h) / 2;
  const dx = w / Math.max(1, n_fibers_x), dy = h / Math.max(1, n_fibers_y);
  const lines = [];
  for (let i = 1; i < n_fibers_x; i++) lines.push(<line key={"vx"+i} x1={x0+i*dx} x2={x0+i*dx} y1={y0} y2={y0+h} stroke="var(--rule)" strokeWidth={0.5/view.k}/>);
  for (let j = 1; j < n_fibers_y; j++) lines.push(<line key={"vy"+j} x1={x0} x2={x0+w} y1={y0+j*dy} y2={y0+j*dy} stroke="var(--rule)" strokeWidth={0.5/view.k}/>);

  return (
    <div style={{ position: 'relative', width, height, overflow: 'hidden', cursor: dragRef.current ? 'grabbing' : 'grab' }}
         onWheel={onWheel} onMouseDown={onMouseDown} onMouseMove={onMouseMove}
         onMouseUp={onMouseUp} onMouseLeave={onMouseUp}>
      <svg width={width} height={height} style={{ display: 'block' }}>
        <g transform={`translate(${view.tx} ${view.ty}) scale(${view.k})`}>
          <rect x={x0} y={y0} width={w} height={h}
                fill="oklch(88% 0.015 85 / 0.5)" stroke="var(--ink-2)" strokeWidth={1.2/view.k} />
          {lines}
          {(rebars || []).map((r, i) => {
            const cx = x0 + (r.x / B) * w;
            const cy = y0 + h - (r.y / H) * h;
            const rad = (r.diameter / 2) * sc * 1.6;
            return (
              <g key={i}>
                <circle cx={cx} cy={cy} r={rad} fill="var(--ink)" />
                <circle cx={cx} cy={cy} r={rad*0.5} fill="var(--paper)" opacity="0.15" />
              </g>
            );
          })}
        </g>
        <g fontFamily="var(--ff-mono)" fontSize="10" fill="var(--ink-3)">
          <text x={width/2} y={height - 10} textAnchor="middle">B = {B} mm   ·   H = {H} mm   ·   drag = pan, wheel = zoom</text>
        </g>
        <g fontFamily="var(--ff-serif)" fontStyle="italic" fontSize="11" fill="var(--ink-2)">
          <text x={width - 14} y={20} textAnchor="end">fiber mesh {n_fibers_x} × {n_fibers_y}</text>
        </g>
      </svg>
      <PlotToolbar>
        <ToolBtn title="Zoom in"  onClick={()=>setView(v=>({...v, k: Math.min(20, v.k*1.2)}))}>＋</ToolBtn>
        <ToolBtn title="Zoom out" onClick={()=>setView(v=>({...v, k: Math.max(0.25, v.k/1.2)}))}>－</ToolBtn>
        <ToolBtn title="Reset" onClick={reset}>⟲</ToolBtn>
      </PlotToolbar>
    </div>
  );
}

// ---- Toolbar export helpers (PNG / CSV / JSON) ---------------
async function exportCurrentPlot(kind, format) {
  const node = getPlotNode(kind);
  const D = window.GS_DATA;

  if (format === 'png') {
    if (!node || !window.Plotly) return;
    await window.Plotly.downloadImage(node, {
      format: 'png', scale: 2,
      filename: `gensec-${kind}`,
      width: Math.max(900, node.clientWidth),
      height: Math.max(600, node.clientHeight),
    });
    return;
  }

  // ---- Build a plain payload for CSV/JSON from GS_DATA ----
  let payload = null;
  if (kind === 'mxmy') {
    const N = window.__GS_LAST_N || 0;
    const pts = D.makeMxMyContour ? D.makeMxMyContour(N) : [];
    payload = { kind: 'Mx-My contour', N_kN: N, columns: ['Mx_kNm', 'My_kNm'], rows: pts };
  } else if (kind === 'nm') {
    const pts = D.makeNM ? D.makeNM() : [];
    payload = { kind: 'N-M envelope', columns: ['N_kN', 'Mx_kNm'], rows: pts };
  } else if (kind === 'surface') {
    const slices = D.makeSurfaceSlices ? D.makeSurfaceSlices() : [];
    payload = {
      kind: '3D resistance surface',
      slices: slices.map(s => ({ N_kN: s.N, points: s.contour })),
    };
  } else if (kind === 'mchi') {
    const series = D.makeMchi ? D.makeMchi() : [];
    payload = {
      kind: 'M-chi curves',
      series: series.map(s => ({ N_kN: s.N, columns: ['chi_per_mm', 'M_kNm'], rows: s.pts })),
    };
  } else if (kind === 'polar') {
    payload = { kind: 'Polar ductility (synthetic)', note: 'Backend payload not yet wired.' };
  } else if (kind === 'section') {
    payload = { kind: 'Section', section: D.SECTION };
  }

  if (!payload) return;

  if (format === 'json') {
    download(`gensec-${kind}.json`, JSON.stringify(payload, null, 2), 'application/json');
  } else if (format === 'csv') {
    download(`gensec-${kind}.csv`, toCSV(payload), 'text/csv');
  }
}

function toCSV(p) {
  const lines = [];
  if (p.rows && p.columns) {
    lines.push(p.columns.join(','));
    for (const r of p.rows) lines.push(r.map(fmtCSV).join(','));
    return lines.join('\n');
  }
  if (p.slices) {
    lines.push('N_kN,Mx_kNm,My_kNm');
    for (const s of p.slices) for (const pt of s.points) lines.push([s.N_kN, pt[0], pt[1]].map(fmtCSV).join(','));
    return lines.join('\n');
  }
  if (p.series) {
    lines.push('N_kN,chi_per_mm,M_kNm');
    for (const s of p.series) for (const r of s.rows) lines.push([s.N_kN, r[0], r[1]].map(fmtCSV).join(','));
    return lines.join('\n');
  }
  // Fallback: dump JSON inside a single CSV cell.
  lines.push('payload');
  lines.push('"' + JSON.stringify(p).replace(/"/g, '""') + '"');
  return lines.join('\n');
}
function fmtCSV(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return Number.isFinite(v) ? v.toString() : '';
  return String(v);
}
function download(name, text, mime) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
}

Object.assign(window, {
  MxMyPlot, NMPlot, Surface3D, MchiPlot, SectionPreview, PolarDuctility,
  exportCurrentPlot,
});
