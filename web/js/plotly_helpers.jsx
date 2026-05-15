/* ==========================================================
 * GenSec GUI — Plotly helpers
 *
 * Pure functions that turn GS_DATA into Plotly trace + layout
 * objects.  Kept here so plots.jsx stays focused on React glue.
 *
 * Theme: every function reads the live CSS variables off
 * :root so dark mode and accent-hue tweaks propagate without
 * a re-mount.  Call refreshTheme() before building traces.
 * ========================================================== */

(function () {
  "use strict";

  // ---- CSS-variable theme cache --------------------------------
  let THEME = null;
  function refreshTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    THEME = {
      paper:    v("--paper")    || "#fafaf7",
      paper2:   v("--paper-2")  || "#f3f3ee",
      ink:      v("--ink")      || "#222",
      ink2:     v("--ink-2")    || "#555",
      ink3:     v("--ink-3")    || "#888",
      rule:     v("--rule")     || "#ddd",
      rule2:    v("--rule-2")   || "#bbb",
      accent:   v("--accent")   || "#4a6fa8",
      accentSoft: v("--accent-soft") || "#cdd9ec",
      fail:     v("--fail")     || "#c33",
      ok:       v("--ok")       || "#3a8",
      s1: v("--series-1") || "#4a6fa8",
      s2: v("--series-2") || "#3a8",
      s3: v("--series-3") || "#c93",
      s4: v("--series-4") || "#c33",
      s5: v("--series-5") || "#869",
    };
    return THEME;
  }
  function theme() { return THEME || refreshTheme(); }

  // ---- Common layout fragments --------------------------------
  function baseLayout(opts) {
    const T = theme();
    return Object.assign({
      paper_bgcolor: T.paper,
      plot_bgcolor:  T.paper,
      font: {
        family: 'Inter, system-ui, sans-serif',
        size: 11,
        color: T.ink2,
      },
      margin: { l: 60, r: 24, t: 28, b: 48 },
      hoverlabel: {
        bgcolor: T.paper2,
        bordercolor: T.rule2,
        font: { family: 'JetBrains Mono, monospace', size: 11, color: T.ink },
      },
      showlegend: false,
      modebar: {
        bgcolor: 'rgba(0,0,0,0)',
        color: T.ink3,
        activecolor: T.accent,
        orientation: 'h',
      },
    }, opts || {});
  }

  function axis2d(label, opts) {
    const T = theme();
    return Object.assign({
      title: { text: label, font: { family: 'Source Serif 4, serif', size: 12, style: 'italic', color: T.ink2 } },
      gridcolor: T.rule,
      zerolinecolor: T.rule2,
      linecolor: T.rule2,
      tickcolor: T.rule2,
      tickfont: { family: 'JetBrains Mono, monospace', size: 10, color: T.ink3 },
      gridwidth: 1,
      zerolinewidth: 1,
      showspikes: false,
    }, opts || {});
  }

  function axis3d(label) {
    const T = theme();
    return {
      title: { text: label, font: { family: 'Source Serif 4, serif', size: 12, color: T.ink2 } },
      backgroundcolor: T.paper,
      gridcolor: T.rule,
      zerolinecolor: T.rule2,
      linecolor: T.rule2,
      tickcolor: T.rule2,
      tickfont: { family: 'JetBrains Mono, monospace', size: 10, color: T.ink3 },
      showbackground: true,
    };
  }

  // ---- Mx-My contour plot -------------------------------------
  function buildMxMy({ N_kN, demands, activeName }) {
    const T = theme();
    const D = window.GS_DATA;
    const contour = D.makeMxMyContour ? D.makeMxMyContour(N_kN, 144) : [];
    const traces = [];

    if (contour.length) {
      traces.push({
        x: contour.map(p => p[0]),
        y: contour.map(p => p[1]),
        mode: 'lines',
        type: 'scatter',
        fill: 'toself',
        fillcolor: hexA(T.accentSoft, 0.35),
        line: { color: T.accent, width: 1.8 },
        name: 'Resistance domain',
        hovertemplate: 'Mx=%{x:.0f}<br>My=%{y:.0f}<extra></extra>',
      });
    }

    const dem = demands || [];
    if (dem.length) {
      const others = dem.filter(d => d.name !== activeName);
      const active = dem.find(d => d.name === activeName);
      if (others.length) {
        traces.push({
          x: others.map(d => d.Mx),
          y: others.map(d => d.My),
          text: others.map(d => d.name),
          mode: 'markers+text',
          type: 'scatter',
          marker: { size: 7, color: T.ink, line: { color: T.paper, width: 1.5 } },
          textposition: 'top right',
          textfont: { family: 'JetBrains Mono, monospace', size: 10, color: T.ink2 },
          hovertemplate: '<b>%{text}</b><br>Mx=%{x:.0f}<br>My=%{y:.0f}<extra></extra>',
          name: 'Demands',
        });
      }
      if (active) {
        traces.push({
          x: [0, active.Mx], y: [0, active.My],
          mode: 'lines', type: 'scatter',
          line: { color: T.fail, width: 1.6, dash: 'dash' },
          hoverinfo: 'skip', showlegend: false,
        });
        traces.push({
          x: [active.Mx], y: [active.My],
          text: [active.name],
          mode: 'markers+text', type: 'scatter',
          marker: { size: 11, color: T.fail, line: { color: T.paper, width: 2 } },
          textposition: 'top right',
          textfont: { family: 'JetBrains Mono, monospace', size: 11, color: T.fail, weight: 600 },
          hovertemplate: '<b>%{text}</b><br>Mx=%{x:.0f}<br>My=%{y:.0f}<extra></extra>',
        });
      }
    }

    const layout = baseLayout({
      xaxis: axis2d('Mx  [kN·m]', { zeroline: true, scaleanchor: 'y', scaleratio: 1 }),
      yaxis: axis2d('My  [kN·m]', { zeroline: true }),
    });

    return { traces, layout };
  }

  // ---- N-M plot -----------------------------------------------
  function buildNM({ demands, activeName }) {
    const T = theme();
    const D = window.GS_DATA;
    const pts = D.makeNM ? D.makeNM() : [];
    const traces = [];

    if (pts.length) {
      traces.push({
        x: pts.map(p => p[0]),
        y: pts.map(p => p[1]),
        mode: 'lines',
        type: 'scatter',
        fill: 'toself',
        fillcolor: hexA(T.accentSoft, 0.35),
        line: { color: T.accent, width: 1.8 },
        hovertemplate: 'N=%{x:.0f}<br>Mx=%{y:.1f}<extra></extra>',
      });
    }

    const dem = demands || [];
    if (dem.length) {
      const others = dem.filter(d => d.name !== activeName);
      const active = dem.find(d => d.name === activeName);
      if (others.length) {
        traces.push({
          x: others.map(d => d.N),
          y: others.map(d => d.Mx),
          text: others.map(d => d.name),
          mode: 'markers+text', type: 'scatter',
          marker: { size: 7, color: T.ink, line: { color: T.paper, width: 1.5 } },
          textposition: 'top right',
          textfont: { family: 'JetBrains Mono, monospace', size: 10, color: T.ink2 },
          hovertemplate: '<b>%{text}</b><br>N=%{x:.0f}<br>Mx=%{y:.1f}<extra></extra>',
        });
      }
      if (active) {
        traces.push({
          x: [active.N], y: [active.Mx],
          text: [active.name],
          mode: 'markers+text', type: 'scatter',
          marker: { size: 11, color: T.fail, line: { color: T.paper, width: 2 } },
          textposition: 'top right',
          textfont: { family: 'JetBrains Mono, monospace', size: 11, color: T.fail, weight: 600 },
          hovertemplate: '<b>%{text}</b><br>N=%{x:.0f}<br>Mx=%{y:.1f}<extra></extra>',
        });
      }
    }

    const layout = baseLayout({
      xaxis: axis2d('N  [kN]', { zeroline: true }),
      yaxis: axis2d('Mx  [kN·m]', { zeroline: true }),
    });

    return { traces, layout };
  }

  // ---- 3D resistance surface ----------------------------------
  // mode: 'slices' | 'mesh' | 'wireframe'
  function buildSurface3D({ mode, highlightN, demands, activeName }) {
    const T = theme();
    const D = window.GS_DATA;
    const slices = D.makeSurfaceSlices ? D.makeSurfaceSlices() : [];
    const traces = [];

    if (!slices.length) {
      return { traces, layout: build3DLayout() };
    }

    const Nvals = slices.map(s => s.N);
    const Nmin = Math.min(...Nvals), Nmax = Math.max(...Nvals);
    const colorAt = (N) => {
      const t = (N - Nmin) / (Nmax - Nmin || 1);
      // gradient through accent ramp
      return `oklch(${(35 + t * 45).toFixed(0)}% 0.12 225)`;
    };

    if (mode === 'slices' || mode === 'wireframe') {
      slices.forEach((s) => {
        const isHighlighted = highlightN !== null && highlightN !== undefined &&
          Math.abs(s.N - highlightN) < (Nmax - Nmin) / (slices.length * 2 + 1);
        traces.push({
          type: 'scatter3d',
          mode: 'lines',
          x: s.contour.map(p => p[0]),
          y: s.contour.map(p => p[1]),
          z: s.contour.map(() => s.N),
          line: {
            color: isHighlighted ? T.fail : colorAt(s.N),
            width: isHighlighted ? 6 : 3,
          },
          hovertemplate: `Mx=%{x:.0f}<br>My=%{y:.0f}<br>N=${s.N.toFixed(0)} kN<extra></extra>`,
          showlegend: false,
        });
        if (mode === 'slices' && !isHighlighted) {
          // faint filled poly at this N
          traces.push({
            type: 'mesh3d',
            x: s.contour.map(p => p[0]),
            y: s.contour.map(p => p[1]),
            z: s.contour.map(() => s.N),
            color: T.accent,
            opacity: 0.08,
            hoverinfo: 'skip',
            showlegend: false,
          });
        }
      });
    } else if (mode === 'mesh') {
      // Continuous surface: collect ALL contour points across all N levels
      // and let mesh3d's Delaunay triangulation handle it.  Works best when
      // the contour densities are similar across slices.
      const X = [], Y = [], Z = [], I = [], J = [], K = [];
      const ringStart = [];
      slices.forEach((s) => {
        ringStart.push(X.length);
        s.contour.forEach((p) => { X.push(p[0]); Y.push(p[1]); Z.push(s.N); });
      });
      // Stitch adjacent rings: triangulate as quad strips.
      for (let r = 0; r < slices.length - 1; r++) {
        const a = ringStart[r];
        const b = ringStart[r + 1];
        const na = (slices[r].contour.length);
        const nb = (slices[r + 1].contour.length);
        const n = Math.min(na, nb) - 1; // skip closing duplicate
        for (let i = 0; i < n; i++) {
          const i2 = (i + 1) % n;
          // two triangles per quad
          I.push(a + i);  J.push(a + i2);  K.push(b + i);
          I.push(b + i);  J.push(a + i2);  K.push(b + i2);
        }
      }
      traces.push({
        type: 'mesh3d',
        x: X, y: Y, z: Z,
        i: I, j: J, k: K,
        color: T.accent,
        opacity: 0.55,
        flatshading: false,
        lighting: { ambient: 0.7, diffuse: 0.6, roughness: 0.9 },
        hovertemplate: 'Mx=%{x:.0f}<br>My=%{y:.0f}<br>N=%{z:.0f}<extra></extra>',
        showlegend: false,
      });
      // Highlighted N ring on top
      if (highlightN !== null && highlightN !== undefined) {
        let best = slices[0], bd = Math.abs(highlightN - best.N);
        for (const s of slices) {
          const d = Math.abs(highlightN - s.N);
          if (d < bd) { best = s; bd = d; }
        }
        traces.push({
          type: 'scatter3d', mode: 'lines',
          x: best.contour.map(p => p[0]),
          y: best.contour.map(p => p[1]),
          z: best.contour.map(() => best.N),
          line: { color: T.fail, width: 6 },
          hovertemplate: `N=${best.N.toFixed(0)} kN<extra></extra>`,
          showlegend: false,
        });
      }
    }

    // Demand points in 3D
    if (demands && demands.length) {
      const others = demands.filter(d => d.name !== activeName);
      const active = demands.find(d => d.name === activeName);
      if (others.length) {
        traces.push({
          type: 'scatter3d', mode: 'markers+text',
          x: others.map(d => d.Mx),
          y: others.map(d => d.My),
          z: others.map(d => d.N),
          text: others.map(d => d.name),
          marker: { size: 4, color: T.ink, line: { color: T.paper, width: 1 } },
          textposition: 'top right',
          textfont: { family: 'JetBrains Mono, monospace', size: 10, color: T.ink2 },
          hovertemplate: '<b>%{text}</b><br>Mx=%{x:.0f}<br>My=%{y:.0f}<br>N=%{z:.0f}<extra></extra>',
          showlegend: false,
        });
      }
      if (active) {
        traces.push({
          type: 'scatter3d', mode: 'markers+text',
          x: [active.Mx], y: [active.My], z: [active.N],
          text: [active.name],
          marker: { size: 8, color: T.fail, line: { color: T.paper, width: 2 } },
          textposition: 'top right',
          textfont: { family: 'JetBrains Mono, monospace', size: 11, color: T.fail, weight: 600 },
          hovertemplate: '<b>%{text}</b><br>Mx=%{x:.0f}<br>My=%{y:.0f}<br>N=%{z:.0f}<extra></extra>',
          showlegend: false,
        });
      }
    }

    return { traces, layout: build3DLayout() };
  }

  function build3DLayout() {
    const T = theme();
    return Object.assign(baseLayout(), {
      margin: { l: 0, r: 0, t: 8, b: 0 },
      scene: {
        xaxis: axis3d('Mx  [kN·m]'),
        yaxis: axis3d('My  [kN·m]'),
        zaxis: axis3d('N  [kN]'),
        camera: {
          eye: { x: 1.6, y: 1.6, z: 0.9 },
          up: { x: 0, y: 0, z: 1 },
        },
        aspectmode: 'cube',
        bgcolor: T.paper,
      },
    });
  }

  // ---- M-χ plot ------------------------------------------------
  function buildMchi({ activeN }) {
    const T = theme();
    const D = window.GS_DATA;
    const series = D.makeMchi ? D.makeMchi() : [];
    const palette = [T.s1, T.s2, T.s3, T.s4, T.s5];

    const traces = series.map((s, i) => {
      const isActive = activeN !== null && activeN !== undefined &&
        Math.abs(s.N - activeN) < 25;
      return {
        x: s.pts.map(p => p[0] * 1e6),
        y: s.pts.map(p => p[1]),
        mode: 'lines',
        type: 'scatter',
        line: {
          color: palette[i % palette.length],
          width: isActive ? 2.6 : 1.4,
        },
        opacity: isActive ? 1 : 0.7,
        name: `N = ${s.N.toFixed(0)} kN`,
        hovertemplate: `<b>N=${s.N.toFixed(0)}</b><br>χ=%{x:.1f}×10⁻⁶<br>M=%{y:.1f}<extra></extra>`,
      };
    });

    const layout = baseLayout({
      showlegend: true,
      legend: {
        x: 0.99, y: 0.02, xanchor: 'right', yanchor: 'bottom',
        bgcolor: hexA(T.paper, 0.85),
        bordercolor: T.rule,
        borderwidth: 1,
        font: { family: 'JetBrains Mono, monospace', size: 10 },
      },
      xaxis: axis2d('χ  ×10⁻⁶  [1/mm]'),
      yaxis: axis2d('M  [kN·m]'),
    });

    return { traces, layout };
  }

  // ---- Polar ductility ----------------------------------------
  function buildPolar() {
    const T = theme();
    // Synthetic ductility rose – the backend does not expose this yet,
    // so we keep the analytic shape from the original mock until it does.
    const n = 72;
    const r = [], theta = [];
    for (let i = 0; i <= n; i++) {
      const t = (i / n) * Math.PI * 2;
      r.push(0.55 + 0.32 * Math.sin(2 * t + 0.3) + 0.08 * Math.cos(4 * t));
      theta.push(t * 180 / Math.PI);
    }

    const traces = [{
      type: 'scatterpolar',
      r, theta,
      mode: 'lines',
      fill: 'toself',
      fillcolor: hexA(T.accentSoft, 0.5),
      line: { color: T.accent, width: 1.8 },
      hovertemplate: 'θ=%{theta:.0f}°<br>μ=%{r:.2f}<extra></extra>',
    }];

    const layout = baseLayout({
      polar: {
        bgcolor: T.paper,
        radialaxis: {
          gridcolor: T.rule,
          linecolor: T.rule2,
          tickfont: { family: 'JetBrains Mono, monospace', size: 10, color: T.ink3 },
          angle: 90,
        },
        angularaxis: {
          gridcolor: T.rule,
          linecolor: T.rule2,
          tickfont: { family: 'JetBrains Mono, monospace', size: 10, color: T.ink3 },
          direction: 'counterclockwise',
        },
      },
    });

    return { traces, layout };
  }

  // ---- helpers -------------------------------------------------
  // Convert any color string to rgba with given alpha (best-effort).
  function hexA(color, a) {
    if (!color) return `rgba(0,0,0,${a})`;
    color = color.trim();
    if (color.startsWith('#')) {
      const h = color.slice(1);
      const n = h.length === 3
        ? h.split('').map(c => parseInt(c + c, 16))
        : [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
      return `rgba(${n[0]},${n[1]},${n[2]},${a})`;
    }
    if (color.startsWith('rgb')) {
      // rgb(...) or rgba(...)  →  swap alpha
      return color.replace(/rgba?\(([^)]+)\)/, (_, body) => {
        const parts = body.split(',').map(s => s.trim());
        return `rgba(${parts[0]},${parts[1]},${parts[2]},${a})`;
      });
    }
    // oklch / hsl / named — let the browser deal via a canvas probe.
    try {
      const ctx = document.createElement('canvas').getContext('2d');
      ctx.fillStyle = color;
      const cssColor = ctx.fillStyle; // normalized
      if (cssColor.startsWith('#')) return hexA(cssColor, a);
      if (cssColor.startsWith('rgb')) return hexA(cssColor, a);
    } catch (_) {}
    return color;
  }

  // ---- Default Plotly config -----------------------------------
  function commonConfig({ filename } = {}) {
    return {
      responsive: true,
      displaylogo: false,
      toImageButtonOptions: {
        format: 'png',
        filename: filename || 'gensec-plot',
        scale: 2,
      },
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      doubleClick: 'reset+autosize',
    };
  }

  // ---- Public surface -----------------------------------------
  window.GS_PLOTLY = {
    refreshTheme,
    buildMxMy, buildNM, buildSurface3D, buildMchi, buildPolar,
    commonConfig,
  };
})();
