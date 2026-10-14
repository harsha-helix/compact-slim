// Minimal canvas line plot (no external libraries, so the UI works offline).
"use strict";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function niceTicks(lo, hi, count) {
  if (!(hi > lo)) { hi = lo + 1; }
  const span = hi - lo;
  const step0 = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

function fmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2).replace(/\.?0+$/, "");
  return v.toPrecision(3);
}

class LinePlot {
  // series: [{key, label, color: '--s1', width, dash}]
  constructor(canvas, opts) {
    this.c = canvas;
    this.opts = Object.assign({ xKey: "step", xLabel: "step", series: [], refs: [], log: false }, opts);
    this.data = [];
    new ResizeObserver(() => this.draw()).observe(canvas);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
  }
  reset() { this.data = []; this.opts.refs = []; this.draw(); }
  add(points) { this.data.push(...points); this.draw(); }
  setRefs(refs) { this.opts.refs = refs.filter(r => r.value !== null && r.value !== undefined && isFinite(r.value)); this.draw(); }
  setLog(on) { this.opts.log = on; this.draw(); }

  draw() {
    const c = this.c, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.font = "11px system-ui, sans-serif";
    const muted = cssVar("--muted"), grid = cssVar("--grid");
    const { xKey, series, refs } = this.opts;
    const log = this.opts.log;
    const d = this.data;
    // On a log axis, values <= 0 (e.g. a perfect partition) are drawn at a floor
    // one decade below the smallest positive value, labelled "0".
    let floor = 1;
    if (log) {
      let mp = Infinity;
      for (const p of d) for (const s of series) { const v = p[s.key]; if (v > 0 && v < mp) mp = v; }
      for (const r of refs) if (r.value > 0 && r.value < mp) mp = r.value;
      floor = Math.pow(10, Math.floor(Math.log10(isFinite(mp) ? mp : 1)) - 1);
    }
    const tr = v => log ? Math.log10(v > 0 ? v : floor) : v;

    // legend
    let lx = 52;
    const legend = series.concat(refs.map(r => ({ label: r.label, color: r.color, dash: [4, 3] })));
    for (const s of legend) {
      g.strokeStyle = cssVar(s.color); g.lineWidth = 2; g.setLineDash(s.dash || []);
      g.beginPath(); g.moveTo(lx, 9); g.lineTo(lx + 16, 9); g.stroke(); g.setLineDash([]);
      g.fillStyle = muted; g.fillText(s.label, lx + 20, 13);
      lx += 30 + g.measureText(s.label).width;
    }

    const L = 52, R = 10, T = 22, B = 26;
    const pw = W - L - R, ph = H - T - B;
    if (!d.length) {
      g.fillStyle = muted; g.fillText("waiting for data…", L + pw / 2 - 45, T + ph / 2);
      return;
    }
    let xmin = d[0][xKey], xmax = d[d.length - 1][xKey];
    if (xmax <= xmin) xmax = xmin + 1;
    let ymin = Infinity, ymax = -Infinity;
    for (const p of d) for (const s of series) {
      const v = p[s.key];
      if (v === null || v === undefined || !isFinite(v) || false) continue;
      const t = tr(v); if (t < ymin) ymin = t; if (t > ymax) ymax = t;
    }
    for (const r of refs) { const t = tr(r.value); ymin = Math.min(ymin, t); ymax = Math.max(ymax, t); }
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    if (ymax === ymin) { ymax += 0.5; ymin -= 0.5; }
    const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad;
    const X = x => L + (x - xmin) / (xmax - xmin) * pw;
    const Y = y => T + (1 - (tr(y) - ymin) / (ymax - ymin)) * ph;

    // grid + axes
    g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = muted;
    const yt = log ? niceTicks(Math.floor(ymin), Math.ceil(ymax), Math.min(6, Math.ceil(ymax) - Math.floor(ymin))).filter(t => Number.isInteger(Math.round(t * 1e6) / 1e6))
                   : niceTicks(ymin, ymax, 5);
    for (const t of yt) {
      const yy = T + (1 - (t - ymin) / (ymax - ymin)) * ph;
      if (yy < T - 1 || yy > T + ph + 1) continue;
      g.beginPath(); g.moveTo(L, yy); g.lineTo(L + pw, yy); g.stroke();
      const lbl = log ? (Math.abs(t - Math.log10(floor)) < 1e-9 ? "0" : fmt(Math.pow(10, t))) : fmt(t);
      g.fillText(lbl, L - 6 - g.measureText(lbl).width, yy + 4);
    }
    for (const t of niceTicks(xmin, xmax, 6)) {
      const xx = X(t); if (xx < L - 1 || xx > L + pw + 1) continue;
      g.beginPath(); g.moveTo(xx, T); g.lineTo(xx, T + ph); g.stroke();
      const lbl = fmt(t); g.fillText(lbl, xx - g.measureText(lbl).width / 2, T + ph + 15);
    }
    g.fillText(this.opts.xLabel, L + pw - g.measureText(this.opts.xLabel).width, H - 1);

    // reference lines
    for (const r of refs) {
      g.strokeStyle = cssVar(r.color); g.setLineDash([5, 4]); g.lineWidth = 1.5;
      const yy = Y(r.value);
      g.beginPath(); g.moveTo(L, yy); g.lineTo(L + pw, yy); g.stroke();
    }
    g.setLineDash([]);

    // series (min/max decimation to ~2 points per pixel)
    g.save(); g.beginPath(); g.rect(L, T, pw, ph); g.clip();
    for (const s of series) {
      g.strokeStyle = cssVar(s.color); g.lineWidth = s.width || 1.5; g.setLineDash(s.dash || []);
      g.beginPath();
      let started = false;
      const per = Math.max(1, Math.floor(d.length / (pw * 2)));
      for (let i = 0; i < d.length; i += per) {
        let lo = null, hi = null;
        for (let j = i; j < Math.min(d.length, i + per); j++) {
          const v = d[j][s.key];
          if (v === null || v === undefined || !isFinite(v) || false) continue;
          if (lo === null || v < lo) lo = v; if (hi === null || v > hi) hi = v;
        }
        if (lo === null) continue;
        const xx = X(d[i][xKey]);
        if (!started) { g.moveTo(xx, Y(lo)); started = true; } else g.lineTo(xx, Y(lo));
        if (hi !== lo) g.lineTo(xx, Y(hi));
      }
      g.stroke();
    }
    g.restore();
  }
}

// Circle-layout graph drawing; after a run, nodes are coloured by partition
// and cut edges are highlighted.
class GraphView {
  constructor(canvas) {
    this.c = canvas; this.graph = null; this.spins = null;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  set(graph) { this.graph = graph; this.spins = null; this.draw(); }
  setSpins(spins) { this.spins = spins; this.draw(); }
  draw() {
    const c = this.c, dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    g.font = "12px system-ui, sans-serif"; g.fillStyle = cssVar("--muted");
    const G = this.graph;
    if (!G) { g.fillText("No graph yet.", 10, 20); return; }
    if (!G.edge_list) { g.fillText(`Graph too large to draw (${G.edges} edges).`, 10, 20); return; }
    const n = G.n, r = Math.min(W, H - 18) / 2 - 12, cx = W / 2, cy = (H - 18) / 2;
    const pos = i => [cx + r * Math.cos(2 * Math.PI * i / n - Math.PI / 2), cy + r * Math.sin(2 * Math.PI * i / n - Math.PI / 2)];
    const s = this.spins;
    const cutCol = cssVar("--s4"), dim = cssVar("--grid"), plain = cssVar("--muted");
    const alpha = Math.max(0.12, Math.min(0.9, 60 / G.edge_list.length));
    for (let pass = 0; pass < 2; pass++) {
      G.edge_list.forEach(([a, b], k) => {
        const isCut = s ? s[a] !== s[b] : null;
        if (s && (pass === 0) === isCut) return;      // draw uncut first, cut on top
        if (!s && pass === 1) return;
        const [x1, y1] = pos(a), [x2, y2] = pos(b);
        g.strokeStyle = s ? (isCut ? cutCol : dim) : plain;
        g.globalAlpha = s ? (isCut ? Math.max(alpha, 0.5) : 0.6) : alpha;
        g.lineWidth = G.edge_weights[k] < 0 ? 1 : 1.2;
        g.setLineDash(G.edge_weights[k] < 0 ? [3, 3] : []);
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
      });
    }
    g.globalAlpha = 1; g.setLineDash([]);
    const nr = Math.max(2, Math.min(7, 300 / n));
    for (let i = 0; i < n; i++) {
      const [x, y] = pos(i);
      g.fillStyle = s ? (s[i] > 0 ? cssVar("--s1") : cssVar("--s2")) : cssVar("--text");
      g.beginPath(); g.arc(x, y, nr, 0, 2 * Math.PI); g.fill();
    }
    if (s) { g.fillStyle = cssVar("--muted"); g.fillText("cut edges highlighted; dashed = negative weight", 8, H - 6); }
  }
}
