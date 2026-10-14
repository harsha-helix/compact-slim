"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function toast(msg, ms = 4000) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok) throw new Error((body && body.error) || `${r.status} ${r.statusText}`);
  return body;
}
const postJSON = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

// ---------------------------------------------------------------- tabs
$$(".tab").forEach(b => b.addEventListener("click", () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === b));
  $$(".tabpane").forEach(p => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  try { localStorage.setItem("tab", b.dataset.tab); } catch (e) {}
}));
try { const t = localStorage.getItem("tab"); if (t) $(`.tab[data-tab="${t}"]`)?.click(); } catch (e) {}

// ---------------------------------------------------------------- run option panels
function segValue(fs) { return $(".seg button.on", fs).dataset.value; }
function updateModeVisibility(fs) {
  const hw = segValue(fs) === "hardware";
  $$(".hw-only", fs).forEach(e => e.classList.toggle("hidden", !hw));
  $$(".sim-only", fs).forEach(e => e.classList.toggle("hidden", hw));
}
$$(".run-opts").forEach(fs => {
  $$(".seg button", fs).forEach(b => b.addEventListener("click", () => {
    $$(".seg button", fs).forEach(x => x.classList.toggle("on", x === b));
    updateModeVisibility(fs);
  }));
  updateModeVisibility(fs);
});
function collectOpts(fs) {
  const o = { mode: segValue(fs) };
  $$("[name]", fs).forEach(el => {
    if (el.type === "checkbox") o[el.name] = el.checked;
    else if (el.value !== "") o[el.name] = el.type === "number" ? Number(el.value) : el.value;
  });
  return o;
}

// ---------------------------------------------------------------- hardware status
let lastStatus = null;
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    lastStatus = s;
    const b = [];
    b.push(s.hardware.heds_available ? `<span class="badge ok">HOLOEYE SDK found</span>`
                                     : `<span class="badge warn">HOLOEYE SDK not found — simulation only</span>`);
    const ports = s.hardware.serial_ports;
    b.push(`<span class="badge ${ports.length ? "ok" : ""}">${ports.length} serial port${ports.length === 1 ? "" : "s"}</span>`);
    if (s.projecting) b.push(`<span class="badge ok">SLM: hologram shown</span>`);
    if (s.job && (s.job.status === "running" || s.job.status === "starting")) b.push(`<span class="badge warn">running: ${s.job.kind}</span>`);
    $("#hw-badges").innerHTML = b.join("");
    $("#ports").innerHTML = ports.map(p => `<option value="${p.device}">${p.description}</option>`).join("");
    return s;
  } catch (e) { $("#hw-badges").innerHTML = `<span class="badge warn">server unreachable</span>`; }
}
setInterval(refreshStatus, 4000);

// ---------------------------------------------------------------- job polling
class JobWatcher {
  constructor({ onPoints, onLogs, onProgress, onRefs, onDone }) {
    Object.assign(this, { onPoints, onLogs, onProgress, onRefs, onDone });
    this.id = null;
  }
  start(id) {
    this.id = id; this.next = 0; this.logNext = 0; this.refsSeen = false;
    clearTimeout(this.timer); this.tick();
  }
  async tick() {
    const id = this.id;
    let j;
    try { j = await api(`/api/job/${id}?since=${this.next}&log_since=${this.logNext}`); }
    catch (e) { this.timer = setTimeout(() => this.tick(), 1500); return; }
    if (id !== this.id) return;
    this.next = j.next; this.logNext = j.log_next;
    if (j.points.length) this.onPoints(j.points);
    if (j.logs.length) this.onLogs(j.logs);
    this.onProgress(j);
    if (j.refs && !this.refsSeen && this.onRefs) { this.refsSeen = true; this.onRefs(j.refs); }
    if (["done", "stopped", "error"].includes(j.status)) { this.onDone(j); refreshStatus(); return; }
    this.timer = setTimeout(() => this.tick(), 400);
  }
}

function appendLog(pre, lines) {
  pre.textContent += lines.join("\n") + "\n";
  pre.scrollTop = pre.scrollHeight;
}
function fmtTime(s) {
  if (!isFinite(s)) return "–";
  if (s < 60) return `${s.toFixed(0)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${(s / 3600).toFixed(1)} h`;
}

// ================================================================ ANNEALING TABS
function setupAnnealTab(cfg) {
  const fs = $(`.run-opts[data-problem="${cfg.problem}"]`);
  const runBtn = $(".run", fs), stopBtn = $(".stop", fs), eta = $(".eta", fs);
  const pane = $(`#tab-${cfg.problem}`);
  const progress = $(".progress", pane);
  const logPre = $(cfg.logSel);
  const energyPlot = new LinePlot($(cfg.energySel), {
    series: [{ key: "E", label: "current", color: "--s1", width: 1.2 },
             { key: "best_E", label: "best measured", color: "--s3", width: 2 }] });
  const metricPlot = new LinePlot($(cfg.metricSel), {
    series: [{ key: "m", label: "current", color: "--s1", width: 1.2 },
             { key: "best_m", label: "best visited", color: "--s3", width: 2 }],
    log: cfg.metricLog ? cfg.metricLog() : false });
  if (cfg.logToggle) $(cfg.logToggle).addEventListener("change", e => metricPlot.setLog(e.target.checked));

  const watcher = new JobWatcher({
    onPoints: pts => { energyPlot.add(pts); metricPlot.add(pts); },
    onLogs: lines => appendLog(logPre, lines),
    onRefs: refs => metricPlot.setRefs(cfg.refLines(refs)),
    onProgress: j => {
      if (!j.progress) { progress.textContent = j.status; return; }
      const { done, total } = j.progress;
      const rate = done / Math.max(1e-6, j.elapsed);
      const remain = done > 0 ? (total - done) / rate : NaN;
      progress.textContent = `${done} / ${total} steps · ${fmtTime(j.elapsed)} elapsed` + (done > 0 && done < total ? ` · ~${fmtTime(remain)} left` : "");
    },
    onDone: j => {
      runBtn.disabled = false; stopBtn.disabled = true;
      if (j.status === "error") toast("Run failed: " + j.error, 8000);
      if (j.result && j.result.best_measured_spins) cfg.showResult(j.result, j);
      else if (j.status === "error") $(cfg.resultsSel).innerHTML = `<p class="muted">Error: ${escapeHTML(j.error || "")}</p>`;
    },
  });

  runBtn.addEventListener("click", async () => {
    const opts = Object.assign(collectOpts(fs), { problem: cfg.problem });
    try {
      const r = await postJSON("/api/run", opts);
      energyPlot.reset(); metricPlot.reset(); logPre.textContent = "";
      $(cfg.resultsSel).innerHTML = `<p class="muted">Running…</p>`;
      cfg.onStart && cfg.onStart();
      runBtn.disabled = true; stopBtn.disabled = false;
      watcher.start(r.job_id);
      refreshStatus();
    } catch (e) { toast(e.message, 6000); }
  });
  stopBtn.addEventListener("click", () => watcher.id && api(`/api/job/${watcher.id}/stop`, { method: "POST" }));

  // rough duration estimate
  const updateEta = () => {
    const o = collectOpts(fs);
    const n = cfg.size();
    if (!n || !(o.cooling_rate > 0 && o.cooling_rate < 1) || !(o.t_final > 0 && o.t_initial > o.t_final)) { eta.textContent = ""; return; }
    const steps = o.steps_per_temp || n;
    const temps = Math.ceil(Math.log(o.t_final / o.t_initial) / Math.log(o.cooling_rate));
    const total = temps * steps;
    let txt = `${temps} temperatures × ${steps} = ${total} evaluations`;
    if (o.mode === "hardware") {
      const masks = cfg.masksPerEval(n) * total;
      txt += ` · ≈${masks} SLM frames ≈ ${fmtTime(masks / 60)} at 60 Hz`;
    }
    eta.textContent = txt;
  };
  fs.addEventListener("input", updateEta); fs.addEventListener("click", updateEta);
  cfg.updateEta = updateEta;

  return { watcher, runBtn, stopBtn, updateEta };
}

function escapeHTML(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function num(v, digits = 4) { return v === null || v === undefined || !isFinite(v) ? "–" : (Math.abs(v) >= 1000 ? v.toFixed(0) : Number(v.toPrecision(digits)).toString()); }

function runInfo(r, j) {
  return `<p class="muted" style="margin-top:8px">${r.mode} · ${r.evaluations} evaluations in ${fmtTime(r.elapsed_s)}`
    + (r.evals_per_s ? ` (${r.evals_per_s.toFixed(1)}/s)` : "")
    + ` · acceptance ${(100 * r.acceptance_rate).toFixed(0)}% · T scale ${num(r.temperature_scale, 3)}`
    + (r.saved_to ? ` · saved to <code>${escapeHTML(r.saved_to)}</code>` : "")
    + ` · <a href="/api/job/${j.id}/download">download JSON</a></p>`;
}

// ---------------------------------------------------------------- Max-Cut
const graphView = new GraphView($("#mc-graph"));
let mcGraph = null;

function showGraphSummary(g) {
  mcGraph = g;
  $("#mc-summary").innerHTML = `<b>${escapeHTML(g.name)}</b>: ${g.n} nodes, ${g.edges} edges, total weight ${num(g.total_weight)}`;
  graphView.set(g);
  mc.updateEta();
}
function updateMcKind() {
  const k = $("#mc-kind").value;
  $(".kind-er").classList.toggle("hidden", k !== "erdos_renyi");
  $(".kind-reg").classList.toggle("hidden", k !== "regular");
}
$("#mc-kind").addEventListener("change", updateMcKind); updateMcKind();
$("#mc-weights").addEventListener("change", () => $(".w-int").classList.toggle("hidden", $("#mc-weights").value !== "int"));

$("#mc-generate").addEventListener("click", async () => {
  try {
    showGraphSummary(await postJSON("/api/maxcut/generate", {
      n: +$("#mc-n").value, kind: $("#mc-kind").value, p: +$("#mc-p").value, degree: +$("#mc-degree").value,
      weights: $("#mc-weights").value, wmax: +$("#mc-wmax").value, seed: $("#mc-seed").value,
    }));
  } catch (e) { toast(e.message); }
});
$("#mc-file").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  try { showGraphSummary(await api("/api/maxcut/upload", { method: "POST", body: fd })); }
  catch (err) { toast(err.message); }
  e.target.value = "";
});

const mc = setupAnnealTab({
  problem: "maxcut",
  energySel: "#mc-plot-energy", metricSel: "#mc-plot-metric", resultsSel: "#mc-results", logSel: "#mc-log",
  size: () => mcGraph && mcGraph.n,
  masksPerEval: n => n,   // MaxCutAnnealer streams every eigenmode
  refLines: refs => [
    { value: refs.cpu_sa, label: "CPU SA", color: "--s2" },
    { value: refs.exact, label: "exact optimum", color: "--s4" },
  ],
  onStart: () => graphView.setSpins(null),
  showResult: (r, j) => {
    const refs = r.refs;
    const best = refs.exact ?? Math.max(refs.cpu_sa, r.best_visited_metric);
    const pct = v => best > 0 ? `${(100 * v / best).toFixed(1)}%` : "–";
    const rows = [
      ["Photonic — best measured state", r.best_measured_metric, "photonic"],
      ["Photonic — best state visited (judged on CPU)", r.best_visited_metric, ""],
      [`CPU simulated annealing (${refs.cpu_time_s.toFixed(1)} s incl. exact)`, refs.cpu_sa, ""],
      ["Exact optimum (brute force)", refs.exact, ""],
      ["Random partition (expected)", refs.random_expected, ""],
    ];
    $("#mc-results").innerHTML = `<table class="cmp"><tr><th>Method</th><th class="num">Cut</th><th class="num">vs best</th></tr>`
      + rows.map(([name, v, cls]) => `<tr class="${cls}"><td>${name}</td><td class="num">${v === null || v === undefined ? "<span class='muted'>n too large</span>" : num(v)}</td><td class="num">${v === null || v === undefined ? "" : pct(v)}</td></tr>`).join("")
      + `</table>`
      + `<p style="margin-top:8px">Total edge weight ${num(refs.total_weight)} · best measured energy ${num(r.best_measured_energy)}</p>`
      + runInfo(r, j);
    graphView.setSpins(r.best_measured_spins);
  },
});

// ---------------------------------------------------------------- NPP
let nppSet = null;
function showNppSummary(s) {
  nppSet = s;
  const preview = s.weights.slice(0, 40).map(v => num(v)).join(", ") + (s.n > 40 ? ", …" : "");
  $("#npp-summary").innerHTML = `<b>${s.n} numbers</b>, sum ${num(s.sum)}<br><span class="muted">${preview}</span>`;
  npp.updateEta();
}
$("#npp-generate").addEventListener("click", async () => {
  try {
    showNppSummary(await postJSON("/api/npp/generate", {
      n: +$("#npp-n").value, wmax: +$("#npp-wmax").value, seed: $("#npp-seed").value, custom: $("#npp-custom").value.trim(),
    }));
  } catch (e) { toast(e.message); }
});

const npp = setupAnnealTab({
  problem: "npp",
  energySel: "#npp-plot-energy", metricSel: "#npp-plot-metric", resultsSel: "#npp-results", logSel: "#npp-log-text",
  logToggle: "#npp-log", metricLog: () => $("#npp-log").checked,
  size: () => nppSet && nppSet.n,
  masksPerEval: () => 1,
  refLines: refs => [
    { value: refs.karmarkar_karp, label: "Karmarkar–Karp", color: "--s2" },
    { value: refs.exact, label: "exact optimum", color: "--s4" },
  ],
  onStart: () => { $("#npp-partition").innerHTML = "—"; },
  showResult: (r, j) => {
    const refs = r.refs;
    const rows = [
      ["Photonic — best measured state", r.best_measured_metric, "photonic"],
      ["Photonic — best state visited (judged on CPU)", r.best_visited_metric, ""],
      ["Karmarkar–Karp (differencing)", refs.karmarkar_karp, ""],
      ["CPU simulated annealing", refs.cpu_sa, ""],
      ["Exact optimum (subset-sum DP)", refs.exact, ""],
    ];
    const opt = refs.exact;
    $("#npp-results").innerHTML = `<table class="cmp"><tr><th>Method</th><th class="num">|difference|</th><th></th></tr>`
      + rows.map(([name, v, cls]) => {
        let tag = "";
        if (opt !== null && opt !== undefined && v !== null && v !== undefined) tag = v <= opt + 1e-9 ? `<span class="pill good">optimal</span>` : `<span class="pill bad">+${num(v - opt)}</span>`;
        return `<tr class="${cls}"><td>${name}</td><td class="num">${v === null || v === undefined ? "<span class='muted'>n/a</span>" : num(v)}</td><td>${tag}</td></tr>`;
      }).join("")
      + `</table><p style="margin-top:8px">Sum ${num(refs.sum)} · best measured PD energy ${num(r.best_measured_energy)}</p>`
      + runInfo(r, j);
    const [a, b] = r.best_measured_partition;
    const mx = Math.max(a, b) || 1;
    const w = nppSet ? nppSet.weights : [];
    const s = r.best_measured_spins;
    $("#npp-partition").innerHTML = `<div class="bars">
        <div class="bar" style="width:${(100 * a / mx).toFixed(1)}%">A: ${num(a)}</div>
        <div class="bar b2" style="width:${(100 * b / mx).toFixed(1)}%">B: ${num(b)}</div></div>
      <div class="chips">${s.map((v, i) => `<span class="chip ${v > 0 ? "" : "b2"}">${num(w[i])}</span>`).join("")}</div>`;
  },
});

// ================================================================ HOLOGRAPHY
const holoPlot = new LinePlot($("#holo-plot"), {
  xLabel: "iteration",
  series: [{ key: "efficiency", label: "light in image window", color: "--s3", width: 2 },
           { key: "rmse", label: "amplitude error (RMSE)", color: "--s2", width: 2 }] });
let holoFile = null;

function setHoloFile(f) {
  if (!f || !f.type.startsWith("image/")) { toast("Please choose an image file."); return; }
  holoFile = f;
  const img = $("#holo-thumb"); img.src = URL.createObjectURL(f); img.classList.remove("hidden");
  $("#holo-drop span").textContent = f.name;
}
$("#holo-file").addEventListener("change", e => setHoloFile(e.target.files[0]));
const drop = $("#holo-drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setHoloFile(e.dataTransfer.files[0]));

function updateHoloVisibility() {
  const gs = $("#holo-method").value === "gs";
  $$(".gs-only").forEach(e => e.classList.toggle("hidden", !gs));
  $(".no-slm").classList.toggle("hidden", $("#holo-project").checked);
  $(".holo-gauss-opts").classList.toggle("hidden", !$("#holo-gauss").checked);
  if (!gs) $("#holo-size").value = $("#holo-size").value === "0.35" ? "0.7" : $("#holo-size").value;
}
["#holo-method", "#holo-project", "#holo-gauss"].forEach(s => $(s).addEventListener("change", updateHoloVisibility));
updateHoloVisibility();

const holoWatcher = new JobWatcher({
  onPoints: pts => holoPlot.add(pts),
  onLogs: lines => appendLog($("#holo-log"), lines),
  onProgress: j => { $("#holo-progress").textContent = j.progress ? `iteration ${j.progress.done} / ${j.progress.total}` : j.status; },
  onDone: j => {
    $("#holo-run").disabled = false; $("#holo-stop").disabled = true;
    if (j.status === "error") { $("#holo-status").textContent = "Error: " + j.error; toast(j.error, 8000); return; }
    const r = j.result || {};
    if (r.preview_recon) { $("#holo-recon").src = r.preview_recon; $("#holo-phase").src = r.preview_phase; $("#holo-download").classList.remove("hidden"); }
    $("#holo-status").textContent = (r.projected ? "Projected on the SLM. " : "Computed (not projected). ")
      + (r.efficiency !== undefined && r.efficiency !== null ? `Efficiency ${(100 * r.efficiency).toFixed(1)}%, correlation ${num(r.correlation, 4)}.` : "");
  },
});

$("#holo-run").addEventListener("click", async () => {
  if (!holoFile) { toast("Choose an image first."); return; }
  const v = s => $(s).value === "" ? null : Number($(s).value);
  const opts = {
    method: $("#holo-method").value, project: $("#holo-project").checked,
    size: v("#holo-size"), iterations: v("#holo-iters"), invert: $("#holo-invert").checked, gamma: v("#holo-gamma"),
    gaussian_beam: $("#holo-gauss").checked, beam_sigma_x: v("#holo-sx"), beam_sigma_y: v("#holo-sy"),
    grating_x: v("#holo-gx"), grating_y: v("#holo-gy"), focal_mm: v("#holo-f"), wavelength_nm: v("#holo-wl"),
    pixel_um: v("#holo-px"), width: v("#holo-w"), height: v("#holo-h"),
  };
  const fd = new FormData(); fd.append("image", holoFile); fd.append("opts", JSON.stringify(opts));
  try {
    const r = await api("/api/holo/run", { method: "POST", body: fd });
    holoPlot.reset(); $("#holo-log").textContent = ""; $("#holo-status").textContent = "Computing…";
    $("#holo-run").disabled = true; $("#holo-stop").disabled = false;
    holoWatcher.start(r.job_id);
  } catch (e) { toast(e.message, 6000); }
});
$("#holo-stop").addEventListener("click", () => holoWatcher.id && api(`/api/job/${holoWatcher.id}/stop`, { method: "POST" }));
$("#holo-blank").addEventListener("click", async () => {
  try { const r = await api("/api/holo/blank", { method: "POST" }); toast(r.released ? "SLM blanked and released." : "No hologram was being shown."); refreshStatus(); }
  catch (e) { toast(e.message); }
});

// ---------------------------------------------------------------- restore state after reload
(async () => {
  const s = await refreshStatus();
  if (!s) return;
  if (s.graph) showGraphSummary(s.graph);
  if (s.weights) showNppSummary(s.weights);
  if (s.job && (s.job.status === "running" || s.job.status === "starting")) {
    const t = s.job.kind === "holo" ? null : (s.job.kind === "maxcut" ? mc : npp);
    if (t) { t.runBtn.disabled = true; t.stopBtn.disabled = false; t.watcher.start(s.job.id); }
    else { $("#holo-run").disabled = true; $("#holo-stop").disabled = false; holoWatcher.start(s.job.id); }
  }
})();
