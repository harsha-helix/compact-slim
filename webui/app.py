"""
Web UI for the compact SLM photonic Ising machine.

    python -m webui                 # from the repo root, then open http://127.0.0.1:8050
    python -m webui --port 9000 --host 0.0.0.0

Tabs: Max-Cut, Number Partitioning, Holography. Only one job runs at a time,
because there is one SLM.
"""
import argparse
import base64
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid

os.environ.setdefault("MPLBACKEND", "Agg")  # annealer module imports pyplot; never open GUI windows from threads

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from webui import backends, cgh, problems, virtual_hw

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RUNS_DIR = os.path.join(REPO_ROOT, "webui_runs")
MAX_POINTS = 20000          # points kept per curve (older runs are decimated)
MAX_DRAW_EDGES = 6000       # edges sent to the browser for the graph drawing

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.status = "starting"     # starting | running | done | stopped | error
        self.points = []
        self.logs = []
        self.result = None
        self.error = None
        self.progress = None         # {done, total}
        self.started = time.time()
        self.stop_event = threading.Event()
        self.virtual_hw = virtual_hw.enabled()

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(f"[webui:{self.kind}] {msg}")
        self.logs.append(line)


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.job = None              # current/last job
        self.jobs = {}
        self.graph = None            # Max-Cut weight matrix
        self.graph_name = None
        self.weights = None          # NPP weights
        self.projector = None        # backends.SLMProjector while a hologram is shown
        self.last_phase = None       # last hologram phase (radians)

    def busy(self) -> bool:
        return self.job is not None and self.job.status in ("starting", "running")

    def release_projector(self):
        if self.projector is not None:
            try:
                self.projector.close()
            finally:
                self.projector = None


S = State()


def _start_job(kind: str, target, *args):
    with S.lock:
        if S.busy():
            return None, (jsonify(error=f"A {S.job.kind} job is already running. Stop it first."), 409)
        job = Job(kind)
        S.job = job
        S.jobs[job.id] = job

    def runner():
        try:
            job.status = "running"
            target(job, *args)
            if job.status == "running":
                job.status = "stopped" if job.stop_event.is_set() else "done"
        except backends.HardwareError as e:
            job.error = str(e)
            job.status = "error"
            job.log(f"Hardware error: {e}")
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
            job.status = "error"
            job.log(traceback.format_exc())
        job.log(f"Job {job.status}.")

    threading.Thread(target=runner, daemon=True).start()
    return job, None


def _f(d, key, default, cast=float):
    v = d.get(key, default)
    if v is None or v == "":
        return default
    return cast(v)


# ---------------------------------------------------------------------------
# Annealing (shared by Max-Cut and NPP)
# ---------------------------------------------------------------------------

def _anneal_job(job: Job, problem: str, data: np.ndarray, opts: dict):
    n = len(data)
    rng = np.random.default_rng(opts.get("seed") or None)
    is_maxcut = problem == "maxcut"

    if is_maxcut:
        W = data
        metric = lambda s: problems.cut_value(W, s)          # higher is better
        better = lambda a, b: a > b
        metric_name = "cut"
    else:
        w = data
        metric = lambda s: abs(problems.npp_diff(w, s))      # lower is better
        better = lambda a, b: a < b
        metric_name = "|diff|"

    # --- CPU references (shown as dashed lines while annealing) ---
    job.log("Computing CPU reference solutions...")
    t0 = time.time()
    refs = {}
    sweeps = int(_f(opts, "cpu_sweeps", 3000, int))
    if is_maxcut:
        refs["total_weight"] = problems.upper_weight(W)
        sa = problems.maxcut_cpu_sa(W, sweeps=sweeps, restarts=4)
        refs["cpu_sa"] = sa[0]
        ex = problems.maxcut_exact(W, max_n=int(_f(opts, "exact_max_n", 24, int)))
        refs["exact"] = ex[0] if ex else None
        refs["random_expected"] = refs["total_weight"] / 2.0
    else:
        refs["sum"] = float(w.sum())
        refs["karmarkar_karp"] = problems.npp_karmarkar_karp(w)
        refs["cpu_sa"] = problems.npp_cpu_sa(w, sweeps=sweeps, restarts=4)[0]
        refs["exact"] = problems.npp_exact(w)
    refs["cpu_time_s"] = time.time() - t0
    job.result = {"problem": problem, "n": n, "refs": refs, "metric": metric_name}
    job.log("CPU reference: " + ", ".join(f"{k}={v:.4g}" for k, v in refs.items() if isinstance(v, float)))
    if job.stop_event.is_set():
        return

    # --- Backend ---
    mode = opts.get("mode", "simulation")
    job.log(f"Opening {mode} backend (n={n})...")
    backend = backends.make_backend(problem, data, opts)
    try:
        _anneal_loop(job, backend, n, rng, opts, metric, better, is_maxcut, data)
    finally:
        backend.close()
        job.log("Backend closed.")


def _anneal_loop(job, backend, n, rng, opts, metric, better, is_maxcut, data):
    T0 = _f(opts, "t_initial", 1.0)
    Tf = _f(opts, "t_final", 0.01)
    rate = _f(opts, "cooling_rate", 0.9)
    steps_per_temp = max(1, _f(opts, "steps_per_temp", n, int))
    move = opts.get("move", "single")
    nc0 = max(1, _f(opts, "cluster_size", max(1, int(0.15 * n)), int))
    if not (0 < rate < 1) or T0 <= 0 or Tf <= 0 or Tf >= T0:
        raise ValueError("need 0 < cooling rate < 1 and 0 < final T < initial T")

    n_temps = math.ceil(math.log(Tf / T0) / math.log(rate))
    total = n_temps * steps_per_temp
    stride = max(1, math.ceil(total / MAX_POINTS))
    job.progress = {"done": 0, "total": total}

    evaluate = backend.evaluate
    rand_state = lambda: rng.choice([-1.0, 1.0], size=n)

    # --- Temperature scale ---
    scale = 1.0
    if opts.get("temp_mode", "relative") == "relative":
        k = max(2, _f(opts, "calibration_samples", 6, int))
        job.log(f"Calibrating energy scale from {k} random states...")
        samples = [evaluate(rand_state()) for _ in range(k)]
        scale = float(np.std(samples)) or float(np.mean(np.abs(samples))) or 1.0
        job.log(f"Energy scale (std of random states) = {scale:.4g}; T is multiplied by it.")
    job.result["temperature_scale"] = scale

    s = rand_state()
    E = evaluate(s)
    E0 = abs(E) if abs(E) > 1e-12 else 1.0
    m_cur = metric(s)
    best_E, best_s = E, s.copy()
    best_m, best_m_s = m_cur, s.copy()

    T = T0
    step = 0
    evals = 0
    accepted = 0
    t_start = time.time()
    job.log(f"Annealing: {n_temps} temperatures x {steps_per_temp} steps = {total} evaluations"
            + (f" (x{backend.evals_per_call} SLM masks each)" if backend.evals_per_call > 1 else ""))

    def record():
        job.points.append({"step": step, "T": T, "E": E, "best_E": best_E,
                           "m": m_cur, "best_m": best_m})

    record()
    for _ in range(n_temps):
        if job.stop_event.is_set():
            break
        for _ in range(steps_per_temp):
            if job.stop_event.is_set():
                break
            step += 1
            prop = s.copy()
            if move == "cluster":
                ratio = max(0.0, E / E0) if not is_maxcut else 1.0
                nc = int(round(nc0 * ratio ** (1 / 15))) if not is_maxcut else nc0
                nc = max(1, min(nc, n))
                prop[rng.choice(n, size=nc, replace=False)] *= -1
            else:
                prop[rng.integers(n)] *= -1

            E_new = evaluate(prop)
            evals += 1
            if not np.isfinite(E_new):
                job.log(f"step {step}: measurement failed, skipped")
                continue
            m_new = metric(prop)
            if better(m_new, best_m):
                best_m, best_m_s = m_new, prop.copy()
            dE = E_new - E
            if dE <= 0 or rng.random() < math.exp(-dE / (T * scale)):
                s, E, m_cur = prop, E_new, m_new
                accepted += 1
                if E < best_E:
                    best_E, best_s = E, s.copy()
            if step % stride == 0:
                record()
            job.progress = {"done": step, "total": total}
        T *= rate
    record()

    elapsed = time.time() - t_start
    r = job.result
    r.update({
        "steps": step,
        "evaluations": evals,
        "acceptance_rate": accepted / max(1, evals),
        "elapsed_s": elapsed,
        "evals_per_s": evals / elapsed if elapsed > 0 else None,
        "best_measured_energy": best_E,
        "best_measured_spins": best_s.astype(int).tolist(),
        "best_measured_metric": metric(best_s),
        "best_visited_metric": best_m,
        "best_visited_spins": best_m_s.astype(int).tolist(),
        "mode": backend.name,
    })
    if not is_maxcut:
        r["best_measured_partition"] = _partition_sums(data, best_s)
    job.log(f"Done: best measured state has {job.result['metric']} = {r['best_measured_metric']:.6g}; "
            f"best visited = {best_m:.6g} ({evals} evaluations in {elapsed:.1f}s)")
    _save_run(job, data)


def _partition_sums(w, s):
    w = np.asarray(w)
    return [float(w[s > 0].sum()), float(w[s < 0].sum())]


def _save_run(job, data):
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        path = os.path.join(RUNS_DIR, f"{job.kind}_{int(job.started)}_{job.id}.json")
        payload = {"result": job.result, "points": job.points, "logs": job.logs}
        if job.virtual_hw:
            payload["virtual_hardware"] = True
        payload["weights" if job.kind == "npp" else "W"] = np.asarray(data).tolist()
        with open(path, "w") as fh:
            json.dump(payload, fh)
        job.result["saved_to"] = os.path.relpath(path, REPO_ROOT)
    except Exception as e:
        job.log(f"Could not save run: {e}")


# ---------------------------------------------------------------------------
# Holography
# ---------------------------------------------------------------------------

def _holo_job(job: Job, image: bytes, opts: dict):
    method = opts.get("method", "gs")
    project = bool(opts.get("project"))

    if project:
        if S.projector is None:
            job.log("Opening SLM...")
            S.projector = backends.SLMProjector()
            job.log(f"SLM open: {S.projector.width} x {S.projector.height} px, "
                    f"pixel {S.projector.pixel_um} um")
        H, W = S.projector.height, S.projector.width
        pixel_um = _f(opts, "pixel_um", 0.0) or S.projector.pixel_um or 8.0
    else:
        H, W = _f(opts, "height", 1080, int), _f(opts, "width", 1920, int)
        pixel_um = _f(opts, "pixel_um", 8.0)

    if method == "sdk":
        if not project:
            raise ValueError("The HOLOEYE built-in CGH runs inside the SDK; enable 'Project on SLM'.")
        from PIL import Image
        import io
        img = np.asarray(Image.open(io.BytesIO(image)).convert("L"), dtype=np.uint8)
        if opts.get("invert"):
            img = 255 - img
        job.log("Computing + showing CGH with the HOLOEYE SDK...")
        S.projector.show_sdk_cgh(img, _f(opts, "size", 0.7))
        S.last_phase = None
        job.result = {"method": "sdk", "projected": True, "width": W, "height": H}
        job.log("Hologram is on the SLM.")
        return

    target, mask = cgh.load_target(image, H, W, size=_f(opts, "size", 0.35),
                                   invert=bool(opts.get("invert")), gamma=_f(opts, "gamma", 1.0))
    beam = cgh.gaussian_beam(H, W, _f(opts, "beam_sigma_x", 0.0), _f(opts, "beam_sigma_y", 0.0)) \
        if opts.get("gaussian_beam") else cgh.gaussian_beam(H, W, None, None)
    iters = max(1, _f(opts, "iterations", 30, int))
    job.progress = {"done": 0, "total": iters}
    job.log(f"Gerchberg-Saxton: {iters} iterations on {W} x {H}...")

    def on_iter(i, rmse, eff):
        job.points.append({"step": i, "rmse": rmse, "efficiency": eff})
        job.progress = {"done": i, "total": iters}
        return not job.stop_event.is_set()

    phase, recon, hist = cgh.gerchberg_saxton(target, mask, beam, iters, on_iter=on_iter)
    final = cgh.add_grating_and_lens(phase, _f(opts, "grating_x", 0.0), _f(opts, "grating_y", 0.0),
                                     _f(opts, "focal_mm", 0.0), _f(opts, "wavelength_nm", 633.0), pixel_um)
    S.last_phase = final
    sig = recon[mask]
    corr = float(np.corrcoef(sig.ravel(), target[mask].ravel())[0, 1]) if sig.size > 1 else None
    if corr is not None and not np.isfinite(corr):
        corr = None  # e.g. a uniform target; NaN is not valid JSON
    ys, xs = np.nonzero(mask)
    pad = 20
    crop = recon[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
    b64 = lambda png: "data:image/png;base64," + base64.b64encode(png).decode()
    job.result = {
        "method": "gs", "width": W, "height": H, "iterations": len(hist),
        "efficiency": hist[-1]["efficiency"] if hist else None,
        "correlation": corr,
        "preview_recon": b64(cgh.to_png(crop)),
        "preview_phase": b64(cgh.to_png(final, phase=True)),
        "projected": False,
    }
    job.log(f"GS done: efficiency {job.result['efficiency']:.1%}"
            + (f", correlation with target {corr:.3f}" if corr is not None else ""))
    if project and not job.stop_event.is_set():
        S.projector.show_phase(final)
        job.result["projected"] = True
        job.log("Hologram is on the SLM.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/api/status")
def status():
    j = S.job
    return jsonify(
        hardware=backends.hardware_status(),
        job=None if j is None else {"id": j.id, "kind": j.kind, "status": j.status},
        projecting=S.projector is not None,
        virtual=virtual_hw.enabled(),
        graph=None if S.graph is None else _graph_summary(S.graph, S.graph_name),
        weights=None if S.weights is None else _npp_summary(S.weights),
    )


def _graph_summary(W, name):
    n = len(W)
    iu = np.array(np.nonzero(np.triu(W, 1))).T
    out = {"name": name, "n": n, "edges": int(len(iu)), "total_weight": problems.upper_weight(W),
           "weights_distinct": sorted(set(np.round(W[iu[:, 0], iu[:, 1]], 6).tolist()))[:6] if len(iu) else []}
    if len(iu) <= MAX_DRAW_EDGES and n <= 1000:
        out["edge_list"] = iu.tolist()
        out["edge_weights"] = W[iu[:, 0], iu[:, 1]].tolist()
    return out


def _npp_summary(w):
    return {"n": len(w), "sum": float(w.sum()), "weights": w.tolist()}


@app.post("/api/virtual")
def set_virtual():
    on = bool(request.get_json(force=True).get("on"))
    with S.lock:
        if S.busy():
            return jsonify(error=f"A {S.job.kind} job is running. Stop it first."), 409
        if on != virtual_hw.enabled():
            S.release_projector()
            S.last_phase = None
            (virtual_hw.enable if on else virtual_hw.disable)()
    return jsonify(on=virtual_hw.enabled())


@app.post("/api/maxcut/generate")
def maxcut_generate():
    d = request.get_json(force=True)
    try:
        seed = d.get("seed")
        W = problems.random_graph(
            n=_f(d, "n", 20, int), kind=d.get("kind", "erdos_renyi"), p=_f(d, "p", 0.5),
            degree=_f(d, "degree", 3, int), weights=d.get("weights", "unit"),
            wmax=_f(d, "wmax", 10, int), seed=None if seed in (None, "") else int(seed))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if not np.any(W):
        return jsonify(error="graph has no edges; raise p"), 400
    S.graph, S.graph_name = W, f"{d.get('kind', 'erdos_renyi')} n={len(W)}"
    return jsonify(_graph_summary(W, S.graph_name))


@app.post("/api/maxcut/upload")
def maxcut_upload():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="no file"), 400
    try:
        W = problems.parse_edge_list(f.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return jsonify(error=f"could not parse graph: {e}"), 400
    S.graph, S.graph_name = W, f.filename
    return jsonify(_graph_summary(W, S.graph_name))


@app.post("/api/npp/generate")
def npp_generate():
    d = request.get_json(force=True)
    try:
        if d.get("custom"):
            w = np.array([float(x) for x in d["custom"].replace(",", " ").split()])
            if len(w) < 2 or np.any(w <= 0):
                raise ValueError("need at least 2 positive numbers")
        else:
            seed = d.get("seed")
            w = problems.random_npp(_f(d, "n", 40, int), _f(d, "wmax", 100, int),
                                    seed=None if seed in (None, "") else int(seed))
    except (ValueError, KeyError) as e:
        return jsonify(error=str(e)), 400
    S.weights = w
    return jsonify(_npp_summary(w))


@app.post("/api/run")
def run():
    d = request.get_json(force=True)
    problem = d.get("problem")
    if problem == "maxcut":
        data = S.graph
    elif problem == "npp":
        data = S.weights
    else:
        return jsonify(error="unknown problem"), 400
    if data is None:
        return jsonify(error="generate or load a problem first"), 400
    if d.get("mode") == "hardware":
        S.release_projector()   # the annealer opens the SLM itself
    job, err = _start_job(problem, _anneal_job, problem, data.copy(), d)
    if err:
        return err
    return jsonify(job_id=job.id)


@app.post("/api/holo/run")
def holo_run():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="no image"), 400
    try:
        opts = json.loads(request.form.get("opts", "{}"))
    except ValueError:
        return jsonify(error="bad options"), 400
    image = f.read()
    job, err = _start_job("holo", _holo_job, image, opts)
    if err:
        return err
    return jsonify(job_id=job.id)


@app.post("/api/holo/blank")
def holo_blank():
    if S.busy() and S.job.kind == "holo":
        return jsonify(error="hologram job still running"), 409
    was = S.projector is not None
    S.release_projector()
    return jsonify(released=was)


@app.get("/api/holo/mask.png")
def holo_mask():
    if S.last_phase is None:
        return jsonify(error="no hologram computed yet"), 404
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.fromarray(cgh.phase_to_u8(S.last_phase), mode="L").save(buf, format="PNG")
    return Response(buf.getvalue(), mimetype="image/png",
                    headers={"Content-Disposition": "attachment; filename=cgh_phase_8bit.png"})


@app.get("/api/job/<job_id>")
def job_status(job_id):
    j = S.jobs.get(job_id)
    if j is None:
        return jsonify(error="unknown job"), 404
    since = int(request.args.get("since", 0))
    log_since = int(request.args.get("log_since", 0))
    pts = j.points[since:]
    return jsonify(id=j.id, kind=j.kind, status=j.status, error=j.error,
                   points=pts, next=since + len(pts),
                   logs=j.logs[log_since:], log_next=log_since + len(j.logs[log_since:]),
                   progress=j.progress, elapsed=time.time() - j.started,
                   result=j.result if j.status not in ("starting", "running") or j.kind != "holo" else None,
                   refs=(j.result or {}).get("refs"))


@app.post("/api/job/<job_id>/stop")
def job_stop(job_id):
    j = S.jobs.get(job_id)
    if j is None:
        return jsonify(error="unknown job"), 404
    j.stop_event.set()
    return jsonify(ok=True)


@app.get("/api/job/<job_id>/download")
def job_download(job_id):
    j = S.jobs.get(job_id)
    if j is None:
        return jsonify(error="unknown job"), 404
    body = json.dumps({"result": j.result, "points": j.points, "logs": j.logs}, indent=1)
    return Response(body, mimetype="application/json",
                    headers={"Content-Disposition": f"attachment; filename={j.kind}_{j.id}.json"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Photonic Ising machine web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--mock", action="store_true", help="start with the mock hardware toggle on")
    args = ap.parse_args(argv)
    if args.mock:
        virtual_hw.enable()
    print(f"Web UI on http://{args.host}:{args.port}  (HOLOEYE SDK "
          f"{'found' if virtual_hw.enabled() or backends.pa.HEDS is not None else 'NOT found - hardware modes disabled'})")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        S.release_projector()


if __name__ == "__main__":
    main()
