"""
Measurement backends for the web UI.

A backend turns a spin vector into one measured energy:
  - Hardware*: the real SLM + photodiode through the annealer classes in
    photonic_annealing.py (MaxCutAnnealer streams every eigenmode and sums
    lambda_k * PD_k; RobustDynamicNPPAnnealer shows one rank-1 mask and reads
    the PD).
  - Simulated*: a CPU model of the ideal optical measurement plus optional
    Gaussian noise, for trying the UI without hardware.
"""
import time

import numpy as np

import photonic_annealing as pa


class HardwareError(RuntimeError):
    pass


def hardware_status() -> dict:
    ports = []
    try:
        from serial.tools import list_ports
        ports = [{"device": p.device, "description": p.description} for p in list_ports.comports()]
    except Exception:
        pass
    return {"heds_available": pa.HEDS is not None, "serial_ports": ports}


def _check_connected(annealer, need_pd: bool = True):
    problems = []
    if annealer.slm is None:
        problems.append("SLM not connected (HOLOEYE SDK missing or SLM.Init failed)")
    if need_pd and annealer.ser is None:
        problems.append(f"photodiode serial port {annealer.serial_port} could not be opened")
    if problems:
        annealer.disconnect_hardware()
        raise HardwareError("; ".join(problems) + ". Check the server console for details, "
                            "or use Simulation mode.")


def _configure(annealer, opts: dict):
    annealer.sync_slm = not opts.get("turbo", False)
    if not opts.get("beam_compensation", True):
        annealer.beamcomp = False
        annealer._precompute_alpha_matrix()


class HardwareMaxCut:
    name = "hardware"

    def __init__(self, W: np.ndarray, opts: dict):
        self.annealer = pa.MaxCutAnnealer(
            J=W,
            beam_sigma_x=float(opts.get("beam_sigma_x", 600)),
            beam_sigma_y=float(opts.get("beam_sigma_y", 600)),
            serial_port=opts.get("serial_port", "COM21"),
            use_uint8=True,
        )
        _check_connected(self.annealer)
        _configure(self.annealer, opts)
        self.evals_per_call = len(self.annealer._active_modes)

    def evaluate(self, s: np.ndarray) -> float:
        return float(self.annealer.evaluate_energy(np.asarray(s)))

    def close(self):
        self.annealer.disconnect_hardware()


class HardwareNPP:
    name = "hardware"

    def __init__(self, w: np.ndarray, opts: dict):
        self.annealer = pa.RobustDynamicNPPAnnealer(
            npp_weights=w,
            beam_sigma_x=float(opts.get("beam_sigma_x", 600)),
            beam_sigma_y=float(opts.get("beam_sigma_y", 600)),
            serial_port=opts.get("serial_port", "COM21"),
            use_uint8=True,
        )
        _check_connected(self.annealer)
        _configure(self.annealer, opts)
        self.evals_per_call = 1

    def evaluate(self, s: np.ndarray) -> float:
        return float(self.annealer.evaluate_energy_direct(np.asarray(s)))

    def close(self):
        if self.annealer._direct_handle is not None:
            self.annealer._direct_handle.release()
            self.annealer._direct_handle = None
        self.annealer.disconnect_hardware()


class SimulatedMaxCut:
    """E = s^T Jn s with Jn = W / max|W| (the normalisation MaxCutAnnealer uses)."""
    name = "simulation"
    evals_per_call = 1

    def __init__(self, W: np.ndarray, opts: dict, seed=None):
        m = np.abs(W).max()
        self.J = W / m if m > 0 else W
        self.noise = float(opts.get("noise", 0.0)) * float(np.sqrt((self.J ** 2).sum()))
        self.delay = float(opts.get("sim_delay_ms", 0.0)) / 1000.0
        self.rng = np.random.default_rng(seed)

    def evaluate(self, s):
        if self.delay:
            time.sleep(self.delay)
        s = np.asarray(s, dtype=float)
        return float(s @ self.J @ s + self.noise * self.rng.standard_normal())

    def close(self):
        pass


class SimulatedNPP:
    """E = (w.s / sum w)^2, a PD-like reading in [0, 1]."""
    name = "simulation"
    evals_per_call = 1

    def __init__(self, w: np.ndarray, opts: dict, seed=None):
        self.w = np.asarray(w, dtype=float)
        self.scale = float(np.abs(self.w).sum()) or 1.0
        self.noise = float(opts.get("noise", 0.0))
        self.delay = float(opts.get("sim_delay_ms", 0.0)) / 1000.0
        self.rng = np.random.default_rng(seed)

    def evaluate(self, s):
        if self.delay:
            time.sleep(self.delay)
        v = (np.dot(self.w, s) / self.scale) ** 2
        return float(v + self.noise * self.rng.standard_normal())

    def close(self):
        pass


def make_backend(problem: str, data: np.ndarray, opts: dict):
    hw = opts.get("mode", "simulation") == "hardware"
    if problem == "maxcut":
        return HardwareMaxCut(data, opts) if hw else SimulatedMaxCut(data, opts)
    if problem == "npp":
        return HardwareNPP(data, opts) if hw else SimulatedNPP(data, opts)
    raise ValueError(problem)


# ---------------------------------------------------------------------------
# Holography: direct SLM access (no photodiode needed)
# ---------------------------------------------------------------------------

class SLMProjector:
    """Opens the SLM once and keeps it open while a hologram is projected."""

    def __init__(self):
        if pa.HEDS is None:
            raise HardwareError("HOLOEYE SDK not available, so the hologram cannot be projected. "
                                "Untick 'Project on SLM' to only compute and download the phase mask.")
        err = pa.HEDS.SDK.Init(4, 1)
        if err != pa.HEDSERR_NoError:
            raise HardwareError(f"SDK init failed: {pa.HEDS.SDK.ErrorString(err)}")
        self.slm = pa.HEDS.SLM.Init(openPreview=True)
        if self.slm.errorCode() != pa.HEDSERR_NoError:
            raise HardwareError(f"SLM init failed: {pa.HEDS.SDK.ErrorString(self.slm.errorCode())}")
        self.width = self.slm.width_px()
        self.height = self.slm.height_px()
        try:
            self.pixel_um = float(self.slm.pixelsize_um())
        except Exception:
            self.pixel_um = None

    def _check(self, err, what):
        if err != pa.HEDSERR_NoError:
            raise HardwareError(f"{what} failed: {pa.HEDS.SDK.ErrorString(err)}")

    def show_phase(self, phase: np.ndarray):
        """phase: float32 radians, shape (height, width)."""
        self._check(self.slm.showPhaseData(np.ascontiguousarray(phase, dtype=np.float32)), "showPhaseData")

    def show_sdk_cgh(self, image_u8: np.ndarray, size_factor: float):
        self._check(self.slm.showCGHFromImageData(np.ascontiguousarray(image_u8, dtype=np.uint8),
                                                  image_size_factor=float(size_factor)),
                    "showCGHFromImageData")

    def close(self):
        try:
            self.slm.showBlankScreen(0)
            self.slm.window().close()
        except Exception:
            pass
