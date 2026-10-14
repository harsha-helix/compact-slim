"""
Virtual SLM + photodiode, installed underneath photonic_annealing.py.

Everything above this layer (annealer classes, mask pipeline, PD reader
thread, holography) runs unchanged. The SLM keeps the last displayed phase
mask and switches it at 60 Hz V-sync; the photodiode streams Tiva-format
frames ([0xA5][lo][hi], 2000 frames/s) of the on-axis (zero-order) intensity

    I = |sum_xy A(x, y) exp(i phi(x, y))|^2 / (sum_xy A)^2

for a Gaussian illumination A, which is what the focused photodiode sees.
"""
import collections
import threading
import time

import numpy as np

import photonic_annealing as pa

WIDTH, HEIGHT = 1920, 1080
PIXEL_UM = 8.0
FRAME_PERIOD = 1.0 / 60.0
LC_DELAY = 0.001            # liquid-crystal response after V-sync
PD_RATE = 2000.0            # Tiva firmware OUTPUT_RATE_HZ
PD_OFFSET_V = 0.02
PD_GAIN_V = 10.0         # random states sit well inside the ADC range; fully aligned ones clip at 3.3 V
PD_NOISE_LSB = 0.67
PORT = "COM21"


class _Optics:
    def __init__(self):
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()
        self.history = collections.deque([(self.t0, 1.0)], maxlen=256)  # (t_effective, I)
        self.last_vsync = self.t0
        self.set_beam(600.0, 600.0)

    def set_beam(self, sx, sy):
        y = (np.arange(HEIGHT) - HEIGHT / 2)[:, None]
        x = (np.arange(WIDTH) - WIDTH / 2)[None, :]
        amp = np.exp(-x ** 2 / (4 * sx ** 2) - y ** 2 / (4 * sy ** 2))
        self.amp = amp.ravel()
        self.amp_sum = float(self.amp.sum())
        self.lut = np.exp(1j * 2 * np.pi * np.arange(256) / 256.0)

    def intensity(self, data) -> float:
        a = np.asarray(data)
        if a.shape != (HEIGHT, WIDTH):
            return 0.0
        if a.dtype == np.uint8:
            field = np.dot(np.bincount(a.ravel(), weights=self.amp, minlength=256), self.lut)
        else:
            field = np.sum(self.amp * np.exp(1j * a.astype(np.float32).ravel()))
        return float(abs(field) ** 2 / self.amp_sum ** 2)

    def display(self, I: float):
        now = time.perf_counter()
        k = np.ceil((now - self.t0) / FRAME_PERIOD)
        vs = self.t0 + k * FRAME_PERIOD
        with self.lock:
            self.last_vsync = vs
            self.history.append((vs + LC_DELAY, I))

    def at(self, t: float) -> float:
        with self.lock:
            val = self.history[0][1]
            for ts, I in self.history:
                if ts <= t:
                    val = I
                else:
                    break
        return val


_optics = None


class _Handle:
    def __init__(self, I):
        self._I = I

    def show(self):
        _optics.display(self._I)
        return 0

    def release(self):
        pass


class _Window:
    def close(self):
        return 0


class _SLM:
    def errorCode(self):
        return 0

    def width_px(self):
        return WIDTH

    def height_px(self):
        return HEIGHT

    def pixelsize_um(self):
        return PIXEL_UM

    def refreshrate_hz(self):
        return 60.0

    def loadPhaseData(self, data, flags=None, phase_unit=None):
        return 0, _Handle(_optics.intensity(data))

    def showPhaseData(self, data, flags=None, phase_unit=None):
        _optics.display(_optics.intensity(data))
        return 0

    def showCGHFromImageData(self, data, flags=None, image_size_factor=0.7, max_unit_cell_size=0):
        _optics.display(0.0)
        return 0

    def showBlankScreen(self, value=0):
        _optics.display(1.0)
        return 0

    def waitForLastFrameDisplayed(self):
        dt = _optics.last_vsync - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        return 0

    def window(self):
        return _Window()


class _HEDS:
    class SDK:
        @staticmethod
        def Init(major, minor, base_port=0):
            return 0

        @staticmethod
        def ErrorString(err):
            return f"error {err}"

    class SLM:
        @staticmethod
        def Init(preselect="", openPreview=True, previewScale=1.0):
            return _SLM()

    class HEDSSlmShowPhaseFlags:
        SHOW_IN_BACKGROUND = 0


class _Serial:
    def __init__(self, port=None, baudrate=115200, timeout=None, **kw):
        if port != PORT:
            raise _real_serial_exception(f"could not open port '{port}': FileNotFoundError")
        self.port = port
        self.timeout = timeout or 0.001
        self.is_open = True
        self._rng = np.random.default_rng()
        self._reset()

    def _reset(self):
        self._t_start = time.perf_counter()
        self._sent = 0

    def reset_input_buffer(self):
        self._reset()

    flushInput = reset_input_buffer

    def read(self, n=1):
        due = int((time.perf_counter() - self._t_start) * PD_RATE) - self._sent
        if due <= 0:
            time.sleep(self.timeout)
            due = int((time.perf_counter() - self._t_start) * PD_RATE) - self._sent
            if due <= 0:
                return b""
        due = min(due, max(1, n // 3))
        out = bytearray()
        for k in range(due):
            t = self._t_start + (self._sent + k + 0.5) / PD_RATE
            v = PD_OFFSET_V + PD_GAIN_V * _optics.at(t)
            raw = int(round(v / 3.3 * 4095 + PD_NOISE_LSB * self._rng.standard_normal()))
            raw = min(4095, max(0, raw))
            out += bytes((0xA5, raw & 0xFF, raw >> 8))
        self._sent += due
        return bytes(out)

    def close(self):
        self.is_open = False


_saved = None
_real_serial_exception = pa.serial.SerialException


def enabled() -> bool:
    return _saved is not None


def enable():
    global _saved, _optics
    if _saved is not None:
        return
    _optics = _Optics()
    _saved = (pa.HEDS, pa.HEDSERR_NoError, pa.serial.Serial)
    pa.HEDS = _HEDS
    pa.HEDSERR_NoError = 0
    pa.serial.Serial = _Serial


def disable():
    global _saved
    if _saved is None:
        return
    pa.HEDS, pa.HEDSERR_NoError, pa.serial.Serial = _saved
    _saved = None


def set_beam(sx, sy):
    if _optics is not None:
        _optics.set_beam(float(sx), float(sy))


def ports():
    return [{"device": PORT, "description": f"Stellaris Virtual Serial Port ({PORT})"}]
