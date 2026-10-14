"""
Computer-generated holography (Fourier hologram) via Gerchberg-Saxton.

The SLM sits in the back focal plane geometry: the far field (or the focal
plane of a lens) is the 2D Fourier transform of the SLM field. Given a target
intensity image we iterate between the SLM plane (amplitude fixed to the
illumination beam, phase free) and the image plane (amplitude fixed to the
target inside the signal window, free outside it -- "GS with a noise window",
which trades a little light for much better uniformity).
"""
import io
from typing import Callable, Optional

import numpy as np
from PIL import Image

try:
    from scipy import fft as _fft
    def _fft2(a):
        return _fft.fft2(a, workers=-1)
    def _ifft2(a):
        return _fft.ifft2(a, workers=-1)
except ImportError:
    _fft2, _ifft2 = np.fft.fft2, np.fft.ifft2


def load_target(image_bytes: bytes, height: int, width: int, size: float = 0.5,
                invert: bool = False, gamma: float = 1.0):
    """
    Load an image, convert to grey, fit it into a box of size*min(H,W) and
    centre it on an HxW canvas. Returns (target_intensity, signal_mask).
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L")
    box = max(8, int(size * min(height, width)))
    scale = box / max(img.size)
    new = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
    img = img.resize(new, Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        a = 1.0 - a
    if gamma != 1.0:
        a = a ** gamma
    target = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    y0 = (height - a.shape[0]) // 2
    x0 = (width - a.shape[1]) // 2
    target[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    mask[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = True
    return target, mask


def gaussian_beam(height: int, width: int, sigma_x: Optional[float], sigma_y: Optional[float]):
    if not sigma_x or not sigma_y:
        return np.ones((height, width), dtype=np.float32)
    y = np.arange(height, dtype=np.float32) - height / 2
    x = np.arange(width, dtype=np.float32) - width / 2
    return np.exp(-(x[None, :] ** 2) / (4 * sigma_x ** 2) - (y[:, None] ** 2) / (4 * sigma_y ** 2)).astype(np.float32)


def gerchberg_saxton(target: np.ndarray, mask: np.ndarray, beam: np.ndarray, iterations: int = 50,
                     seed: int = 0, on_iter: Optional[Callable[[int, float, float], bool]] = None):
    """
    Returns (phase in [0, 2pi), reconstructed intensity, history).
    on_iter(i, rmse, efficiency) may return False to stop early.
    """
    rng = np.random.default_rng(seed)
    t_amp = np.sqrt(np.clip(target, 0, None)).astype(np.float32)
    # Work in fft-shifted coordinates so the image is centred on the optical axis.
    t_amp_s = np.fft.ifftshift(t_amp)
    mask_s = np.fft.ifftshift(mask)
    t_norm = np.sqrt((t_amp_s[mask_s] ** 2).sum()) or 1.0
    beam = beam.astype(np.float32)

    phase = rng.uniform(0, 2 * np.pi, size=target.shape).astype(np.float32)
    history = []
    recon = None
    for it in range(1, iterations + 1):
        F = _fft2(beam * np.exp(1j * phase))
        amp = np.abs(F)
        # metrics on this iteration's far field
        sig = amp[mask_s]
        total = float((amp ** 2).sum()) or 1.0
        eff = float((sig ** 2).sum()) / total
        sig_n = sig / (np.sqrt((sig ** 2).sum()) or 1.0)
        rmse = float(np.sqrt(np.mean((sig_n - t_amp_s[mask_s] / t_norm) ** 2)) * np.sqrt(sig.size))
        history.append({"iter": it, "rmse": rmse, "efficiency": eff})
        if on_iter is not None and on_iter(it, rmse, eff) is False:
            break
        # image-plane constraint: target amplitude in the window, free outside
        G = np.where(mask_s, t_amp_s * np.exp(1j * np.angle(F)) * (np.sqrt(total) / t_norm), F)
        phase = np.angle(_ifft2(G)).astype(np.float32)
    recon = np.fft.fftshift(np.abs(_fft2(beam * np.exp(1j * phase))) ** 2)
    return np.mod(phase, 2 * np.pi).astype(np.float32), recon, history


def add_grating_and_lens(phase: np.ndarray, grating_x: float = 0.0, grating_y: float = 0.0,
                         focal_mm: float = 0.0, wavelength_nm: float = 633.0, pixel_um: float = 8.0):
    """
    Add a blazed grating (period in pixels; 0 = none) to move the image off the
    zero order, and an optional Fresnel lens (focal length in mm; 0 = none).
    """
    h, w = phase.shape
    y = (np.arange(h, dtype=np.float32) - h / 2)[:, None]
    x = (np.arange(w, dtype=np.float32) - w / 2)[None, :]
    out = phase.astype(np.float32)
    if grating_x:
        out = out + 2 * np.pi * x / float(grating_x)
    if grating_y:
        out = out + 2 * np.pi * y / float(grating_y)
    if focal_mm:
        p = pixel_um * 1e-6
        lam = wavelength_nm * 1e-9
        out = out - np.pi * ((x * p) ** 2 + (y * p) ** 2) / (lam * focal_mm * 1e-3)
    return np.mod(out, 2 * np.pi).astype(np.float32)


def to_png(a: np.ndarray, max_side: int = 640, log: bool = False, phase: bool = False) -> bytes:
    a = np.asarray(a, dtype=np.float64)
    if phase:
        img = (a / (2 * np.pi) * 255.0)
    else:
        if log:
            a = np.log1p(a / (np.percentile(a, 99.9) or 1.0) * 1000)
        hi = np.percentile(a, 99.9) or a.max() or 1.0
        img = np.clip(a / hi, 0, 1) * 255.0
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))),
                       Image.NEAREST if phase else Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def phase_to_u8(phase: np.ndarray) -> np.ndarray:
    """0..2pi -> 0..255 grey levels (same mapping as fill_mask_uint8)."""
    return (np.mod(phase, 2 * np.pi) * (255.0 / (2 * np.pi))).astype(np.uint8)
