# compact-slim

A compact **spatial photonic Ising machine** (SPIM) built around a HOLOEYE
phase-only SLM and a single photodiode. Spin configurations are encoded as
phase masks on the SLM; the photodiode measures the focused intensity, which
is proportional to the Ising energy of one (Mattis) eigenmode of the coupling
matrix. A simulated-annealing loop on the PC drives the SLM and uses those
measurements to search for low-energy states.

Supported problems:

- **Max-Cut / general Ising**: `J` is eigendecomposed and every eigenmode is
  streamed to the SLM. The energy is `sum_k lambda_k * PD_k`.
- **Number partitioning (NPP)**: rank-1 `J = w wᵀ`, so one mask per
  evaluation is enough. The photodiode reading is proportional to `(w·s)²`.

The repo also has a **web UI** with Max-Cut, NPP and holography (CGH) tabs.

---

## Hardware

| Part | Notes |
|---|---|
| HOLOEYE phase-only SLM | Driven through the *HOLOEYE SLM Display SDK (Python) v4.1*. Default resolution assumed 1920×1080. |
| Laser + optics | Gaussian beam onto the SLM, then a lens focusing onto the photodiode. Beam waist (σx, σy in SLM pixels) is used for amplitude compensation. |
| Photodiode + amplifier | Output must stay within **0–3.3 V** at the ADC pin. |
| TI EK-TM4C123GXL LaunchPad | Digitises the photodiode on **PE3 / AIN0** and streams it to the PC over the on-board ICDI virtual COM port. |

### Photodiode firmware (`firmware/tiva_pd_reader/`)

- Streams 3-byte frames `[0xA5][lo][hi]` (12-bit ADC value, little-endian) at
  **2000 frames/s**, **115200 8N1**. That is the format
  `PhotonicAnnealer._pd_reader_worker` parses. The host converts with
  `V = val / 4095 * 3.3`.
- Each frame averages 256 raw ADC samples: 64× hardware oversampling × 4 in
  software, timer-triggered at 8 kHz.
- The green LED (PF3) blinks at about 1 Hz while streaming.
- A prebuilt image is in `firmware/tiva_pd_reader/gcc/` (arm-none-eabi-gcc,
  TivaWare 2.2.0.295). Flash it with:

  ```sh
  cd firmware/tiva_pd_reader
  openocd -f board/ek-tm4c123gxl.cfg -c "program gcc/tiva_pd_reader.axf verify reset exit"
  ```

- To build with **CCS**, import TivaWare's
  `examples/boards/ek-tm4c123gxl/blinky` project. Replace `blinky.c` with
  `main.c`, and `startup_ccs.c` with the one in this folder. The stock
  startup file sends the ADC interrupt to `IntDefaultHandler`, which hangs
  the CPU.

---

## Software setup

Python 3.10+ (developed on Windows).

```sh
pip install -r requirements.txt
```

**HOLOEYE SDK.** Install the *SLM Display SDK (Python) v4.1*. Its installer
sets the `HEDS_4_1_PYTHON` environment variable, which the `HEDS` package in
this repo uses to find the SDK's native library. `photonic_annealing.py`
also looks in `HOLOEYE_SDK_PATH` (default
`C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0`).

**Without the SDK or the photodiode**, everything still imports and runs in
*offline mode*: no SLM output, and measurements are mocked. This is useful
for development. In the web UI, use *Simulation* mode instead.

**Serial port.** The default is `COM21`. Change it to the LaunchPad's port
(Device Manager on Windows, `/dev/ttyACM0` on Linux).

---

## Web UI

```sh
python -m webui                      # http://127.0.0.1:8050
python -m webui --host 0.0.0.0 --port 9000
```

Run it from the repo root. Only one job runs at a time, because there is one
SLM. Results are saved to `webui_runs/*.json` and can also be downloaded from
the page.

### Max-Cut tab

1. **Generate a graph**: Erdős–Rényi G(n, p), random d-regular, complete or
   2D torus, with unit, ±1, integer 1..k or Gaussian weights. You can also
   **load a G-set file** (`n m` header, then `i j w` lines, 1-based, e.g.
   `G17.txt`).
2. **Choose a mode**:
   - **SLM hardware**: `MaxCutAnnealer` streams all *n* eigenmodes per
     evaluation (n SLM frames). Settings: serial port, beam σx/σy, beam
     compensation on/off, turbo mode (fixed 8 ms settle instead of V-sync).
   - **Simulation**: an ideal CPU model of the measurement (`sᵀ J s`) with
     optional noise.
3. **Start annealing.** Two plots update live:
   - *Measured energy*: the current and best measured values.
   - *Cut value of the current state, computed on the CPU*: shown with
     dashed reference lines for classical CPU simulated annealing and, when
     n ≤ 24, the **exact optimum** (brute-force Gray-code search).
4. At the end, a **comparison table** shows:
   - the photonic best (by measured energy);
   - the best state visited (judged on the CPU);
   - CPU simulated annealing;
   - the exact optimum;
   - a random-partition baseline.

   The graph is redrawn with the best partition and its cut edges.

The temperature is, by default, a multiple of the spread of measured energies
over a few random states, which are measured at the start. The same schedule
therefore works for simulation and for photodiode volts. Choose *absolute* to
enter temperatures in measured-energy units instead.

### Number Partitioning tab

You can generate random integers or paste your own. Hardware mode uses
`RobustDynamicNPPAnnealer.evaluate_energy_direct`, which shows one rank-1
mask and reads the PD. Moves are adaptive cluster flips (`Nc = Nc0·(E/E0)^(1/15)`)
or single flips. The live plots show the PD energy and |partition difference|
(on a log scale). The results are compared with:

- **Karmarkar–Karp**;
- **CPU simulated annealing**;
- the **exact optimum** (subset-sum dynamic programming, integer weights).

### Holography tab

Upload an image and compute a phase-only **Fourier CGH** with
Gerchberg–Saxton. The window-constrained variant keeps light in the image
area and leaves the rest free. Then project it on the SLM and look at it
with your camera; the web UI does not show camera images. Options:

- image size;
- iterations;
- Gaussian illumination;
- **blazed grating**, to move the image off the zero-order spot (default
  period 6 px in x);
- optional **Fresnel lens** (focal length, wavelength, pixel pitch);
- invert and gamma.

There is also an option to use the **HOLOEYE SDK's built-in CGH**
(`showCGHFromImageData`). The page shows the GS convergence curve, a
simulated reconstruction for a sanity check, and the phase mask, which you
can download as a full-resolution 8-bit PNG. **Blank SLM** releases the
display. Starting a hardware annealing run also releases it automatically.

---

## Using the annealers from Python

```python
import numpy as np
from photonic_annealing import MaxCutAnnealer, RobustDynamicNPPAnnealer

J = np.random.randn(20, 20); J = (J + J.T) / 2
a = MaxCutAnnealer(J, beam_sigma_x=600, beam_sigma_y=600, serial_port="COM21")
try:
    spins, e_min, trace = a.run_annealing(initial_temp=1000, final_temp=0.1,
                                          cooling_rate=0.9, steps_per_temp=5)
finally:
    a.disconnect_hardware()
```

`python photonic_annealing.py` runs the built-in NPP benchmark. `test_slm.py`
runs the threaded NPP benchmark used for the `PD_benchmark_results_*.npy`
files.

---

## Repository layout

| Path | What it is |
|---|---|
| `photonic_annealing.py` | Main library. `PhotonicAnnealer` is the base class: SLM/serial connection, background PD reader thread, Numba mask fillers, and a producer/consumer mask pipeline. Subclasses: `MaxCutAnnealer`, `NumberPartitioningAnnealer` (batched), `RobustDynamicNPPAnnealer` (direct drive, adaptive cluster flips) and `ThreadedDynamicNPPAnnealer`. |
| `webui/` | Web UI (Flask): `app.py` server and job runner, `backends.py` hardware and simulated measurement, `problems.py` graph generators and CPU reference solvers, `cgh.py` Gerchberg–Saxton, `static/` front end (no CDN dependencies, so it works on an offline lab PC). |
| `HEDS/` | Python wrapper of the HOLOEYE SLM Display SDK v4.1 (vendor code; needs the SDK installed). |
| `firmware/tiva_pd_reader/` | TM4C123GXL photodiode streaming firmware, with a prebuilt image in `gcc/`. |
| `beam_comp.py`, `vectorized_beam_comp.py`, `beam_comp_multithreaded.py` | Earlier stand-alone versions of the compensated Mattis phase-mask generator (plotting only). |
| `simulated_annealing.py` | Earlier stand-alone SA script using `beam_comp` and an older SDK import. |
| `scripts/` | `compensated_phasemask.py` (mask generator), `linear_phase_to_greyscale_mapper.py` (radians to 8-bit). |
| `simulating_spim.py` | Free-space propagation simulation of an SLM pattern with `diffractsim` (optional dependency). |
| `slm_test.py` | Logs raw photodiode data from the Tiva over serial. |
| `*.ipynb` | Analysis and experiment notebooks. |
| `G17.txt`, `MC100.npz` | Benchmark Max-Cut instances (G-set G17, n=800; a 100-node complete graph stored as a scipy sparse matrix). |
| `data/`, `data_*.npz`, `gsetdata*.npz`, `*_benchmark_results_*.npy` | Recorded experiment results. |

---

## Known issues

These were found while checking the repo and are not fixed yet:

- `ThreadedDynamicNPPAnnealer.evaluate_energy_direct` fills the mask, but its
  upload/show code is commented out, so the SLM never changes during a run.
  `test_slm.py` uses this class. The web UI uses `RobustDynamicNPPAnnealer`
  instead.
- `test_slm.py` ends with `np.save(..., results)` while the line that defines
  `results` is commented out, so it raises `NameError` after the run.
- In `photonic_annealing.py`, the `__main__` block defines a placeholder
  `run_maxcut_benchmark()` (with `...` arguments) that is never called. Only
  the NPP benchmark runs.
- `beam_comp.py` and `scripts/compensated_phasemask.py` crash as scripts,
  because `run()` has `self.prep()` commented out.
- `MaxCutAnnealer` (Max-Cut hardware mode) needs n SLM frames per energy
  evaluation, so with V-sync (60 Hz) large graphs are slow. The web UI shows
  an estimate before you start.
