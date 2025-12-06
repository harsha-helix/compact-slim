import sys
import os
import numpy as np
import random
import math
import time
import serial
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
import threading
import queue
import struct
from abc import ABC, abstractmethod # Import for Abstract Base Class
import collections

# Numba
try:
    import numba as nb
except ImportError:
    print("Warning: Numba not installed. Performance will be significantly reduced.")
    print("Install with: pip install numba")
    # Create dummy decorator if numba is not present
    class DummyNumba:
        def njit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def prange(self, *args):
            return range(*args)
    nb = DummyNumba()

# -----------------------------------------------------------------
# 1. HOLOEYE SDK Setup
# -----------------------------------------------------------------

# [USER] PLEASE VERIFY THIS PATH
# sdk_install_path = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"
# # --- Use a relative path or environment variable for better portability ---
sdk_install_path = os.environ.get("HOLOEYE_SDK_PATH", r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0")
python_api_path = os.path.join(sdk_install_path, "api", "python")

if python_api_path not in sys.path:
    if os.path.exists(python_api_path):
        sys.path.append(python_api_path)
        print(f"Added {python_api_path} to Python path.")
    else:
        print(f"Warning: Holoeye SDK path not found at {python_api_path}")
        print("Please verify the 'sdk_install_path' variable or set HOLOEYE_SDK_PATH environment variable.")


try:
    import HEDS
    from hedslib.heds_types import *
except ImportError:
    print(f"FATAL ERROR: Could not import HEDS library from {python_api_path}")
    print("This script will not be able to connect to the SLM.")
    # We won't exit, to allow for "offline" testing if HEDS is mocked
    HEDS = None
    HEDSERR_NoError = 0 # Define dummy error code

# -----------------------------------------------------------------
# 2. Global Numba Functions
# -----------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def fill_mask_uint8(buf, base_checkerboard,
                    alpha_ik, spin_phases,
                    spin_y0, spin_x0,
                    macro_pix_y, macro_pix_x):
    """
    Optimized filler for UINT8 buffers. 
    Maps phase 0..2pi -> 0..255 integers.
    """
    scale = 255.0 / (2.0 * math.pi)
    n = alpha_ik.shape[0]

    # Iterate spins in parallel
    for i in nb.prange(n):
        amplitude = alpha_ik[i]
        spin_phase = spin_phases[i]
        y0 = spin_y0[i]
        x0 = spin_x0[i]
        
        # Calculate values for +1 and -1 checkerboard entries
        # val = (spin_phase + checker * amplitude)
        val_plus = (spin_phase + amplitude) * scale
        val_minus = (spin_phase - amplitude) * scale
        
        # Fast modulo 256 logic for uint8 wrapping
        v_p = int(val_plus) % 256
        v_m = int(val_minus) % 256
        
        for yy in range(macro_pix_y):
            for xx in range(macro_pix_x):
                # Checkerboard logic: (-1)^(x+y)
                if base_checkerboard[yy, xx] > 0:
                    buf[y0 + yy, x0 + xx] = v_p
                else:
                    buf[y0 + yy, x0 + xx] = v_m

@nb.njit(parallel=True, cache=True)
def fill_mask_blocks_numba(buf, base_checkerboard,
                           alpha_ik, spin_phases,
                           spin_y0, spin_x0,
                           macro_pix_y, macro_pix_x):
    """
    Numba-compiled mask filler.
    - buf: 2D float32 array (SLM height x width)
    - base_checkerboard: 2D float32 array (macro_pix_y x macro_pix_x)
    - alpha_ik: 1D float32 length n_spins
    - spin_phases: 1D float32 length n_spins
    - spin_y0, spin_x0: 1D int32 start coords (length n_spins)
    - macro_pix_y, macro_pix_x: ints
    """
    two_pi = 2.0 * math.pi
    n = alpha_ik.shape[0]

    # iterate spins in parallel
    for i in nb.prange(n):
        amplitude = alpha_ik[i]
        spin_phase = spin_phases[i]
        y0 = spin_y0[i]
        x0 = spin_x0[i]
        # inner block fill
        for yy in range(macro_pix_y):
            for xx in range(macro_pix_x):
                val = spin_phase + base_checkerboard[yy, xx] * amplitude
                buf[y0 + yy, x0 + xx] = val % two_pi

# -----------------------------------------------------------------
# 3. Abstract Base Class: PhotonicAnnealer
# -----------------------------------------------------------------

class PhotonicAnnealer(ABC):
    """
    Abstract Base Class for a Photonic Ising Machine.

    Manages hardware connection (SLM, Photodiode), beam layout,
    and high-performance, streaming (FIFO) phase mask generation.

    Child classes must implement problem-specific logic:
    - _perform_eigendecomposition()
    - evaluate_energy()
    - run_annealing()
    """

    def __init__(
        self,
        beam_sigma_x: float,
        beam_sigma_y: float,
        serial_port: str = 'COM21',
        serial_baud: int = 115200,
        serial_timeout: float = 0.001,
        use_uint8: bool = True
    ):
        """
        Initializes the annealer and connects to hardware.

        Args:
            beam_sigma_x (float): Gaussian beam std dev in x.
            beam_sigma_y (float): Gaussian beam std dev in y.
            serial_port (str): The COM port for the Tiva microcontroller.
            serial_baud (int): The baud rate for the serial connection.
        """
        print("Initializing Photonic Annealer Base...")
        if HEDS is None:
            print("WARNING: HEDS SDK not loaded. SLM will not function.")
        self.stats = {
                    'mask_gen': 0.0,      # Time to produce 1 mask (Math)
                    'buffer_upload': 0.0, # Time to upload to GPU/SLM (API)
                    'slm_show': 0.0,      # Time to trigger display
                    'measurement': 0.0,   # Time for PD settle + Read
                    'queue_wait': 0.0,    # Dead time (Consumer waiting for Producer)
                    'producer_wait': 0.0, # Dead time (Producer waiting for Free Buffer)
                    'count_masks': 0,
                    'count_evals': 0
                }
        # --- Hardware Handles ---
        self.use_uint8 = use_uint8 # <--- STORE FLAG
        self.sync_slm = True # Synchronous SLM display by default
        self.slm: Optional[HEDS.SLM] = None
        self.ser: Optional[serial.Serial] = None
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.serial_timeout = serial_timeout

        # --- SLM Properties ---
        self.slm_width: int = 1920 # Default, will be overwritten
        self.slm_height: int = 1080 # Default, will be overwritten

        # --- Photodiode Reading Thread ---
        self._pd_thread: Optional[threading.Thread] = None
        self._pd_stop_event = threading.Event()
        self._pd_lock = threading.Lock()
        self._last_pd_raw = 0.0
        self._last_pd_avg = 0.0
        self._pd_buffer = collections.deque(maxlen=80) 
        self._last_photodiode_val: float = 0.0

        # --- Ising Model Properties (to be set by child) ---
        self.J: np.ndarray = np.empty((0, 0))
        self.num_spins: int = 0
        self.beam_sigma_x = beam_sigma_x
        self.beam_sigma_y = beam_sigma_y
        self.eigvals: np.ndarray = np.empty(0)
        self.eigvecs: np.ndarray = np.empty((0, 0))
        self._active_modes: List[int] = []

        # --- Connect to Hardware ---
        # Child __init__ must set num_spins *before* calling prep()
        self._connect_hardware()

        # --- FIFO Streaming Attributes ---
        self._buffer_pool_size = 10 # choose pool size
        dtype = np.uint8 if self.use_uint8 else np.float32
        
        self._mask_buffers: List[np.ndarray] = [
            np.zeros((self.slm_height, self.slm_width), dtype=dtype)
            for _ in range(self._buffer_pool_size)
        ]
        self._handle_lock = threading.Lock()
        self._handle_pool: List[Optional[HEDS.SLMDataHandle]] = [None] * self._buffer_pool_size
        self._free_buf_queue = queue.Queue(maxsize=self._buffer_pool_size)
        self._filled_buf_queue = queue.Queue(maxsize=self._buffer_pool_size + 2)
        self._reset_buffer_queues()

        self.producer_thread: Optional[threading.Thread] = None
        self.stop_producer_event = threading.Event()

        # --- Pre-computation Attributes (to be set by prep) ---
        self._center_x: float = self.slm_width / 2
        self._center_y: float = self.slm_height / 2
        self._grid_rows: int = 0
        self._grid_cols: int = 0
        self._macro_pix_x: int = 0
        self._macro_pix_y: int = 0
        self._compensation_factors: np.ndarray = np.empty(0)
        self._base_checkerboard: np.ndarray = np.empty((0, 0))
        self._spin_y0: np.ndarray = np.empty(0, dtype=np.int32)
        self._spin_x0: np.ndarray = np.empty(0, dtype=np.int32)
        self._spin_coords_x: np.ndarray = np.empty(0)
        self._spin_coords_y: np.ndarray = np.empty(0)
        self._alpha_matrix: np.ndarray = np.empty((0, 0))
        self.beamcomp: bool = True
         # For diagnostics

        # Upload initial empty buffers as handles
        self._upload_buffer_pool_as_handles()
        print("Base annealer initialized.")

    # ---------------------------------------------------
    # 1. Hardware Connection & Control
    # ---------------------------------------------------

    def _connect_hardware(self):
        """Initializes and connects to the HEDS SLM and Tiva Serial Port."""
        print("Connecting to hardware...")

        # 1. Initialize SDK & SLM
        if HEDS is not None:
            try:
                err = HEDS.SDK.Init(4, 1)
                if err != HEDSERR_NoError:
                    raise RuntimeError(f"Error initializing SDK: {HEDS.SDK.ErrorString(err)}")

                self.slm = HEDS.SLM.Init(openPreview=True)
                if self.slm.errorCode() != HEDSERR_NoError:
                    raise RuntimeError(f"Error initializing SLM: {HEDS.SDK.ErrorString(self.slm.errorCode())}")

                self.slm_width = self.slm.width_px()
                self.slm_height = self.slm.height_px()
                print(f"SLM connected. Resolution: {self.slm_width} x {self.slm_height}")
            except Exception as e:
                print(f"Failed to initialize HEDS SLM: {e}")
                print("Continuing in offline mode. SLM will not be used.")
                self.slm = None
        else:
            print("HEDS SDK not available. Running in offline mode.")
            # Use default resolution for layout calculations
            self.slm_width = 1920
            self.slm_height = 1080

        # Update centers
        self._center_x = self.slm_width / 2
        self._center_y = self.slm_height / 2


        # 2. Connect to Serial Port
        try:
            self.ser = serial.Serial(port=self.serial_port, baudrate=self.serial_baud, timeout=self.serial_timeout)
            time.sleep(1.0) # Wait for device to reset
            try:
                self.ser.reset_input_buffer()
            except AttributeError:
                self.ser.flushInput()
            if self.ser.is_open:
                print(f"Serial port {self.serial_port} opened at {self.serial_baud} baud.")
            else:
                raise RuntimeError("Serial port failed to open.")
            self.start_pd_thread()
        except serial.SerialException as e:
            print(f"Failed to connect to serial port {self.serial_port}: {e}")
            print("Continuing in offline mode. Photodiode will not be read.")
            self.ser = None

    def disconnect_hardware(self):
        """Safely disconnects from SLM and Serial port."""
        print("\nDisconnecting hardware...")
        self.stop_pd_thread()

        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.serial_port} closed.")

        self._clear_handle_pool()
        if self.slm is not None and HEDS is not None:
            self.slm.showBlankScreen(0) # Blank the SLM
            err = self.slm.window().close()
            if err == HEDSERR_NoError:
                print("SLM window closed.")
            # HEDS.SDK.Exit()
            print("HEDS SDK closed.")
        # elif HEDS is not None:
        #      # SDK might be init'd even if SLM failed
        #     try:
        #         HEDS.SDK.Exit()
        #         print("HEDS SDK closed (cleanup).")
        #     except Exception:
        #         pass

    def _pd_reader_worker(self):
        """Continuously read frames from serial and update PD raw & averaged values."""
        if self.ser is None or not self.ser.is_open:
            print("PD Reader: Serial port not open. Exiting thread.")
            return

        buf = bytearray()
        FRAME_SIZE = 3
        START_BYTE = 0xA5

        while not self._pd_stop_event.is_set():
            try:
                chunk = self.ser.read(512)
            except Exception as e:
                print(f"PD Reader: Serial read error: {e}")
                time.sleep(0.1)
                continue

            if not chunk:
                time.sleep(0.0002)
                continue

            buf.extend(chunk)
            i = 0

            while i + FRAME_SIZE <= len(buf):
                if buf[i] != START_BYTE:
                    i += 1
                    continue

                lo = buf[i+1]
                hi = buf[i+2]
                val = lo | (hi << 8)

                # Convert to voltage
                voltage = (val / 4095.0) * 3.3

                # Update buffer & moving average
                self._pd_buffer.append(voltage)
                avg_voltage = sum(self._pd_buffer) / len(self._pd_buffer)

                with self._pd_lock:
                    self._last_pd_raw = voltage
                    self._last_pd_avg = avg_voltage

                i += FRAME_SIZE

            if i > 0:
                del buf[:i]

    def start_pd_thread(self):
        if self.ser is None:
            print("Cannot start PD thread: Serial port not connected.")
            return
        if self._pd_thread is not None and self._pd_thread.is_alive():
            return
        self._pd_stop_event.clear()
        self._pd_thread = threading.Thread(target=self._pd_reader_worker, daemon=True)
        self._pd_thread.start()
        print("Photodiode reader thread started.")

    def stop_pd_thread(self):
        if self._pd_thread is None:
            return
        self._pd_stop_event.set()
        self._pd_thread.join(timeout=1.0)
        self._pd_thread = None
        print("Photodiode reader thread stopped.")

    # ---------------------------------------------------
    # 2. Buffer, Handle, and Streaming Management
    # ---------------------------------------------------
    ## Timing
    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0.0

    def print_stats(self):
        c = max(1, self.stats['count_masks'])
        print("\n--- PROFILING RESULTS (Average per Mask) ---")
        print(f"1. Mask Generation (Numba):   {self.stats['mask_gen']/c*1000:.3f} ms")
        print(f"2. Buffer Upload (SDK):       {self.stats['buffer_upload']/c*1000:.3f} ms")
        print(f"   (Total 'Produce' Time):    {(self.stats['mask_gen']+self.stats['buffer_upload'])/c*1000:.3f} ms")
        print(f"3. SLM Display Call:          {self.stats['slm_show']/c*1000:.3f} ms")
        print(f"4. Measurement (Settle+Read): {self.stats['measurement']/c*1000:.3f} ms")
        print(f"5. Dead Time (Consumer Wait): {self.stats['queue_wait']/c*1000:.3f} ms")
        print(f"   Dead Time (Producer Wait): {self.stats['producer_wait']/c*1000:.3f} ms")
        print("--------------------------------------------")

    def _reset_buffer_queues(self):
        """(Re)create and populate free/filled buffer queues safely."""
        self._free_buf_queue = queue.Queue(maxsize=len(self._mask_buffers))
        self._filled_buf_queue = queue.Queue(maxsize=len(self._mask_buffers) + 2)
        for i in range(len(self._mask_buffers)):
            self._free_buf_queue.put_nowait(i)
        with self._handle_lock:
            self._handle_pool = [None] * len(self._mask_buffers)

    def _safe_load_phase_data(self, buf: np.ndarray):
        """Upload 'buf' to the SLM and return a data-handle, or None on failure."""
        if self.slm is None:
            return None
        try:
            maybe = self.slm.loadPhaseData(buf)
        except Exception as e:
            print("Producer: loadPhaseData exception:", e)
            return None

        dh = None
        try:
            if isinstance(maybe, (tuple, list)):
                if len(maybe) >= 2:
                    err, cand = maybe[0], maybe[1]
                    if err == HEDSERR_NoError:
                        dh = cand
                    else:
                        print("Producer: loadPhaseData err:", HEDS.SDK.ErrorString(err))
                elif len(maybe) == 1:
                    dh = maybe[0]
            else:
                dh = maybe
        except Exception:
            dh = None # Ignore parsing errors

        return dh

    def _release_handle(self, handle):
        """Try to release an SLM data-handle (best-effort)."""
        if handle is None or self.slm is None:
            return
        # try:
        #     # Check for various SDK wrapper versions
        #     rel_fn = getattr(handle, "release", None) or \
        #              getattr(handle, "free", None) or \
        #              getattr(self.slm, "releasePhaseDataHandle", None) or \
        #              getattr(self.slm, "freeDataHandle", None)

        #     if callable(rel_fn):
        #         rel_fn(handle)
        # except Exception:
        #     pass # Ignore errors during release
        
        handle.release() # Try standard method

    def _upload_buffer_pool_as_handles(self):
        """Upload all buffers to populate the handle pool."""
        if self.slm is None:
            return
        for i, buf in enumerate(self._mask_buffers):
            dh = self._safe_load_phase_data(buf)
            with self._handle_lock:
                prev = self._handle_pool[i]
                self._handle_pool[i] = dh
            if prev is not None and prev is not dh:
                prev.release() # Release previous handle if it exists

    def _show_handle_or_buffer(self, buf_idx: int, buf: np.ndarray, wait_for_frame: bool = True):
        """Show mask by handle, falling back to buffer."""
        if self.slm is None:
            return (HEDSERR_NoError, False) # Simulate success in offline mode

        err = HEDSERR_NoError
        used_handle = False
        handle = None

        with self._handle_lock:
            if buf_idx < len(self._handle_pool):
                handle = self._handle_pool[buf_idx]

        if handle is not None:
            try:
                # # HEDS.ShowDataHandles expects a list
                # ret = HEDS.ShowDataHandles([handle])
                ret = handle.show() # Newer SDK versions may have this method
                err = int(ret)
                if err == HEDSERR_NoError:
                    used_handle = True
                else:
                    print(f"Warning: ShowDataHandles returned error: {HEDS.SDK.ErrorString(err)}")
            except Exception as e:
                print(f"Warning: ShowDataHandles(handle) raised exception: {e}")

        if not used_handle:
            try:
                flags = HEDS.HEDSSlmShowPhaseFlags.SHOW_IN_BACKGROUND
                err = self.slm.showPhaseData(buf, flags)
            except Exception:
                try: # Fallback to basic call
                    err = self.slm.showPhaseData(buf)
                except Exception as e:
                    print(f"Error: showPhaseData(buffer) failed: {e}")
                    return (getattr(HEDS, "HEDSERR_GeneralError", -1), False)

        if wait_for_frame:
            if self.sync_slm:
                # Standard Mode: Wait for hardware V-Sync (Reliable, capped at 60Hz)
                try:
                    wait_fn = getattr(self.slm, 'waitForLastFrameDisplayed', None)
                    if callable(wait_fn):
                        wait_fn()
                except Exception:
                    pass 
            else:
                # Turbo Mode: Manual sleep (Fast)
                # 8ms is safe for Liquid Crystal settling. 
                # You can try reducing this to 0.006 or 0.005 to go even faster.
                time.sleep(0.008) 
        # ----------------------------------------------------------------

        return (int(err), used_handle)

    def _clear_handle_pool(self):
        """Release & clear all stored data-handles."""
        with self._handle_lock:
            for i, h in enumerate(self._handle_pool):
                if h is not None:
                    h.release() # Release previous handle if it exists
                    self._handle_pool[i] = None

    def _producer_worker_handles(self, spin_vector: np.ndarray):
        """
        Producer thread with INSTRUMENTATION.
        """
        try:
            # OPTIMIZATION HINT: You create this array every single time. Move this out or cache it?
            spin_phases = np.where(spin_vector == 1, np.pi/2, 3*np.pi/2).astype(np.float32)

            for j_idx, k in enumerate(self._active_modes):
                if self.stop_producer_event.is_set():
                    break

                # MEASURE: Producer Waiting for Buffer
                t_wait_start = time.perf_counter()
                try:
                    buf_idx = self._free_buf_queue.get(timeout=5.0)
                except queue.Empty:
                    break
                self.stats['producer_wait'] += (time.perf_counter() - t_wait_start)

                buf = self._mask_buffers[buf_idx]
                
                # MEASURE: Mask Generation
                t_gen_start = time.perf_counter()
                try:
                    # Get the alpha column corresponding to this active mode
                    alpha_ik = self._alpha_matrix[:, j_idx]
                    
                    # <--- NEW SELECTION LOGIC --->
                    if self.use_uint8:
                        fill_mask_uint8(
                            buf,
                            self._base_checkerboard,
                            alpha_ik,
                            spin_phases,
                            self._spin_y0,
                            self._spin_x0,
                            self._macro_pix_y,
                            self._macro_pix_x
                        )
                    else:
                        # Original float32 filler
                        fill_mask_blocks_numba(
                            buf,
                            self._base_checkerboard,
                            alpha_ik,
                            spin_phases,
                            self._spin_y0,
                            self._spin_x0,
                            self._macro_pix_y,
                            self._macro_pix_x
                        )
                    # <--- END SELECTION LOGIC --->
                    
                except Exception as e:
                    print(f"Producer: fill failed at mode k={k} (idx {j_idx}): {e}")
                    self._free_buf_queue.put_nowait(buf_idx)
                    continue
                self.stats['mask_gen'] += (time.perf_counter() - t_gen_start)
                
                # MEASURE: Buffer Upload
                t_up_start = time.perf_counter()
                dh = self._safe_load_phase_data(buf)
                with self._handle_lock:
                    prev = self._handle_pool[buf_idx]
                    self._handle_pool[buf_idx] = dh
                if prev is not None and prev is not dh:
                    prev.release() # Release previous handle if it exists
                self.stats['buffer_upload'] += (time.perf_counter() - t_up_start)

                self.stats['count_masks'] += 1

                try:
                    self._filled_buf_queue.put((buf_idx, k), timeout=3.0)
                except queue.Full:
                    self._free_buf_queue.put_nowait(buf_idx)
                    continue

        except Exception as e:
            print("Producer exception:", e)
        finally:
            try:
                self._filled_buf_queue.put(None, timeout=1.0)
            except Exception:
                pass
    # ---------------------------------------------------
    # 3. Core Measurement Mechanisms
    # ---------------------------------------------------

    def _evaluate_spin_vector_streaming(self, spin_vector: np.ndarray, settle_time: float = 0.001):
        """
        Consumer generator with INSTRUMENTATION.
        """
        self.stop_producer_event.clear()
        
        if self.producer_thread is not None and self.producer_thread.is_alive():
             self.stop_producer_event.set()
             self.producer_thread.join(timeout=0.1)

        self.producer_thread = threading.Thread(
            target=self._producer_worker_handles,
            args=(spin_vector,),
            daemon=True
        )
        self.producer_thread.start()

        num_measured = 0
        
        try:
            while num_measured < len(self._active_modes):
                # MEASURE: Dead Time (Consumer waiting for Producer)
                t_wait_start = time.perf_counter()
                try:
                    item = self._filled_buf_queue.get(timeout=5.0)
                except queue.Empty:
                    break 
                self.stats['queue_wait'] += (time.perf_counter() - t_wait_start)

                if item is None: break 
                buf_idx, k = item
                buf = self._mask_buffers[buf_idx]

                # MEASURE: Show Mask
                t_show_start = time.perf_counter()
                err, used_handle = self._show_handle_or_buffer(buf_idx, buf, wait_for_frame=True)
                self.stats['slm_show'] += (time.perf_counter() - t_show_start)

                if err != HEDSERR_NoError:
                    self._free_buf_queue.put(buf_idx)
                    yield (k, float('inf'))
                    continue
                
                # MEASURE: Measurement (Settle + Read)
                t_meas_start = time.perf_counter()
                time.sleep(settle_time)
                with self._pd_lock:
                    measured_val = self._last_photodiode_val if self.ser is not None else print("cant read pd")
                self.stats['measurement'] += (time.perf_counter() - t_meas_start)

                num_measured += 1
                yield (k, measured_val)

                try:
                    self._free_buf_queue.put(buf_idx, timeout=1.0)
                except Exception:
                    pass

        finally:
            self.stop_producer_event.set()
            if self.producer_thread is not None:
                self.producer_thread.join(timeout=3.0)
            while True:
                try:
                    leftover = self._filled_buf_queue.get_nowait()
                    if leftover is None: break
                    idx, _ = leftover
                    self._free_buf_queue.put_nowait(idx)
                except Exception:
                    break
    def _evaluate_spin_vector_single_mode(self, spin_vector: np.ndarray, mode_index: int = 0, settle_time: float = 0.001) -> float:
        """
        Evaluates a single spin vector for a single eigenmode (e.g., for NPP).
        This is a blocking, non-streaming function.
        """
        try:
            buf_idx = self._free_buf_queue.get(timeout=2.0)
        except queue.Empty:
            print("SingleEval: timed out waiting for free buffer.")
            return float('inf')

        buf = self._mask_buffers[buf_idx]
        
        # Build spin_phases
        spin_phases = np.where(spin_vector == 1, np.pi/2, 3*np.pi/2).astype(np.float32)

        try:
            # Get the single alpha column for the specified mode
            alpha_column = self._alpha_matrix[:, mode_index]
        except Exception as e:
            print(f"SingleEval: Failed to get alpha_matrix column {mode_index}: {e}")
            self._free_buf_queue.put_nowait(buf_idx) # Return buffer
            return float('inf')

        # Fill mask
        try:
            fill_mask_blocks_numba(
                buf, self._base_checkerboard,
                alpha_column, spin_phases,
                self._spin_y0, self._spin_x0,
                self._macro_pix_y, self._macro_pix_x
            )
        except Exception as e:
            print(f"SingleEval: fill_mask failed: {e}")
            self._free_buf_queue.put_nowait(buf_idx)
            return float('inf')

        # Upload handle (best-effort)
        dh = self._safe_load_phase_data(buf)
        if dh is not None:
            with self._handle_lock:
                prev = self._handle_pool[buf_idx]
                self._handle_pool[buf_idx] = dh
            if prev is not None and prev is not dh:
                prev.release() # Release previous handle if it exists

        # Show and measure
        measured_val = float('inf')
        try:
            err, used_handle = self._show_handle_or_buffer(buf_idx, buf, wait_for_frame=True)
            if err != HEDSERR_NoError:
                print(f"SingleEval: show failed: {HEDS.SDK.ErrorString(err)}")
            else:
                time.sleep(settle_time)
                with self._pd_lock:
                    measured_val = self._last_photodiode_val if self.ser is not None else (0.5 + 0.1 * np.random.randn()) # Mock data
        except Exception as e:
            print(f"SingleEval: exception during show/measure: {e}")
        
        # Return buffer to free pool
        self._free_buf_queue.put_nowait(buf_idx)
        return measured_val


    def _evaluate_spin_vector_batch_single_mode(self, spin_vectors: List[np.ndarray], mode_index: int = 0, settle_time: float = 0.001):
        """
        Evaluates a batch of spin vectors, all for the same single eigenmode.
        Returns (list[measured_values], list[theory_values=None])
        """
        n_in = len(spin_vectors)
        measured_values = [float('inf')] * n_in
        theory_values = [None] * n_in # Not relevant here
        
        try:
            alpha_column = self._alpha_matrix[:, mode_index].astype(np.float32)
        except Exception as e:
            print(f"BatchEval: Failed to get alpha_matrix column {mode_index}: {e}")
            return measured_values, theory_values
        
        produced_map = {} # Maps input index -> buf_idx
        
        # 1. Produce all masks
        for i, svec in enumerate(spin_vectors):
            try:
                buf_idx = self._free_buf_queue.get(timeout=2.0)
            except queue.Empty:
                print(f"BatchEval: No free buffer for input index {i}")
                break # Stop producing

            buf = self._mask_buffers[buf_idx]
            spin_phases = np.where(svec == 1, np.pi/2, 3*np.pi/2).astype(np.float32)
            
            try:
                fill_mask_blocks_numba(
                    buf, self._base_checkerboard,
                    alpha_column, spin_phases,
                    self._spin_y0, self._spin_x0,
                    self._macro_pix_y, self._macro_pix_x
                )
            except Exception as e:
                print(f"BatchEval: fill failed at idx {i}: {e}")
                self._free_buf_queue.put_nowait(buf_idx) # Return buffer
                continue # Skip to next vector

            # Upload handle (best-effort)
            dh = self._safe_load_phase_data(buf)
            if dh is not None:
                with self._handle_lock:
                    prev = self._handle_pool[buf_idx]
                    self._handle_pool[buf_idx] = dh
                if prev is not None and prev is not dh:
                    prev.release() # Release previous handle if it exists

            produced_map[i] = buf_idx

        # 2. Consume all produced masks
        for i, buf_idx in produced_map.items():
            buf = self._mask_buffers[buf_idx]
            try:
                err, used_handle = self._show_handle_or_buffer(buf_idx, buf, wait_for_frame=True)
                if err != HEDSERR_NoError:
                    print(f"BatchEval: show failed for input idx {i}")
                else:
                    time.sleep(settle_time)
                    with self._pd_lock:
                        measured_values[i] = self._last_photodiode_val if self.ser is not None else (0.5 + 0.1 * np.random.randn()) # Mock
            except Exception as e:
                print(f"BatchEval: show/measure exception for idx {i}: {e}")
            
            # Always return buffer
            self._free_buf_queue.put_nowait(buf_idx)

        return measured_values, theory_values

    # ---------------------------------------------------
    # 4. Preparation & Abstract Methods
    # ---------------------------------------------------

    def prep(self):
        """
        Runs all necessary setup calculations in the correct order.
        Called by child class __init__ AFTER num_spins is set.
        """
        if self.num_spins == 0:
            raise RuntimeError("prep() called before num_spins was set by child class.")
        
        print(f"Preparing model for {self.num_spins} spins...")
        self._setup_layout()
        print(f"  - SLM Layout: {self._grid_rows} rows x {self._grid_cols} cols")
        print(f"  - Macropixel Size: {self._macro_pix_x} x {self._macro_pix_y} pixels")
        self._precompute_checkerboard()
        self._compute_compensation_factors()
        
        # Call abstract method to be implemented by child
        self._perform_eigendecomposition()
        print(f"  - Identified {len(self._active_modes)} active eigenmodes.")
        
        self._precompute_alpha_matrix()
        print(f"  - Precomputed alpha matrix ({self._alpha_matrix.shape}).")
        print("Preparation complete.")

    def _setup_layout(self):
        """Determines an optimal rectangular macropixel layout."""
        best_layout = (0, 0)
        max_area = 0
        for rows in range(1, self.num_spins + 1):
            cols = math.ceil(self.num_spins / rows)
            if cols > self.slm_width or rows > self.slm_height:
                continue
            
            pixel_width = self.slm_width // cols
            pixel_height = self.slm_height // rows
            if pixel_width == 0 or pixel_height == 0:
                continue
            
            # Favor squarer macropixels
            aspect_ratio_diff = abs((pixel_width / pixel_height) - (self.slm_width / self.slm_height))
            area = pixel_width * pixel_height

            if area > max_area:
                max_area = area
                best_layout = (rows, cols)

        self._grid_rows, self._grid_cols = best_layout
        if self._grid_rows * self._grid_cols < self.num_spins or max_area == 0:
             raise RuntimeError(f"Failed to find a valid grid for {self.num_spins} spins on a {self.slm_width}x{self.slm_height} SLM.")

        self._macro_pix_x = self.slm_width // self._grid_cols
        self._macro_pix_y = self.slm_height // self._grid_rows
        total_width = self._grid_cols * self._macro_pix_x
        total_height = self._grid_rows * self._macro_pix_y
        grid_offset_x = (self.slm_width - total_width) // 2
        grid_offset_y = (self.slm_height - total_height) // 2

        # Re-initialize coordinate arrays
        self._spin_coords_x = np.zeros(self.num_spins)
        self._spin_coords_y = np.zeros(self.num_spins)
        self._spin_y0 = np.empty(self.num_spins, dtype=np.int32)
        self._spin_x0 = np.empty(self.num_spins, dtype=np.int32)

        for i in range(self.num_spins):
            row = i // self._grid_cols
            col = i % self._grid_cols
            x0 = grid_offset_x + col * self._macro_pix_x
            y0 = grid_offset_y + row * self._macro_pix_y
            self._spin_coords_x[i] = x0 + self._macro_pix_x / 2
            self._spin_coords_y[i] = y0 + self._macro_pix_y / 2
            self._spin_y0[i] = y0
            self._spin_x0[i] = x0

    def _precompute_checkerboard(self):
        """Pre-computes the base checkerboard pattern once."""
        lx = np.arange(self._macro_pix_x, dtype=np.int32)
        ly = np.arange(self._macro_pix_y, dtype=np.int32)
        lx_grid, ly_grid = np.meshgrid(lx, ly)
        self._base_checkerboard = ((-1)**(lx_grid + ly_grid)).astype(np.float32)

    def _compute_compensation_factors(self):
        """Computes compensation factors (1/sqrt(I)) vectorized."""
        intensities = np.exp(
            -(((self._spin_coords_x - self._center_x)**2) / (2 * self.beam_sigma_x**2) +
              ((self._spin_coords_y - self._center_y)**2) / (2 * self.beam_sigma_y**2))
        )
        intensities[intensities < 1e-9] = 1e-9
        self._compensation_factors = 1.0 / np.sqrt(intensities)

    def _precompute_alpha_matrix(self):
        """Precompute alpha_ik but only for active modes."""
        n = self.num_spins
        m = len(self._active_modes) # Number of active modes
        self._alpha_matrix = np.empty((n, m), dtype=np.float32)

        for j_idx, k in enumerate(self._active_modes):
            # k is the *actual* eigenvector index (0 to N-1)
            # j_idx is the *column* index in our alpha matrix (0 to M-1)
            target_amplitudes = self._compensation_factors * self.eigvecs[:, k]
            max_abs_val = np.max(np.abs(target_amplitudes))
            if max_abs_val < 1e-9:
                max_abs_val = 1.0
            normalized = np.clip(target_amplitudes / max_abs_val, -1.0, 1.0)
            self._alpha_matrix[:, j_idx] = np.arccos(normalized).astype(np.float32)
        
        if not self.beamcomp:
            self._alpha_matrix.fill(np.pi / 4) # No compensation

    # --- Abstract methods to be implemented by children ---

    @abstractmethod
    def _perform_eigendecomposition(self):
        """
        Problem-specific: Calculate eigvals, eigvecs, and _active_modes.
        This method must set:
        - self.eigvals (1D array)
        - self.eigvecs (2D array)
        - self._active_modes (list of ints, e.g., [0, 1, 5, ...])
        """
        pass

    @abstractmethod
    def evaluate_energy(self, spin_vector: Union[np.ndarray, List[np.ndarray]]):
        """
        Problem-specific: Evaluate the energy of one or more spin vectors.
        - If spin_vector is a 1D array, return a single float energy.
        - If spin_vector is a list of 1D arrays, return a list of float energies.
        """
        pass

    @abstractmethod
    def run_annealing(self, initial_temp, final_temp, cooling_rate, steps_per_temp, **kwargs):
        """
        Problem-specific: Run the simulated annealing loop.
        """
        pass


# -----------------------------------------------------------------
# 4. Child Class: MaxCutAnnealer
# -----------------------------------------------------------------

class MaxCutAnnealer(PhotonicAnnealer):
    """
    Photonic Annealer for the Max-Cut problem (or any general Ising model)
    defined by a full interaction matrix J.
    """
    def __init__(
        self,
        J: np.ndarray,
        beam_sigma_x: float,
        beam_sigma_y: float,
        serial_port: str = 'COM21',
        serial_baud: int = 115200,
        use_uint8: bool = True,
        factorizing: bool = False
    ):
        """
        Initializes the annealer for a Max-Cut problem.

        Args:
            J (np.ndarray): The n_spins x n_spins interaction matrix.
            ... (other args passed to base class)
        """
        print("Initializing MaxCut Annealer...")
        # Call base __init__ to connect hardware
        super().__init__(
            beam_sigma_x=beam_sigma_x,
            beam_sigma_y=beam_sigma_y,
            serial_port=serial_port,
            serial_baud=serial_baud,
            use_uint8=use_uint8
        )

        if not isinstance(J, np.ndarray) or J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square 2D numpy array.")
        if not np.allclose(J, J.T):
            print("Warning: Interaction matrix J is not symmetric.")

        max_coupling = np.max(np.abs(J))
        self.J = J / max_coupling if max_coupling > 0 else J
        self.num_spins = J.shape[0]
        self.factorizing = factorizing
        
        # --- Run Full Preparation ---
        # This calls _perform_eigendecomposition() implemented below
        self.prep()

    def _perform_eigendecomposition(self):
        """
        (Max-Cut) Decompose J and find all eigenmodes above a threshold.
        """
        print("  - (MaxCut) Performing full eigendecomposition...")
        self.eigvals, self.eigvecs = np.linalg.eigh(self.J)
        magnitudes = np.abs(self.eigvals)
        max_mag = np.max(magnitudes) if magnitudes.size else 0.0

        # Keep all modes above a small threshold
        threshold = 0.01 * max_mag
        self._active_modes = [k for k in range(self.num_spins) if True] #magnitudes[k] >= threshold] condition of if statement removed to use all modes
        
        # Ensure we have at least one mode if possible
        if not self._active_modes and self.num_spins > 0:
             self._active_modes = [int(np.argmax(magnitudes))]

    def evaluate_energy(self, spin_vector: Union[np.ndarray, List[np.ndarray]]) -> float:
        """
        (Max-Cut) Evaluate energy by streaming all active modes and summing
        their weighted measurements.
        
        Note: This implementation does not support batching.
        """
        if isinstance(spin_vector, list):
            raise NotImplementedError("Batch evaluation is not implemented for MaxCutAnnealer. Use NumberPartitioningAnnealer for batching.")
        
        total_energy = 0.0
        modes_measured = 0
        
        # Use the streaming consumer from the base class
        # It yields (k, measured_val) for each k in self._active_modes
        for k, measured_val in self._evaluate_spin_vector_streaming(spin_vector, settle_time=0):
            if np.isinf(measured_val):
                print(f"Warning: Measurement failed for mode k={k}")
                total_energy = float('inf') # Propagate error
                break
                
            try:
                # k is the *actual* eigenvector index
                total_energy += measured_val * self.eigvals[k]
                modes_measured += 1
            except IndexError:
                print(f"Error: Mode index {k} out of bounds for eigenvalues.")
                total_energy = float('inf')
                break
        
        if modes_measured != len(self._active_modes) and not np.isinf(total_energy):
            print(f"Warning: Measured {modes_measured}/{len(self._active_modes)} modes.")
            # Optionally return inf if not all modes were measured
            # total_energy = float('inf')
        print(f"  - Total measured energy: {total_energy:.4f}")
        return total_energy

    def run_annealing(self, initial_temp, final_temp, cooling_rate, steps_per_temp, **kwargs):
        """
        (Max-Cut) Performs the standard simulated annealing algorithm.
        """
        current_spin_vector = np.random.choice([-1, 1], size=self.num_spins)
        
        print("Evaluating initial random spin configuration (Max-Cut)...")
        start_time_initial = time.time()
        current_energy = self.evaluate_energy(current_spin_vector)
        end_time_initial = time.time()
        print(f"Initial energy: {current_energy:.4f} (eval time: {end_time_initial - start_time_initial:.2f}s)")

        energy_plot = [current_energy]
        temp = initial_temp
        annealing_start_time = time.time()
        min_energy = current_energy
        min_spin_vector = np.copy(current_spin_vector)

        while temp > final_temp:
            print(f"\nCurrent Temperature: {temp:.4f}")
            start_temp_time = time.time()
            
            for step in range(steps_per_temp):
                idx = random.randint(0, self.num_spins - 1)
                
                proposed_spin_vector = np.copy(current_spin_vector)
                proposed_spin_vector[idx] *= -1
                if self.factorizing:
                    proposed_spin_vector[-1] = 1

                eval_start = time.time()
                proposed_energy = self.evaluate_energy(proposed_spin_vector)
                eval_time = time.time() - eval_start
                
                if np.isinf(proposed_energy):
                    print(f"  Step {step+1}/{steps_per_temp} | Eval failed. Skipping.")
                    continue

                delta_energy = proposed_energy - current_energy

                if proposed_energy < min_energy:
                    min_energy = proposed_energy
                    min_spin_vector = np.copy(current_spin_vector)
                
                acceptance_prob = math.exp(-delta_energy / temp) if temp > 0 else 0.0
                is_accepted = (delta_energy < 0) or (random.random() < acceptance_prob)

                if is_accepted:
                    current_spin_vector = proposed_spin_vector
                    current_energy = proposed_energy
                    print(f"  Step {step+1}/{steps_per_temp} | New Energy Accepted: {current_energy:.4f} (eval: {eval_time:.2f}s)")
                
                energy_plot.append(current_energy)

            end_temp_time = time.time()
            print(f"  Temp step took {end_temp_time - start_temp_time:.2f}s")
            temp *= cooling_rate
            
        print("\nSimulated annealing (Max-Cut) finished.")
        total_time = time.time() - annealing_start_time
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Minimum energy found: {min_energy:.4f}")

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(energy_plot)
        plt.xlabel("Annealing Step")
        plt.ylabel("Measured Energy")
        plt.title("Photonic Annealing (Max-Cut) Energy Convergence")
        plt.show(block=False)

        return min_spin_vector, min_energy, energy_plot

# -----------------------------------------------------------------
# 5. Child Class: NumberPartitioningAnnealer
# -----------------------------------------------------------------

class NumberPartitioningAnnealer(PhotonicAnnealer):
    """
    Photonic Annealer for the Number Partitioning Problem (NPP).
    This is a rank-1 problem, allowing for much faster single-mode
    and batched evaluation.
    """
    def __init__(
        self,
        npp_weights: np.ndarray,
        beam_sigma_x: float,
        beam_sigma_y: float,
        serial_port: str = 'COM21',
        serial_baud: int = 115200
    ):
        """
        Initializes the annealer for an NPP problem.

        Args:
            npp_weights (np.ndarray): 1D array of weights to be partitioned.
            ... (other args passed to base class)
        """
        print("Initializing Number Partitioning Annealer...")
        # Call base __init__ to connect hardware
        super().__init__(
            beam_sigma_x=beam_sigma_x,
            beam_sigma_y=beam_sigma_y,
            serial_port=serial_port,
            serial_baud=serial_baud
        )

        w = np.asarray(npp_weights, dtype=np.float64).flatten()
        self.npp_weights = w
        self.num_spins = w.shape[0]
        
        # Construct the rank-1 J matrix
        self.J = np.outer(w, w)
        
        # --- Run Full Preparation ---
        self.prep()

    def _perform_eigendecomposition(self):
        """
        (NPP) Decompose J and find the *single* dominant eigenmode.
        """
        print("  - (NPP) Performing rank-1 eigendecomposition...")
        self.eigvals, self.eigvecs = np.linalg.eigh(self.J)
        magnitudes = np.abs(self.eigvals)

        # For rank-1, we only care about the single largest magnitude mode
        k_star = int(np.argmax(magnitudes))
        self._active_modes = [k_star]
        
        # Store the dominant eigenvalue (for theoretical energy)
        self.dominant_eigval = self.eigvals[k_star]

    def evaluate_energy(self, spin_vector: Union[np.ndarray, List[np.ndarray]]):
        """
        (NPP) Evaluate "energy" (raw photodiode value) for one or more vectors.
        This is a dispatcher for single-mode evaluation.
        The "energy" is just the photodiode reading, which is proportional
        to (w . s)^2.
        """
        # The alpha matrix for NPP only has one column (j_idx=0),
        # which corresponds to the dominant eigenmode (k_star).
        MODE_INDEX = 0 
        
        if isinstance(spin_vector, list):
            # Batch evaluation
            measured_vals, _ = self._evaluate_spin_vector_batch_single_mode(
                spin_vector, mode_index=MODE_INDEX
            )
            return measured_vals
        else:
            # Single vector evaluation
            measured_val = self._evaluate_spin_vector_single_mode(
                spin_vector, mode_index=MODE_INDEX
            )
            return measured_val

    def run_annealing(
        self,
        initial_temp: float,
        final_temp: float,
        cooling_rate: float,
        steps_per_temp: int,
        batch_size: int = 8,
        verbose: bool = True,
        **kwargs
    ):
        """
        (NPP) Batched (approximate) Simulated Annealing.
        Measures `batch_size` candidates in parallel and selects one.
        """
        print(f"Running Batched NPP Annealing (batch_size={batch_size})...")
        batch_size = max(1, int(batch_size))
        steps_per_temp = max(1, int(steps_per_temp))

        # initialize state
        current_spin_vector = np.random.choice([-1, 1], size=self.num_spins)
        
        t0 = time.time()
        current_energy = self.evaluate_energy(current_spin_vector)
        if verbose:
            print(f"Initial measured energy: {current_energy:.6f} (eval time: {time.time()-t0:.2f}s)")

        energy_trace = [current_energy]
        temp = float(initial_temp)
        anneal_start = time.time()
        min_energy = current_energy
        min_spin_vector = np.copy(current_spin_vector)

        EPS = 1e-12
        MAX_EXP_ARG = 700 # avoid overflow in exp

        while temp > final_temp:
            if verbose:
                print(f"\nTemperature: {temp:.6f}")
            start_temp_t = time.time()

            for step in range(steps_per_temp):
                # 1) Build batch of candidate spin vectors
                # ... inside the step loop ...

                # 1) Build batch of candidates (SAME AS YOUR CODE)
                candidates = []
                candidate_indices = []
                for _ in range(batch_size):
                    idx = random.randint(0, self.num_spins - 1)
                    candidate = np.copy(current_spin_vector)
                    candidate[idx] *= -1
                    candidates.append(candidate)
                    candidate_indices.append(idx)

                # 2) Evaluate batch (SAME AS YOUR CODE)
                measured_energies = self.evaluate_energy(candidates) 

                # 3) Compute selection weights
                weights = []
                valid_candidates = []

                # --- FIX START: Add probability of staying put ---
                # We add the current state as a valid option.
                # Delta E is 0, so Weight is exp(0) = 1.0
                weights.append(1.0) 
                valid_candidates.append({
                    'index_in_batch': -1, # Marker for "Current State"
                    'delta_E': 0.0,
                    'prob': 1.0,
                    'E_prop': current_energy
                })
                # --- FIX END ---

                for i, E_prop in enumerate(measured_energies):
                    if not np.isfinite(E_prop):
                        weights.append(0.0)
                    else:
                        delta = float(E_prop) - float(current_energy)
                        # Note: If delta is negative (good move), arg is positive (weight > 1)
                        # If delta is positive (bad move), arg is negative (weight < 1)
                        arg = -delta / max(EPS, temp)
                        
                        if arg > MAX_EXP_ARG: arg = MAX_EXP_ARG
                        
                        try:
                            w = math.exp(arg)
                        except OverflowError:
                            w = float('inf') if arg > 0 else 0.0
                        
                        weights.append(w)
                        valid_candidates.append({
                            'index_in_batch': i,
                            'delta_E': delta,
                            'prob': w,
                            'E_prop': E_prop
                        })
                
                # 4) Roulette Wheel Selection
                total_w = float(sum(weights))
                
                if total_w > 0.0:
                    r = random.random() * total_w
                    cum = 0.0
                    picked_candidate = None

                    # Iterate through valid_candidates to match logic
                    # (Your previous loop iterated weights, but we need to map back to indices)
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            # We found our winner
                            # We need to find which candidate this corresponds to
                            # Since we appended "Stay" (index -1) first, or last, handle carefuly.
                            # Better to just use the valid_candidates list directly:
                            picked_candidate = valid_candidates[i] 
                            break
                    
                    # 5) Execute Move (if not staying)
                    if picked_candidate['index_in_batch'] != -1:
                        # We picked a neighbor, so we update
                        batch_idx = picked_candidate['index_in_batch']
                        current_spin_vector = candidates[batch_idx]
                        current_energy = measured_energies[batch_idx]
                        
                        accepted_idx_in_spins = candidate_indices[batch_idx]
                        
                        if verbose and (step % max(1, steps_per_temp // 5) == 0):
                             print(f"  Step {step+1}: Accepted energy {current_energy:.6f} (idx flipped: {accepted_idx_in_spins})")
                    else:
                        # We picked index -1, which means "Stay". 
                        # Do nothing to current_spin_vector
                        pass

                    # Track global best
                    if current_energy < min_energy:
                        min_energy = current_energy
                        min_spin_vector = np.copy(current_spin_vector)
                energy_trace.append(current_energy)

            end_temp_t = time.time()
            if verbose:
                print(f"  Temp iteration time: {end_temp_t - start_temp_t:.2f}s")
            temp *= cooling_rate

        total_elapsed = time.time() - anneal_start
        print("\nBatched NPP annealing finished.")
        print(f"Total elapsed time: {total_elapsed:.2f}s")
        
        # Final theoretical energy
        final_dot = np.dot(min_spin_vector, self.npp_weights)
        final_theory_energy = final_dot**2
        print(f"Best measured energy found: {min_energy:.6f}")
        print(f"Best spin vector sum (partition diff): {final_dot:.4f}")
        print(f"Best theoretical energy (sum^2): {final_theory_energy:.4f}")

        # plot trace
        plt.figure(figsize=(10, 6))
        plt.plot(energy_trace)
        plt.xlabel("Annealing Step (Batched)")
        plt.ylabel("Measured Energy (Raw Photodiode)")
        plt.title("Photonic Annealing (NPP) Energy Trace")
        plt.show(block=False)

        return min_spin_vector, min_energy
import time
import numpy as np
import math
import random

class RobustDynamicNPPAnnealer(NumberPartitioningAnnealer):
    """
    A robust implementation of the NPP Annealer that supports Dynamic Cluster Flipping.
    
    FIXES:
    - Bypasses the Base Class Queue system (which causes deadlocks in sequential modes).
    - Uses a dedicated, pre-allocated buffer for high-speed single-shot evaluation.
    - Implements the 'Adaptive Polychromatic' cluster sizing logic.
    """

    def __init__(self, npp_weights: np.ndarray, beam_sigma_x: float, beam_sigma_y: float, 
                 serial_port: str = 'COM21', serial_baud: int = 115200, use_uint8: bool = True):
        
        super().__init__(
            npp_weights=npp_weights, 
            beam_sigma_x=beam_sigma_x, 
            beam_sigma_y=beam_sigma_y,
            serial_port=serial_port, 
            serial_baud=serial_baud
        )
        self.use_uint8 = use_uint8
        
        # --- DEDICATED RESOURCES FOR DIRECT DRIVE ---
        # We bypass the queue system to prevent "timed out waiting for free buffer" errors
        dtype = np.uint8 if self.use_uint8 else np.float32
        self._direct_buffer = np.zeros((self.slm_height, self.slm_width), dtype=dtype)
        self._direct_handle = None 
        
        # Pre-cache the phases for +1 and -1 spins for speed
        self._phase_map = {
            1: np.float32(np.pi/2),
            -1: np.float32(3*np.pi/2)
        }        
        # --- Profiling Storage ---
        # We store the last 1000 samples to keep memory low but get good averages
        self.timings = {
            'mask_creation': [], # Numba fill time
            'slm_upload': [],    # USB transfer + Show command
            'photodiode': [],    # Settle + Serial Read
            'total_step': []     # Full Metropolis iteration
        }
        
        # Dedicated buffer resources (same as before)
        dtype = np.uint8 if self.use_uint8 else np.float32
        self._direct_buffer = np.zeros((self.slm_height, self.slm_width), dtype=dtype)
        self._direct_handle = None 

    def evaluate_energy_direct(self, spin_vector: np.ndarray, settle_time: float = 0.002) -> float:
        """
        Evaluates energy and profiles the hardware interaction steps.
        """
        # --- TIMER START: Mask Creation ---
        t0 = time.perf_counter()
        
        # 1. Hardware Sanity Check
        if self.slm is None:
            return 100.0 + np.random.randn()

        # 2. Prepare Data
        spin_phases = np.where(spin_vector == 1, np.pi/2, 3*np.pi/2).astype(np.float32)
        alpha_col = self._alpha_matrix[:, 0]

        # 3. Fill Buffer
        if self.use_uint8:
            fill_mask_uint8(
                self._direct_buffer, self._base_checkerboard,
                alpha_col, spin_phases,
                self._spin_y0, self._spin_x0,
                self._macro_pix_y, self._macro_pix_x
            )
        else:
            fill_mask_blocks_numba(
                self._direct_buffer, self._base_checkerboard,
                alpha_col, spin_phases,
                self._spin_y0, self._spin_x0,
                self._macro_pix_y, self._macro_pix_x
            )
        
        # --- TIMER SPLIT: Mask Creation Done ---
        t1 = time.perf_counter()
        self.timings['mask_creation'].append(t1 - t0)

        # 4. Upload & Show
        if self._direct_handle is not None:
            self._direct_handle.release()
            self._direct_handle = None
            
        self._direct_handle = self._safe_load_phase_data(self._direct_buffer)
        
        if self._direct_handle:
            self._direct_handle.show() 
            pass
        else:
            self.slm.showPhaseData(self._direct_buffer)
            pass

        # Wait for VSync/Settle
        if self.sync_slm:
            try:
                self.slm.waitForLastFrameDisplayed()
            except:
                time.sleep(0.008)
                print("Warning: SLM waitForLastFrameDisplayed() failed, using fixed sleep.")
        else:
            # time.sleep(0.006)
            pass

        # --- TIMER SPLIT: SLM Upload/Show Done ---
        t2 = time.perf_counter()
        self.timings['slm_upload'].append(t2 - t1)

        # 5. Measure (Photodiode)
        # time.sleep(settle_time)
        
        val = 0.0
        with self._pd_lock:
            if self.ser is None or not self.ser.is_open:
                val = (np.dot(spin_vector, self.npp_weights)**2) * 0.0001
                print("Warning: Serial port not open, returning theoretical value with noise.")
            else:
                print("entering photodiode measurement loop")
                for i in range(100):
                    energy = self._last_photodiode_val
                    val += energy
                    print(val)
                    i+=1
                    time.sleep(0.01)  # Small delay to avoid overwhelming the serial port

        # --- TIMER SPLIT: Measurement Done ---
        t3 = time.perf_counter()
        self.timings['photodiode'].append(t3 - t2)
                
        return val/80

    def run_annealing(self, 
                      initial_temp: float, 
                      final_temp: float, 
                      cooling_rate: float, 
                      steps_per_temp: int, 
                      Nc0: int = None, 
                      verbose: bool = True):
        
        # Reset timings
        for k in self.timings: self.timings[k] = []

        if Nc0 is None: Nc0 = max(1, int(self.num_spins * 0.15))
        
        current_spins = np.random.choice([-1, 1], size=self.num_spins)
        current_energy = self.evaluate_energy_direct(current_spins)
        
        E0 = current_energy if current_energy > 1e-6 else 1.0 
        best_spins = current_spins.copy()
        best_energy = current_energy
        temp = initial_temp
        energy_trace = [current_energy]
        
        step_count = 0
        total_steps = int(math.log(final_temp / initial_temp) / math.log(cooling_rate)) * steps_per_temp
        
        print(f"--- Starting Annealing ({total_steps} steps expected) ---")

        t_run_start = time.perf_counter()

        while temp > final_temp:
            for _ in range(steps_per_temp):
                # --- TIMER START: Full Step ---
                t_step_start = time.perf_counter()
                
                step_count += 1
                
                # A. Dynamic Cluster Sizing
                
                ratio = current_energy / E0
                # E0 = current_energy
                if ratio < 0: ratio = 0
                factor = math.pow(ratio, 1/15)
                Nc = int(round(Nc0 * factor))
                Nc = max(1, min(Nc, self.num_spins))
                
                # B. Propose
                proposal = current_spins.copy()
                flip_indices = random.sample(range(self.num_spins), Nc)
                proposal[flip_indices] *= -1
                
                # C. Evaluate
                new_energy = self.evaluate_energy_direct(proposal)
                
                # D. Accept/Reject
                delta_E = new_energy - current_energy
                if new_energy < current_energy:
                    accepted = True
                else:
                    prob = math.exp(-delta_E / temp)
                    accepted = random.random() < prob
                    
                if accepted:
                    current_spins = proposal
                    current_energy = new_energy
                    print(f" Step {step_count}: Accepted new energy {current_energy:.6f} (Nc={Nc})")
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_spins = current_spins.copy()

                # --- TIMER END: Full Step ---
                t_step_end = time.perf_counter()
                self.timings['total_step'].append(t_step_end - t_step_start)

            energy_trace.append(current_energy)
            temp *= cooling_rate
            
            if verbose and step_count % (steps_per_temp * 5) == 0:
                # Print Rolling Averages
                avg_step = np.mean(self.timings['total_step'][-50:]) * 1000
                avg_disp = np.mean(self.timings['slm_upload'][-50:]) * 1000
                print(f"Step {step_count} | T={temp:.4f} | E={current_energy:.4f} | "
                      f"StepTime: {avg_step:.1f}ms | DispTime: {avg_disp:.1f}ms | Nc = {Nc}" )

        t_run_end = time.perf_counter()
        total_run_time = t_run_end - t_run_start
        print(f"\n--- Annealing Complete in {total_run_time:.2f}s ---")

        return best_spins, best_energy, energy_trace
    

class ThreadedDynamicNPPAnnealer(RobustDynamicNPPAnnealer):
    """
    NPP Annealer that uses the SAME measurement technique as MaxCutAnnealer.
    
    1. Background Thread: RUNNING (continuously updates self._last_photodiode_val)
    2. Measurement: Waits for settle_time, then grabs the variable.
    3. Queues: Bypassed (Direct Buffer) for speed, but PD logic is threaded.
    """

    def __init__(self, *args, **kwargs):
        # Initialize Base (RobustDynamicNPPAnnealer)
        # This normally sets up the buffers and layout
        super().__init__(*args, **kwargs)
        
        # Explicitly Ensure Thread is STARTED (Just like MaxCut)
        self.start_pd_thread()
        print(" [ThreadedAnnealer] Background PD thread is ACTIVE.")

    def evaluate_energy_direct(self, spin_vector: np.ndarray, settle_time: float = 0.005) -> float:
        """
        Max-Cut Style Measurement:
        1. Show Mask.
        2. Sleep (Settle).
        3. Read self._last_photodiode_val (updated by background thread).
        """
        # --- 1. MASK GENERATION ---
        # (Same optimized Numba/Direct Buffer code as before)
        spin_phases = np.where(spin_vector == 1, np.pi/2, 3*np.pi/2).astype(np.float32)
        alpha_col = self._alpha_matrix[:, 0]

        if self.use_uint8:
            fill_mask_uint8(self._direct_buffer, self._base_checkerboard,
                            alpha_col, spin_phases, self._spin_y0, self._spin_x0,
                            self._macro_pix_y, self._macro_pix_x)
        else:
            fill_mask_blocks_numba(self._direct_buffer, self._base_checkerboard,
                                   alpha_col, spin_phases, self._spin_y0, self._spin_x0,
                                   self._macro_pix_y, self._macro_pix_x)
        
        # --- 2. DISPLAY ---
        # Reuse handle for speed
        if self._direct_handle:
            self._direct_handle.release()
            self._direct_handle = None
        # self._direct_handle = self._safe_load_phase_data(self._direct_buffer)
        
        # if self._direct_handle:
        #     self._direct_handle.show()
        # else:
        #     self.slm.showPhaseData(self._direct_buffer)

        # # Hardware Sync (Wait for LC to physically change)
        # if self.sync_slm:
        #     try:
        #         self.slm.waitForLastFrameDisplayed()
        #     except:
        #         time.sleep(0.01) # Turbo fallback
        #         # print("Warning: SLM waitForLastFrameDisplayed() failed, using fixed sleep.")
        # else:
        #      # If sync is off, we MUST sleep enough for the SLM liquid crystals (approx 4-8ms)
        #      # plus the time it takes for the thread to catch the next serial packet.
        #      pass

        # --- 3. MEASURE (The MaxCut Technique) ---
        
        # A. Wait for Photodiode to react AND Serial Thread to pick it up
        # This is the most critical parameter. 
        # It needs to be: SLM_Response_Time + Serial_Latency
        time.sleep(settle_time)
        
        val = 0.0
        
        # B. Simply read the variable that the thread is updating
        with self._pd_lock:
            if self.ser is None or not self.ser.is_open:
                 # Simulation fallback
                 val = (np.dot(spin_vector, self.npp_weights)**2) * 0.0001
                 print("Warning: Serial port not open, returning theoretical value with noise.")
            else:
                 print("Reading photodiode value from background thread...")
                 for i in range(150):
                     energy = self._last_photodiode_val
                     val += energy
                    #  print(val)
                     i+=1
                 print("Finished reading photodiode values.", val)

        return val
# -----------------------------------------------------------------
# 6. Main Execution
# -----------------------------------------------------------------

def run_maxcut_benchmark():
    print("\n" + "="*50)
    print("RUNNING MAX-CUT BENCHMARK")
    print("="*50)
    
    # --- 1. Define Problem ---
    NUM_SPINS = 20 # Smaller for faster testing
    np.random.seed(42)
    J_random = np.random.randn(NUM_SPINS, NUM_SPINS)
    J_random = (J_random + J_random.T) / 2 # Symmetrize
    
    # --- 2. Define Experimental Parameters ---
    BEAM_SIGMA_X = 600
    BEAM_SIGMA_Y = 600
    SERIAL_PORT = 'COM21' # [USER] Verify this port
    
    # --- 3. Define Annealing Parameters ---
    INITIAL_TEMP = 1000.0
    FINAL_TEMP = 0.1
    COOLING_RATE = 0.90
    STEPS_PER_TEMP = 5 # Set low for testing

    annealer = None
    try:
        annealer = MaxCutAnnealer(
            J=J_random,
            beam_sigma_x=BEAM_SIGMA_X,
            beam_sigma_y=BEAM_SIGMA_Y,
            serial_port=SERIAL_PORT
        )
        
        final_spins, min_energy, energy_data = annealer.run_annealing(
            initial_temp=INITIAL_TEMP,
            final_temp=FINAL_TEMP,
            cooling_rate=COOLING_RATE,
            steps_per_temp=STEPS_PER_TEMP
        )
        
        print("\n--- Max-Cut Results ---")
        print(f"Final Spin Configuration: {final_spins}")
        print(f"Minimum Energy (weighted sum): {min_energy:.4f}")

    except Exception as e:
        print(f"\nAn error occurred during Max-Cut benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if annealer is not None:
            annealer.disconnect_hardware()

def run_npp_benchmark():
    print("\n" + "="*50)
    print("RUNNING NUMBER PARTITIONING BENCHMARK")
    print("="*50)

    # --- 1. Define Problem ---
    NUM_SPINS = 40
    np.random.seed(123)
    # Integers from 1 to 100
    weights = np.random.randint(1, 101, size=NUM_SPINS).astype(float)
    
    # --- 2. Define Experimental Parameters ---
    BEAM_SIGMA_X = 600
    BEAM_SIGMA_Y = 600
    SERIAL_PORT = 'COM21' # [USER] Verify this port
    
    # --- 3. Define Annealing Parameters ---
    INITIAL_TEMP = 10.0 # NPP "energy" (raw PD) has different scale
    FINAL_TEMP = 0.001
    COOLING_RATE = 0.95
    STEPS_PER_TEMP = 10
    BATCH_SIZE = 10 # Use batching

    annealer = None
    try:
        annealer = NumberPartitioningAnnealer(
            npp_weights=weights,
            beam_sigma_x=BEAM_SIGMA_X,
            beam_sigma_y=BEAM_SIGMA_Y,
            serial_port=SERIAL_PORT
        )
        
        final_spins, min_energy, energy_data = annealer.run_annealing(
            initial_temp=INITIAL_TEMP,
            final_temp=FINAL_TEMP,
            cooling_rate=COOLING_RATE,
            steps_per_temp=STEPS_PER_TEMP,
            batch_size=BATCH_SIZE
        )
        
        print("\n--- NPP Results ---")
        partition_diff = np.dot(final_spins, weights)
        print(f"Final Spin Configuration: {final_spins}")
        print(f"Minimum Energy (raw photodiode): {min_energy:.6f}")
        print(f"Final Partition Difference (Sum): {partition_diff:.2f}")


    except Exception as e:
        print(f"\nAn error occurred during NPP benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if annealer is not None:
            annealer.disconnect_hardware()
            
import numpy as np
import copy
import diffractsim
from diffractsim import MonochromaticField, Lens, mm, nm, cm

# 1. Force Backend to CUDA
diffractsim.set_backend("CUDA")

class SimulatedMaxCutAnnealer(MaxCutAnnealer):
    def __init__(self, focal_length_cm=50, pixel_pitch_um=8.0, *args, **kwargs):
        self.focal_length = focal_length_cm * cm
        self.pixel_pitch = pixel_pitch_um * 1e-6
        self.sim_field = None
        self.cached_E_gpu = None  # We will cache the raw GPU array here
        super().__init__(*args, **kwargs)
        self.sync_slm = True 

    def _connect_hardware(self):
        print(" [Simulation] Initializing High-Res Optical Simulation...")
        
        # Define 1920x1080 Grid
        self.slm_width = 1920
        self.slm_height = 1080
        extent_x = self.slm_width * self.pixel_pitch
        extent_y = self.slm_height * self.pixel_pitch
        
        # 1. Initialize Field (Allocated on GPU by diffractsim)
        self.sim_field = MonochromaticField(
            wavelength=632.8 * nm, extent_x=extent_x, extent_y=extent_y,
            Nx=self.slm_width, Ny=self.slm_height
        )
        
        # 2. Generate Amplitude Mask on CPU (NumPy) - Safe
        x_np = np.linspace(-extent_x/2, extent_x/2, self.slm_width)
        y_np = np.linspace(-extent_y/2, extent_y/2, self.slm_height)
        X_np, Y_np = np.meshgrid(x_np, y_np)
        
        sigma_phys_x = self.beam_sigma_x * self.pixel_pitch
        sigma_phys_y = self.beam_sigma_y * self.pixel_pitch
        
        amplitude_mask_cpu = np.exp(-((X_np**2)/(2*sigma_phys_x**2) + (Y_np**2)/(2*sigma_phys_y**2)))
        
        # 3. Explicit Transfer and Setup
        if hasattr(self.sim_field.E, 'device'):
            import cupy as cp
            print(" [Simulation] Backend: CUDA. Moving data to GPU...")
            
            # Move mask to GPU
            amplitude_mask_gpu = cp.asarray(amplitude_mask_cpu)
            
            # Apply Mask (Assignment, not in-place, to be safe)
            self.sim_field.E = self.sim_field.E * amplitude_mask_gpu
            
            # CRITICAL: Explicitly cast to COMPLEX128 and cache THIS array
            # This ensures our "clean state" is definitely complex-ready.
            self.cached_E_gpu = self.sim_field.E.astype(cp.complex128)
            
            # Set the field to this complex version
            self.sim_field.E = self.cached_E_gpu.copy()
        else:
            # Fallback for CPU
            print(" [Simulation] Backend: CPU (Warning: Slow).")
            self.sim_field.E = self.sim_field.E * amplitude_mask_cpu
            self.cached_E_gpu = self.sim_field.E.astype(np.complex128)
            self.sim_field.E = self.cached_E_gpu.copy()
        
        # Mock hardware variables
        self.ser = True 
        self._last_photodiode_val = 0.0
        self._center_x = self.slm_width / 2
        self._center_y = self.slm_height / 2

    def _show_handle_or_buffer(self, buf_idx: int, buf: np.ndarray, wait_for_frame: bool = True):
        # 1. Restore Field from Cache
        # We perform a copy so we don't modify the cached clean state
        self.sim_field.E = self.cached_E_gpu.copy()
        
        # 2. Check Device
        if hasattr(self.sim_field.E, 'device'):
            import cupy as cp
            
            # --- GPU PATH ---
            gpu_buf = cp.asarray(buf)
            
            if self.use_uint8:
                # Calculate phase values
                phase_vals = gpu_buf.astype(cp.float64) * (2 * cp.pi / 255.0)
            else:
                phase_vals = gpu_buf

            # Calculate Modulation Factor (Complex)
            modulation = cp.exp(1j * phase_vals)

            # CRITICAL FIX: Use Explicit Assignment (=) instead of In-Place (*=)
            # This allocates a new memory block for the result, avoiding the type error
            self.sim_field.E = self.sim_field.E * modulation
            
            # Propagate
            self.sim_field.add(Lens(f=self.focal_length))
            self.sim_field.propagate(self.focal_length)
            
            # Measure (Sum on GPU -> Float on CPU)
            I = self.sim_field.get_intensity()
            cy, cx = self.slm_height // 2, self.slm_width // 2
            pd_reading = float(cp.sum(I[cy-1:cy+2, cx-1:cx+2]))

        else:
            # --- CPU PATH ---
            if self.use_uint8:
                phase_vals = buf.astype(np.float64) * (2 * np.pi / 255.0)
            else:
                phase_vals = buf
            
            self.sim_field.E = self.sim_field.E * np.exp(1j * phase_vals)
            self.sim_field.add(Lens(f=self.focal_length))
            self.sim_field.propagate(self.focal_length)
            
            I = self.sim_field.get_intensity()
            cy, cx = self.slm_height // 2, self.slm_width // 2
            pd_reading = float(np.sum(I[cy-1:cy+2, cx-1:cx+2]))
        
        with self._pd_lock:
            self._last_photodiode_val = pd_reading
            
        return 0, False

    def start_pd_thread(self): pass
    def stop_pd_thread(self): pass
    def _safe_load_phase_data(self, buf): return buf 
    def _release_handle(self, handle): pass
    def disconnect_hardware(self): print(" [Simulation] Finished.")    

if __name__ == '__main__':
    
    # --- Run Benchmarks ---
    # Note: Each benchmark will connect and disconnect from hardware.
    
    def run_maxcut_benchmark():
        # ...
        try:
            annealer = MaxCutAnnealer(..., use_uint8=True)
            
            # --- ADD THIS LINE TO THE FILE ---
            print("Enabling Turbo Mode...")
            annealer.sync_slm = False 
            # ---------------------------------
            
            annealer.run_annealing(...)
        except Exception as e:
            pass
        finally:
            if annealer is not None:
                annealer.disconnect_hardware()
    
    print("\nPausing for 5 seconds before next benchmark...")
    time.sleep(5)
    
    run_npp_benchmark()

    print("\nAll benchmarks complete. Exiting.")
    
    # This is needed to close plot windows if they are non-blocking
    plt.show()