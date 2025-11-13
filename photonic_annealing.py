import sys
import os
import numpy as np
import random
import math
import time
import serial
import matplotlib.pyplot as plt
from typing import Tuple, List
import time
import struct
import serial
from typing import Optional

# --- [NEW] Imports for FIFO Streaming ---
import threading
import queue

# Numba
import numba as nb
import math

# -----------------------------------------------------------------
# 1. HOLOEYE SDK Setup
# -----------------------------------------------------------------

# [USER] PLEASE VERIFY THIS PATH
sdk_install_path = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"
python_api_path = os.path.join(sdk_install_path, "api", "python")

if python_api_path not in sys.path:
    sys.path.append(python_api_path)
    print(f"Added {python_api_path} to Python path.")

try:
    import HEDS
    from hedslib.heds_types import *
except ImportError:
    print(f"FATAL ERROR: Could not import HEDS library from {python_api_path}")
    print("Please verify the 'sdk_install_path' variable in this script.")
    sys.exit(1)

########### Numba functions ###########
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
# 2. The Merged Photonic Annealer Class
# # -----------------------------------------------------------------
# @numba.jit(nopython=True, cache=True)
# def _compute_one_mask_numba(
#     k: int,
#     phase_mask: np.ndarray,             # (H, W) float32, modified in-place
#     spin_phases: np.ndarray,            # (N,) float
#     compensation_factors: np.ndarray,   # (N,) float
#     eigvecs: np.ndarray,                # (N, N) float
#     base_checkerboard: np.ndarray,      # (MH, MW) float32
#     spin_x0: np.ndarray,                # (N,) int32
#     spin_y0: np.ndarray,                # (N,) int32
#     macro_pix_x: int,
#     macro_pix_y: int,
#     num_spins: int
# ):
#     """
#     [Numba-JITted] Computes a single phase mask for eigenvector k.
#     Modifies 'phase_mask' in-place.
#     """
    
#     # 1. Calculate Amplitudes
#     # Use explicit loops for Numba clarity
#     target_amplitudes = np.empty(num_spins, dtype=np.float64)
#     for i in range(num_spins):
#         target_amplitudes[i] = compensation_factors[i] * eigvecs[i, k]

#     # 2. Normalize
#     max_abs_val = 0.0
#     for i in range(num_spins):
#         val = np.abs(target_amplitudes[i])
#         if val > max_abs_val:
#             max_abs_val = val
    
#     if max_abs_val < 1e-9:
#         max_abs_val = 1.0

#     # 3. Calculate Alphas
#     alpha_ik = np.empty(num_spins, dtype=np.float64) 
#     for i in range(num_spins):
#         # Numba-safe clip
#         val = target_amplitudes[i] / max_abs_val
#         if val > 1.0:
#             val = 1.0
#         elif val < -1.0:
#             val = -1.0
#         alpha_ik[i] = np.arccos(val)

#     # 4. Fill the phase mask
#     two_pi = 2 * np.pi
#     for i in range(num_spins):
#         amplitude = alpha_ik[i]
#         spin_phase = spin_phases[i]
        
#         # Get slice coordinates
#         y_start = spin_y0[i]
#         x_start = spin_x0[i]
        
#         # Create the macropixel block
#         # Use explicit loops for assignment (safest in Numba)
#         for y_idx in range(macro_pix_y):
#             for x_idx in range(macro_pix_x):
#                 # Calculate the phase for this pixel
#                 val = spin_phase + (base_checkerboard[y_idx, x_idx] * amplitude)
                
#                 # Assign to the main mask with modulo
#                 phase_mask[y_start + y_idx, x_start + x_idx] = val % two_pi
    
#     # No return value is needed, phase_mask is modified in-place

class PhotonicAnnealer:
    """
    Manages the entire Photonic Ising Machine experiment.

    This class handles:
    - Hardware connection (SLM, Photodiode).
    - Mattis Hamiltonian decomposition and beam compensation.
    - High-performance, streaming (FIFO) phase mask generation.
    - The simulated annealing optimization loop.
    """
    
    def __init__(
        self,
        J: np.ndarray,
        beam_sigma_x: float,
        beam_sigma_y: float,
        serial_port: str = 'COM21',
        serial_baud: int = 115200,
        serial_timeout: float = 0.001,
    ):

        """
        Initializes the annealer and connects to hardware.

        Args:
            J (np.ndarray): The n_spins x n_spins interaction matrix.
            beam_sigma_x (float): Gaussian beam std dev in x.
            beam_sigma_y (float): Gaussian beam std dev in y.
            serial_port (str): The COM port for the Tiva microcontroller.
            serial_baud (int): The baud rate for the serial connection.
        """
        print("Initializing Photonic Annealer...")
        
        # --- Hardware Handles ---
        self.slm: HEDS.SLM = None
        self.ser: Optional[serial.Serial] = None
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.serial_timeout = serial_timeout

        
        # --- SLM Properties ---
        self.slm_width: int = 0
        self.slm_height: int = 0
        # --- Photodiode Reading Thread ---

        self._pd_thread = None
        self._pd_stop_event = threading.Event()
        self._pd_lock = threading.Lock()
        self._last_photodiode_val = 0.0        
        # --- Connect to Hardware ---
        self._connect_hardware()

        # --- Ising Model Properties ---
        if not isinstance(J, np.ndarray) or J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square 2D numpy array.")
        if not np.allclose(J, J.T):
            raise ValueError("Interaction matrix J is not symmetric.")
            
        self.J = J
        self.num_spins = J.shape[0]
        self.beam_sigma_x = beam_sigma_x
        self.beam_sigma_y = beam_sigma_y

        # --- FIFO Streaming Attributes ---
        self.mask_queue = queue.Queue(maxsize=5) # Max 5 masks in RAM
        self.producer_thread = None
        self.stop_producer_event = threading.Event()
        
        # --- Pre-computation Attributes ---
        self._center_x: float = self.slm_width / 2
        self._center_y: float = self.slm_height / 2
        self.eigvals: np.ndarray = None
        self.eigvecs: np.ndarray = None
        self._grid_rows: int = 0
        self._grid_cols: int = 0
        self._macro_pix_x: int = 0
        self._macro_pix_y: int = 0
        self._compensation_factors: np.ndarray = None
        self._base_checkerboard: np.ndarray = None
        self._spin_slices: List[Tuple[slice, slice]] = []
        self._spin_coords_x: np.ndarray = np.zeros(self.num_spins)
        self._spin_coords_y: np.ndarray = np.zeros(self.num_spins)

        # --- Run Full Preparation ---
        self.prep()
        # choose pool size (2 = double-buffering; 3 = safer for jitter)
        self._buffer_pool_size = 10

        # Pre-allocate buffers once SLM size is known (after _connect_hardware() & prep())
        self._mask_buffers = [np.zeros((self.slm_height, self.slm_width), dtype=np.float32)
                            for _ in range(self._buffer_pool_size)]
        
        self._reset_buffer_queues()
        # Queues to manage buffer indices
        self._free_buf_queue = queue.Queue()
        self._filled_buf_queue = queue.Queue()

        # Populate free queue with indices
        for i in range(self._buffer_pool_size):
            self._free_buf_queue.put(i)
        # --- Buffer / handle pool initialization (required) ---
        # choose buffer pool size earlier: self._buffer_pool_size already set
        # ensure a sane size if not
        try:
            self._buffer_pool_size = int(getattr(self, "_buffer_pool_size", max(10, (os.cpu_count() or 4) * 2)))
        except Exception:
            self._buffer_pool_size = max(10, (os.cpu_count() or 4) * 2)

        # Pre-allocate masks (already present in your file; this is safe if repeated)
        self._mask_buffers = [np.zeros((self.slm_height, self.slm_width), dtype=np.float32)
                            for _ in range(self._buffer_pool_size)]
        self._init_handle_pool()
        self._init_handle_pool()

        # create lock used whenever we touch handle pool
        self._handle_lock = threading.Lock()

        # create a handle pool aligned with mask buffers
        self._handle_pool = [None] * len(self._mask_buffers)

        # create and populate free/filled queues (safe reset)
        def _local_reset_queues():
            self._free_buf_queue = queue.Queue(maxsize=len(self._mask_buffers))
            self._filled_buf_queue = queue.Queue(maxsize=len(self._mask_buffers) + 2)
            for i in range(len(self._mask_buffers)):
                try:
                    self._free_buf_queue.put_nowait(i)
                except queue.Full:
                    break

        _local_reset_queues()

        # Optional: pre-upload empty buffers as handles (fast path for show-by-handle)
        try:
            # upload handles in background — this uses your safe wrapper
            self._upload_buffer_pool_as_handles()
        except Exception as e:
            print("Warning: initial upload of handle pool failed:", e)
        # ---------------------------
        # Insert into PhotonicAnnealer.__init__ AFTER you create self._mask_buffers
        # ---------------------------
        # (Place this line right after you create self._mask_buffers)
        
        # then upload initial buffers as handles (best-effort)
        self._upload_buffer_pool_as_handles()
        # ---------------------------

    # ---------------------------
    # New / replacement methods for handle management
    # ---------------------------

    def _init_handle_pool(self):
        """Create thread-safe structures for the handle pool."""
        # Called after self._mask_buffers exists and self._buffer_pool_size is set
        self._handle_lock = threading.Lock()
        # initialize handle pool to same length as buffers
        self._handle_pool = [None] * len(self._mask_buffers)

    def _safe_load_phase_data(self, buf: np.ndarray):
        """
        Upload 'buf' to the SLM and return an SLMDataHandle on success, or None on failure.
        Wraps SDK differences and common error cases.
        """
        if self.slm is None:
            return None

        try:
            # The SDK wrapper typically returns (err, SLMDataHandle)
            maybe = self.slm.loadPhaseData(buf)
        except Exception as e:
            # some wrappers may raise on internal error
            print("Producer: loadPhaseData exception:", e)
            return None

        # normalize wrapper return formats
        dh = None
        try:
            # common pattern: (err, data_handle) or (err, data_handle_id)
            if isinstance(maybe, tuple) or isinstance(maybe, list):
                if len(maybe) >= 2:
                    err = maybe[0]
                    cand = maybe[1]
                    # if the wrapper gave an SLMDataHandle object, use it
                    if hasattr(cand, "id") or hasattr(cand, "_handle_id") or cand is not None:
                        dh = cand
                    else:
                        dh = None
                elif len(maybe) == 1:
                    # maybe returned only handle
                    dh = maybe[0]
                else:
                    dh = None
            else:
                # some wrapper versions directly return an SLMDataHandle
                dh = maybe
        except Exception:
            dh = None

        # If we got a handle-like object, ensure it exposes expected API
        if dh is None:
            return None

        # If it's an SLMDataHandle object in this SDK, it should have applyErrorCode/errorCode etc.
        # We return the handle object for later usage.
        return dh

    def _release_handle(self, handle):
        """
        Try to release an SLM data-handle if SDK exposes a release/free function.
        Best-effort; swallowing exceptions keeps system robust.
        """
        if handle is None:
            return
        try:
            # Preferred: call object's release() if exposed in this wrapper
            rel = getattr(handle, "release", None)
            if callable(rel):
                try:
                    rel()
                    return
                except Exception:
                    pass

            # Some wrappers implement different names - try common ones
            rel2 = getattr(handle, "free", None)
            if callable(rel2):
                try:
                    rel2()
                    return
                except Exception:
                    pass

            # Fallback: try to extract low-level id and call SDK free api (best-effort)
            handle_id = None
            try:
                handle_id = handle.id()
            except Exception:
                # try internal attr if available
                handle_id = getattr(handle, "_handle_id", None)

            if handle_id is not None and hasattr(HEDS.SDK.libapi, "heds_datahandle_release"):
                try:
                    HEDS.SDK.libapi.heds_datahandle_release(handle_id)
                    return
                except Exception:
                    pass
        except Exception:
            pass
        # if nothing worked, just continue; Python GC + SDK often manages memory

    def _upload_buffer_pool_as_handles(self):
        """
        Upload every preallocated buffer in self._mask_buffers to obtain data handles.
        Populates/overwrites self._handle_pool. Thread-safe.
        """
        if not hasattr(self, "_handle_lock"):
            self._init_handle_pool()

        # Ensure pool has correct length
        with self._handle_lock:
            if not hasattr(self, "_handle_pool") or len(self._handle_pool) != len(self._mask_buffers):
                self._handle_pool = [None] * len(self._mask_buffers)

        for i, buf in enumerate(self._mask_buffers):
            dh = None
            try:
                dh = self._safe_load_phase_data(buf)
            except Exception as e:
                print("Upload handle: exception while loading buffer:", e)
                dh = None

            with self._handle_lock:
                prev = None
                try:
                    prev = self._handle_pool[i]
                except Exception:
                    prev = None
                self._handle_pool[i] = dh

            # release previous handle if replaced
            if prev is not None and prev is not dh:
                try:
                    self._release_handle(prev)
                except Exception:
                    pass

    def _show_handle_or_buffer(self, buf_idx: int, buf: np.ndarray = None, wait_for_frame: bool = True):
        """
        Show mask referenced by buf_idx using stored data-handle if available.
        If handle-show fails, falls back to showing the buffer.
        Returns (err_code, used_handle_flag).

        Strategy:
        - If we have an SLMDataHandle object in _handle_pool[buf_idx], ask the SDK to show it via
            the helper HEDS.ShowDataHandles([dh]) (this builds the low-level id list properly).
        - On any failure, fall back to self.slm.showPhaseData(buf).
        """
        err = HEDSERR_NoError
        used_handle = False

        # get handle (thread-safe)
        handle = None
        try:
            with self._handle_lock:
                if hasattr(self, "_handle_pool") and buf_idx < len(self._handle_pool):
                    handle = self._handle_pool[buf_idx]
        except Exception:
            handle = None

        # prefer handle-based show if we have a valid handle-like object
        if handle is not None:
            try:
                # HEDS.ShowDataHandles expects a list of handles (SLMDataHandle or low-level id)
                # this helper constructs the right id array and calls the low-level heds_datahandles_show
                ret = HEDS.ShowDataHandles([handle])
                if int(ret) == 0:  # HEDSERR_NoError
                    used_handle = True
                    err = int(ret)
                else:
                    # handle-show returned an error code; we'll fallback to buffer
                    print("Warning: ShowDataHandles returned error:", HEDS.SDK.ErrorString(ret) if hasattr(HEDS.SDK, 'ErrorString') else ret)
                    used_handle = False
            except Exception as e:
                # Some wrapper versions or handles may throw (e.g. unexpected internal representation).
                # Fall back to showing buffer below.
                # Print a concise warning for debugging.
                print("Warning: ShowDataHandles(handle) raised exception, falling back to buffer:", e)
                used_handle = False

        # fallback: show the raw buffer data (most reliable)
        if not used_handle:
            if buf is None:
                try:
                    buf = self._mask_buffers[buf_idx]
                except Exception:
                    msg = "Error: no buffer available for buf_idx {}".format(buf_idx)
                    print(msg)
                    return (HEDSERR_GeneralError if 'HEDSERR_GeneralError' in globals() else -1, False)

            try:
                # Use background flag if available in wrapper; otherwise call normal show
                if getattr(HEDS, 'HEDSSlmShowPhaseFlags', None) is not None:
                    try:
                        flags = HEDS.HEDSSlmShowPhaseFlags.SHOW_IN_BACKGROUND
                        err = self.slm.showPhaseData(buf, flags)
                    except Exception:
                        # fallback to basic call
                        err = self.slm.showPhaseData(buf)
                else:
                    err = self.slm.showPhaseData(buf)

                # If the SDK exposes a 'wait' function and we want to wait, call it:
                if wait_for_frame:
                    wait_fn = getattr(self.slm, 'waitForLastFrameDisplayed', None) or getattr(self.slm, 'wait_for_frame', None)
                    if callable(wait_fn):
                        try:
                            wait_fn()
                        except Exception:
                            pass

            except Exception as e:
                print("Error showing buffer via showPhaseData:", e)
                return (HEDSERR_GeneralError if 'HEDSERR_GeneralError' in globals() else -1, False)

        return (int(err) if err is not None else HEDSERR_NoError, used_handle)

    def _clear_handle_pool(self):
        """Release & clear all stored data-handles (best-effort)."""
        if not hasattr(self, "_handle_pool"):
            return
        # Use the lock to avoid races with producer/consumer threads
        with getattr(self, "_handle_lock", threading.Lock()):
            for i, h in enumerate(self._handle_pool):
                if h is not None:
                    try:
                        self._release_handle(h)
                    except Exception:
                        pass
                    self._handle_pool[i] = None

        # -------------------------------------------------------------------



    # ---------------------------------------------------
    # 1. Hardware Connection & Control
    # ---------------------------------------------------

    def _connect_hardware(self):
        """Initializes and connects to the HEDS SLM and Tiva Serial Port."""
        print("Connecting to hardware...")

        # 1. Initialize SDK
        err = HEDS.SDK.Init(4, 1)
        if err != HEDSERR_NoError:
            raise RuntimeError(f"Error initializing SDK: {HEDS.SDK.ErrorString(err)}")

        # 2. Initialize SLM (this may open the GUI)
        
        self.slm = HEDS.SLM.Init()
        if self.slm.errorCode() != HEDSERR_NoError:
            raise RuntimeError(f"Error initializing SLM: {HEDS.SDK.ErrorString(self.slm.errorCode())}")

        self.slm_width = self.slm.width_px()
        self.slm_height = self.slm.height_px()
        print(f"SLM connected. Resolution: {self.slm_width} x {self.slm_height}")

        # 3. Connect to Serial Port
        try:
            # Use serial_timeout (0.0 => non-blocking) for fastest read behavior.
            self.ser = serial.Serial(port=self.serial_port, baudrate=self.serial_baud, timeout=self.serial_timeout)
            # Wait for device to reset / settle
            time.sleep(1.0)
            # Clear any buffered input
            try:
                self.ser.reset_input_buffer()
            except AttributeError:
                # older pyserial
                self.ser.flushInput()
            if self.ser.is_open:
                print(f"Serial port {self.serial_port} opened at {self.serial_baud} baud (timeout={self.serial_timeout}).")
            else:
                raise RuntimeError(f"Serial port {self.serial_port} failed to open.")
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to connect to serial port {self.serial_port} at {self.serial_baud} baud: {e}")
        self.start_pd_thread()

    def disconnect_hardware(self):
        """Safely disconnects from SLM and Serial port."""
        # print("\nDisconnecting hardware...")
        # if self.slm is not None:
        #     self.slm.showBlankScreen(0) # Blank the SLM
        #     err = self.slm.window().close()
        #     if err == HEDSERR_NoError:
        #         print("SLM window closed.")
        #     HEDS.SDK.Exit()
        #     print("HEDS SDK closed.")
        
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.serial_port} closed.")
        self.stop_pd_thread()
        self._clear_handle_pool()



    def _pd_reader_worker(self):
        """Continuously read frames from serial and update self._last_photodiode_val."""
        ser = getattr(self, "ser", None)
        if ser is None or not ser.is_open:
            return
        buf = bytearray()
        FRAME_SIZE = 3
        START_BYTE = 0xA5
        while not self._pd_stop_event.is_set():
            try:
                chunk = ser.read(512)  # non-blocking or very short timeout
            except Exception:
                time.sleep(0.0005)
                continue
            if not chunk:
                # yield CPU briefly
                time.sleep(0.0002)
                continue
            buf.extend(chunk)
            i = 0
            updated = False
            while i + FRAME_SIZE <= len(buf):
                if buf[i] != START_BYTE:
                    i += 1
                    continue
                lo = buf[i+1]
                hi = buf[i+2]
                val = lo | (hi << 8)
                # convert to voltage if desired here (cheap math)
                with self._pd_lock:
                    self._last_photodiode_val = (val / ((1 << 12) - 1)) * 3.3  # change adc params if needed
                updated = True
                i += FRAME_SIZE
            if i > 0:
                del buf[:i]
            if not updated:
                # allow small sleep to avoid busy spin
                time.sleep(0.002)

    def start_pd_thread(self):
        if self._pd_thread is not None and self._pd_thread.is_alive():
            return
        self._pd_stop_event.clear()
        self._pd_thread = threading.Thread(target=self._pd_reader_worker, daemon=True)
        self._pd_thread.start()

    def stop_pd_thread(self):
        if self._pd_thread is None:
            return
        self._pd_stop_event.set()
        self._pd_thread.join(timeout=1.0)
        self._pd_thread = None

    def _display_phase_mask(self, phase_mask_2d: np.ndarray) -> bool:
        """(Internal) Displays a 2D NumPy array on the SLM."""
        if not isinstance(phase_mask_2d, np.ndarray):
            print("Error: The provided phase mask must be a NumPy array.")
            return False
        
        # Convert to float32 for the SDK
        phase_mask_float32 = phase_mask_2d.astype(np.float32)
        
        error = self.slm.showPhaseData(phase_mask_float32)

        if error != HEDSERR_NoError:
            print(f"Error displaying phase mask: {HEDS.SDK.ErrorString(error)}")
            return False
        return True


    def open_serial(port: str, baud: int = 921600) -> serial.Serial:
        # timeout=0 makes read non-blocking (returns immediately with whatever's available)
        # You can use a tiny timeout like 0.01 to yield CPU if you prefer.
        return serial.Serial(port=port, baudrate=baud, timeout=0)

    def _photodiode_measurement(self, duration: float = 0.05,
                                adc_ref: Optional[float] = 3.3,
                                adc_bits: int = 12):
        """
        Reads binary frames [0xA5][lo][hi] from self.ser (non-blocking).
        Returns averaged voltage (if adc_ref) or average raw counts.
        """
        ser = getattr(self, "ser", None)
        if ser is None or not ser.is_open:
            raise RuntimeError("Serial port not open. Call _connect_hardware() first.")

        # Flush any stale data
        try:
            ser.reset_input_buffer()
        except AttributeError:
            ser.flushInput()

        buf = bytearray()
        start_time = time.perf_counter()
        sample_sum = 0
        sample_count = 0
        FRAME_SIZE = 3
        START_BYTE = 0xA5
        read_chunk_size = 512  # enough for 0.05 s @115200

        while time.perf_counter() - start_time < duration:
            chunk = ser.read(read_chunk_size)
            if not chunk:
                time.sleep(0.0005)
                continue
            buf.extend(chunk)
            i = 0
            while i + FRAME_SIZE <= len(buf):
                if buf[i] != START_BYTE:
                    i += 1
                    continue
                lo = buf[i + 1]
                hi = buf[i + 2]
                val = lo | (hi << 8)
                sample_sum += val
                sample_count += 1
                i += FRAME_SIZE
            if i > 0:
                del buf[:i]

        if sample_count == 0:
            print("Warning: No photodiode data received.")
            return getattr(self, "_last_photodiode_val", 0.0)

        avg_count = sample_sum / sample_count
        if adc_ref is not None:
            val = (avg_count / ((1 << adc_bits) - 1)) * adc_ref
        else:
            val = avg_count
        self._last_photodiode_val = val
        return val
    
    def _safe_load_phase_data(self, buf: np.ndarray):
        """
        Upload 'buf' to the SLM and return a data-handle, or None on failure.
        Wraps SDK differences and common error cases.
        """
        try:
            maybe = self.slm.loadPhaseData(buf)
        except Exception as e:
            # SDK sometimes raises; log and return None
            print("Producer: loadPhaseData exception:", e)
            return None

        # wrapper may return (err, handle) or handle directly
        dh = None
        if isinstance(maybe, (tuple, list)):
            # common wrapper pattern: (err, handle) or (err,)
            if len(maybe) >= 2:
                err = maybe[0]
                cand = maybe[1]
                if err == HEDSERR_NoError:
                    dh = cand
                else:
                    # print readable error string if possible
                    try:
                        print("Producer: loadPhaseData err:", HEDS.SDK.ErrorString(err))
                    except Exception:
                        print("Producer: loadPhaseData returned error code", err)
                    dh = None
            elif len(maybe) == 1:
                dh = maybe[0]
            else:
                dh = None
        else:
            # returned the handle directly
            dh = maybe

        # sometimes the handle isn't the expected type; check lightly
        if dh is None:
            return None
        return dh

    def _release_handle(self, handle):
        """
        Try to release an SLM data-handle if SDK exposes a free/release function.
        It's safe to call even if SDK doesn't support it.
        """
        try:
            # try likely API names (depending on HEDS version)
            if hasattr(self.slm, "releasePhaseDataHandle"):
                self.slm.releasePhaseDataHandle(handle)
            elif hasattr(self.slm, "freeDataHandle"):
                self.slm.freeDataHandle(handle)
            elif hasattr(HEDS.SDK, "FreeDataHandle"):
                # sometimes in SDK root
                HEDS.SDK.FreeDataHandle(handle)
            else:
                # no-op; SDK might manage handles automatically
                pass
        except Exception:
            # ignore errors when releasing (we prefer continuing)
            pass

    def _upload_buffer_pool_as_handles(self):
        """
        Upload every preallocated buffer in self._mask_buffers to obtain data handles.
        This populates/overwrites self._handle_pool (same length as buffer pool).
        Safe to call whenever buffers are (re)created.
        """
        # ensure container + lock exist
        if not hasattr(self, "_handle_lock"):
            self._handle_lock = threading.Lock()
        if not hasattr(self, "_handle_pool") or len(self._handle_pool) != len(self._mask_buffers):
            self._handle_pool = [None] * len(self._mask_buffers)

        for i, buf in enumerate(self._mask_buffers):
            dh = self._safe_load_phase_data(buf)
            with self._handle_lock:
                # if there was an old handle, release it
                prev = None
                try:
                    prev = self._handle_pool[i]
                except Exception:
                    prev = None
                self._handle_pool[i] = dh
            if prev is not None and prev is not dh:
                try:
                    self._release_handle(prev)
                except Exception:
                    pass

    def _show_handle_or_buffer(self, buf_idx: int, buf: np.ndarray = None, wait_for_frame: bool = True):
        """
        Show the mask referenced by buf_idx using a stored data-handle if available.
        If no handle exists or show-by-handle fails, falls back to showing the buffer.
        Returns (err_code, used_handle_flag).
        - buf_idx: index into self._mask_buffers / self._handle_pool
        - buf: optional direct buffer (if consumer already has it)
        - wait_for_frame: if True, try to call available wait function after show
        """
        err = HEDSERR_NoError
        used_handle = False
        # choose best show method
        handle = None
        try:
            with self._handle_lock:
                if hasattr(self, "_handle_pool") and buf_idx < len(self._handle_pool):
                    handle = self._handle_pool[buf_idx]
        except Exception:
            handle = None

        # If handle exists, try showing by handle
        if handle is not None:
            try:
                err = self.slm.showPhaseData(handle)
                if err == HEDSERR_NoError:
                    used_handle = True
                else:
                    # attempt fallback to buffer if handle-show returns error
                    try:
                        # handle-show failed; print human-friendly message
                        print("Warning: showPhaseData(handle) failed:", HEDS.SDK.ErrorString(err))
                    except Exception:
                        print("Warning: showPhaseData(handle) failed with code", err)
                    used_handle = False
                # if handle-show succeeded, optionally wait for frame
                if used_handle and wait_for_frame:
                    wait_fn = getattr(self.slm, 'waitForLastFrameDisplayed', None) or getattr(self.slm, 'wait_for_frame', None)
                    if callable(wait_fn):
                        try:
                            wait_fn()
                        except Exception:
                            pass
                    # else SDK might be synchronous for handle-show
            except Exception as e:
                # fallback to showing buffer on any exception
                # print("Warning: showPhaseData(handle) raised exception, falling back to buffer:", e)
                used_handle = False

        # If no handle or handle failed, fall back to show buffer
        if not used_handle:
            # ensure we have a buffer to show
            if buf is None:
                try:
                    buf = self._mask_buffers[buf_idx]
                except Exception:
                    print("Error: no buffer available for buf_idx", buf_idx)
                    return (HEDSERR_GeneralError if 'HEDSERR_GeneralError' in globals() else -1, False)
            try:
                # Use background flag if available
                if getattr(HEDS, 'HEDSSlmShowPhaseFlags', None) is not None:
                    try:
                        flags = HEDS.HEDSSlmShowPhaseFlags.SHOW_IN_BACKGROUND
                        err = self.slm.showPhaseData(buf, flags)
                    except Exception:
                        # fallback to normal call
                        err = self.slm.showPhaseData(buf)
                else:
                    err = self.slm.showPhaseData(buf)
                # optionally wait
                if wait_for_frame:
                    wait_fn = getattr(self.slm, 'waitForLastFrameDisplayed', None) or getattr(self.slm, 'wait_for_frame', None)
                    if callable(wait_fn):
                        try:
                            wait_fn()
                        except Exception:
                            pass
            except Exception as e:
                print("Error showing buffer via showPhaseData:", e)
                return (HEDSERR_GeneralError if 'HEDSERR_GeneralError' in globals() else -1, False)

        return (err, used_handle)
    def _clear_handle_pool(self):
        """Release & clear all stored data-handles (best-effort)."""
        if not hasattr(self, "_handle_pool"):
            return
        with getattr(self, "_handle_lock", threading.Lock()):
            for i, h in enumerate(self._handle_pool):
                if h is not None:
                    try:
                        self._release_handle(h)
                    except Exception:
                        pass
                    self._handle_pool[i] = None

    def _producer_worker_handles(self, spin_vector: np.ndarray, batch_size: int = 8):
        """
        Producer that fills buffers, uploads them (to get handles), and places (buf_idx, ready_time)
        into the filled queue. This version is robust to filled-queue back-pressure and
        returns buffers to free pool if the filled queue is full.
        """
        try:
            # precompute spin phases once
            spin_phases = np.where(spin_vector == 1, np.pi/2, 3*np.pi/2).astype(np.float32)
            n = self.num_spins
            for k in range(n):
                if self.stop_producer_event.is_set():
                    break

                # Wait longer for free buffer (avoid immediate timeout)
                try:
                    buf_idx = self._free_buf_queue.get(timeout=5.0)
                except queue.Empty:
                    print("Producer: timed out waiting for free buffer (get). Retrying or exiting.")
                    # signal consumer & exit gracefully
                    break

                buf = self._mask_buffers[buf_idx]

                # timing for diagnostics
                t_fill_start = time.perf_counter()
                try:
                    alpha_ik = self._alpha_matrix[:, k]
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
                except Exception as e:
                    print(f"Producer: fill failed at k={k}:", e)
                    # return buffer to free pool and continue
                    try:
                        self._free_buf_queue.put_nowait(buf_idx)
                    except Exception:
                        pass
                    continue
                t_fill_end = time.perf_counter()

                # Upload to SDK (safe wrapper)
                t_upload_start = time.perf_counter()
                dh = self._safe_load_phase_data(buf)
                t_upload_end = time.perf_counter()

                # store handle and free previous
                with self._handle_lock:
                    prev = None
                    try:
                        prev = self._handle_pool[buf_idx]
                    except Exception:
                        prev = None
                    self._handle_pool[buf_idx] = dh
                if prev is not None and prev is not dh:
                    try:
                        self._release_handle(prev)
                    except Exception:
                        pass

                t_ready = time.perf_counter()

                # Try to put into filled queue but handle the case when it's full
                try:
                    # be willing to wait a bit for consumer to free space
                    self._filled_buf_queue.put((buf_idx, t_ready), timeout=3.0)
                except queue.Full:
                    # consumer is slow or dead — return buffer index to free queue to avoid deadlock
                    print("Producer: filled queue full, returning buffer to free pool to avoid deadlock.")
                    try:
                        self._free_buf_queue.put_nowait(buf_idx)
                    except Exception:
                        # if even this fails, drop buffer (lost), but don't crash producer
                        print("Producer: failed to return buffer to free pool after filled-queue full.")
                    # optionally break to avoid busy loop
                    time.sleep(0.01)
                    continue

                # small diagnostic print occasionally
                if (k % max(1, self.num_spins // 10)) == 0:
                    # print a short perf summary for this iteration
                    print(f"Producer: k={k} fill={(t_fill_end - t_fill_start):.4f}s upload={(t_upload_end - t_upload_start):.4f}s")
        except Exception as e:
            import traceback
            print("Producer exception (handles):", e)
            traceback.print_exc()
        finally:
            # always try to signal consumer we are done
            try:
                self._filled_buf_queue.put(None, timeout=1.0)
            except Exception:
                pass
    def _reset_buffer_queues(self):
        """(Re)create and populate free/filled buffer queues safely."""
        # recreate queues
        self._free_buf_queue = queue.Queue(maxsize=len(self._mask_buffers))
        self._filled_buf_queue = queue.Queue(maxsize=len(self._mask_buffers) + 2)

        # populate free queue with indices
        for i in range(len(self._mask_buffers)):
            try:
                self._free_buf_queue.put_nowait(i)
            except queue.Full:
                break

        # ensure handle pool exists and matches buffer length
        with getattr(self, "_handle_lock", threading.Lock()):
            if not hasattr(self, "_handle_pool") or len(self._handle_pool) != len(self._mask_buffers):
                self._handle_pool = [None] * len(self._mask_buffers)



    # ---------------------------------------------------
    # 2. Mask Generation & Preparation (Merged Logic)
    # ---------------------------------------------------
    def _precompute_alpha_matrix(self):
        """Precompute alpha_ik for every eigenvector k (shape n_spins x n_spins, float32)."""
        n = self.num_spins
        # store columns as float32 for direct use in Numba
        self._alpha_matrix = np.empty((n, n), dtype=np.float32)  # [spin_i, k]
        # vectorized per-eigvector; keep memory manageable for typical n
        for k in range(n):
            target_amplitudes = self._compensation_factors * self.eigvecs[:, k]
            max_abs_val = np.max(np.abs(target_amplitudes))
            if max_abs_val < 1e-9:
                max_abs_val = 1.0
            normalized = np.clip(target_amplitudes / max_abs_val, -1.0, 1.0)
            self._alpha_matrix[:, k] = np.arccos(normalized).astype(np.float32)
        print(f"[perf] Precomputed alpha matrix ({n}x{n})")

    
    def prep(self):
        """Runs all necessary setup calculations in the correct order."""
        print("Preparing model (layout, compensation, eigendecomposition)...")
        self._setup_layout()
        print(f"  - SLM Layout: {self._grid_rows} rows x {self._grid_cols} cols")
        print(f"  - Macropixel Size: {self._macro_pix_x} x {self._macro_pix_y} pixels")
        self._precompute_checkerboard()
        self._compute_compensation_factors()
        self._perform_eigendecomposition()
        self._precompute_alpha_matrix()
        print("Preparation complete.")
    
    def _setup_layout(self):
        """Determines an optimal rectangular macropixel layout."""
        best_layout = (0, 0)
        max_area = 0
        for rows in range(1, self.num_spins + 1):
            cols = math.ceil(self.num_spins / rows)
            if cols > self.slm_width or rows > self.slm_height:
                continue
            if cols * self.slm_height > rows * self.slm_width:
                continue
            pixel_width = self.slm_width // cols
            pixel_height = self.slm_height // rows
            if pixel_width == 0 or pixel_height == 0:
                continue
            area = pixel_width * pixel_height
            if area > max_area:
                max_area = area
                best_layout = (rows, cols)

        self._grid_rows, self._grid_cols = best_layout
        if self._grid_rows * self._grid_cols < self.num_spins or max_area == 0:
             raise RuntimeError(f"Failed to find a grid for {self.num_spins} spins.")

        self._macro_pix_x = self.slm_width // self._grid_cols
        self._macro_pix_y = self.slm_height // self._grid_rows
        total_width = self._grid_cols * self._macro_pix_x
        total_height = self._grid_rows * self._macro_pix_y
        grid_offset_x = (self.slm_width - total_width) // 2
        grid_offset_y = (self.slm_height - total_height) // 2

        for i in range(self.num_spins):
            row = i // self._grid_cols
            col = i % self._grid_cols
            x0 = grid_offset_x + col * self._macro_pix_x
            y0 = grid_offset_y + row * self._macro_pix_y
            self._spin_coords_x[i] = x0 + self._macro_pix_x / 2
            self._spin_coords_y[i] = y0 + self._macro_pix_y / 2
            self._spin_slices.append((slice(y0, y0 + self._macro_pix_y), slice(x0, x0 + self._macro_pix_x)))
        self._spin_y0 = np.empty(self.num_spins, dtype=np.int32)
        self._spin_x0 = np.empty(self.num_spins, dtype=np.int32)
        for i, (ys, xs) in enumerate(self._spin_slices):
            # slice.start should be int
            self._spin_y0[i] = int(ys.start)
            self._spin_x0[i] = int(xs.start)

    def _precompute_checkerboard(self):
        """Pre-computes the base checkerboard pattern once."""
        lx = np.arange(self._macro_pix_x, dtype=np.int32)
        ly = np.arange(self._macro_pix_y, dtype=np.int32)
        lx_grid, ly_grid = np.meshgrid(lx, ly)
        # checkerboard values as float32
        self._base_checkerboard = ((-1)**(lx_grid + ly_grid)).astype(np.float32)


    def _compute_compensation_factors(self):
        """Computes compensation factors (1/sqrt(I)) vectorized."""
        intensities = np.exp(
            -(((self._spin_coords_x - self._center_x)**2) / (2 * self.beam_sigma_x**2) +
              ((self._spin_coords_y - self._center_y)**2) / (2 * self.beam_sigma_y**2))
        )
        intensities[intensities < 1e-9] = 1e-9
        self._compensation_factors = 1.0 / np.sqrt(intensities)

    def _perform_eigendecomposition(self):
        """Performs eigendecomposition on the J matrix."""
        self.eigvals, self.eigvecs = np.linalg.eigh(self.J)

    # ---------------------------------------------------
    # 3. [NEW] Streaming (FIFO) Mask Generation
    # ---------------------------------------------------

    def _generate_masks_streaming(self, spin_vector: np.ndarray):
        """Yields buffer indices (int) that contain completed masks."""
        spin_phases = np.where(spin_vector == 1, np.pi / 2, 3 * np.pi / 2)

        two_pi = 2 * np.pi
        # Local references for speed
        spin_slices = self._spin_slices
        base_checkerboard = self._base_checkerboard
        comp_factors = self._compensation_factors
        eigvecs = self.eigvecs
        n = self.num_spins

        # Generator loop: compute mask k into a free buffer, then yield its index
        for k in range(n):
            if self.stop_producer_event.is_set():
                return

            # 1) Acquire a free buffer index (will block if none available)
            try:
                buf_idx = self._free_buf_queue.get(timeout=1.0)
            except queue.Empty:
                # no free buffer — bail politely
                print("Producer: timed out waiting for free buffer")
                return

            buf = self._mask_buffers[buf_idx]
            # Fill buffer in-place (no new allocation)
            # Option A: zero the full buffer first (optional)
            # buf.fill(0.0)

            # Compute target amplitudes & alpha (vectorized)
            # target_amplitudes = comp_factors * eigvecs[:, k]
            # max_abs_val = np.max(np.abs(target_amplitudes))
            # if max_abs_val < 1e-9:
            #     max_abs_val = 1.0
            # normalized_amplitudes = np.clip(target_amplitudes / max_abs_val, -1.0, 1.0)
            # alpha_ik = np.arccos(normalized_amplitudes)
            alpha_ik = self._alpha_matrix[:, k]
            # Fill blocks
# Numba-accelerated block fill (single call)
# Ensure dtypes: buf float32, base_checkerboard float32, alpha_ik float32, spin_phases float32, spin_y0/x0 int32
            fill_mask_blocks_numba(
                buf,
                base_checkerboard,
                alpha_ik.astype(np.float32),
                spin_phases.astype(np.float32),
                self._spin_y0,
                self._spin_x0,
                self._macro_pix_y,
                self._macro_pix_x
            )


            # Put the buffer index into the filled queue for consumer to display
            yield buf_idx

    def _producer_worker(self, spin_vector: np.ndarray):
        try:
            for buf_idx in self._generate_masks_streaming(spin_vector):
                if self.stop_producer_event.is_set():
                    break
                # Put the ready buffer index into the filled queue
                self._filled_buf_queue.put(buf_idx)
        except Exception as e:
            import traceback
            print("Producer exception:", e)
            traceback.print_exc()
        finally:
            # signal end
            self._filled_buf_queue.put(None)


    def evaluate_energy(self, spin_vector: np.ndarray) -> float:
        """Consumer: display each buffer provided by producer, measure PD, return energy.

        Robust to receiving either buf_idx or (buf_idx, ready_time) from producer.
        Returns total_energy (float). On fatal display/measure failure returns inf.
        """
        # Reset/prepare
        self.stop_producer_event.clear()

        # Start producer (daemon thread) if not already running
        self.producer_thread = threading.Thread(
            target=self._producer_worker,
            args=(spin_vector,),
            daemon=True
        )
        self.producer_thread.start()

        total_energy = 0.0
        measurement_duration = 1.0 / 60.0
        filled_get_timeout = 2.0
        free_put_timeout = 1.0

        # Feature detection for SDK flags/wait functions
        USE_BACKGROUND_FLAG = getattr(HEDS, 'HEDSSlmShowPhaseFlags', None) is not None
        has_wait_fn = hasattr(self.slm, 'waitForLastFrameDisplayed') or hasattr(self.slm, 'wait_for_frame')

        try:
            for k in range(self.num_spins):
                t_frame_start = time.perf_counter()
                # Get next filled item (may be None sentinel)
                try:
                    item = self._filled_buf_queue.get(timeout=filled_get_timeout)
                except queue.Empty:
                    print("Error: timed out waiting for a filled buffer from producer.")
                    total_energy = float('inf')
                    break

                if item is None:
                    # producer signalled end
                    print("Warning: Producer finished before all masks were measured.")
                    break

                # Accept either (buf_idx, t_ready) or buf_idx
                if isinstance(item, tuple) or isinstance(item, list):
                    if len(item) >= 2:
                        buf_idx, t_ready = item[0], item[1]
                    else:
                        buf_idx = item[0]
                        t_ready = None
                else:
                    buf_idx = item
                    t_ready = None

                # Validate buf_idx type
                if not isinstance(buf_idx, int):
                    print("Error: received non-int buffer index from filled queue:", repr(buf_idx))
                    total_energy = float('inf')
                    break

                # Grab buffer reference (for fallback show)
                try:
                    buf = self._mask_buffers[buf_idx]
                except Exception as e:
                    print("Error: invalid buf_idx", buf_idx, "exception:", e)
                    total_energy = float('inf')
                    # try to continue (return index if possible) then break
                    try:
                        self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                    except Exception:
                        pass
                    break

                # --- ⏱️ BEFORE SHOW ---
                t_before_show = time.perf_counter()

                # Use the safe show helper (tries handle -> buffer fallback)
                err, used_handle = self._show_handle_or_buffer(buf_idx, buf=buf, wait_for_frame=True)

                t_after_show = time.perf_counter()

                if err != HEDSERR_NoError:
                    try:
                        print(f"Error displaying mask k={k}: {HEDS.SDK.ErrorString(err)}")
                    except Exception:
                        print("Error displaying mask k=", k, " err=", err)
                    total_energy = float('inf')
                    # return buffer index to free pool if possible
                    try:
                        self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                    except Exception:
                        pass
                    break

                time.sleep(0.001)

                # Measurement: use latest photodiode value sampled by PD thread
                with self._pd_lock:
                    measured_val = self._last_photodiode_val

                if measured_val is None:
                    print(f"Error measuring mask k={k}.")
                    total_energy = float('inf')
                    try:
                        self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                    except Exception:
                        pass
                    break

                t_after_pd = time.perf_counter()

                # accumulate weighted energy
                try:
                    total_energy += float(measured_val) * float(self.eigvals[k])
                except Exception:
                    # in case eigvals indexing fails, continue but flag error
                    total_energy = float('inf')
                    try:
                        self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                    except Exception:
                        pass
                    break

                t_frame_end = time.perf_counter()

                # === Record timing info (non-blocking logging) ===
                frame_info = {
                    "k": k,
                    "mask_ready": (t_before_show - t_ready) if (t_ready is not None) else None,
                    "show_time": t_after_show - t_before_show,
                    "pd_time": t_after_pd - t_after_show,
                    "frame_total": t_frame_end - t_frame_start,
                    "timestamp": t_frame_start,
                }
                try:
                    self._timing_log.append(frame_info)
                    self._timing_window.append(frame_info)
                except Exception:
                    pass

                # Periodic timing summary (safe)
                try:
                    if len(self._timing_log) % 100 == 0:
                        totals = [f["frame_total"] for f in self._timing_window if f.get("frame_total") is not None]
                        show_times = [f["show_time"] for f in self._timing_window if f.get("show_time") is not None]
                        mask_ready_times = [f["mask_ready"] for f in self._timing_window if f.get("mask_ready") is not None]
                        if totals and len(totals) > 1:
                            import statistics
                            print(f"[Timing] frames={len(totals)} avg={statistics.mean(totals)*1000:.2f} ms  "
                                f"std={statistics.stdev(totals)*1000:.2f} ms  "
                                f"show_avg={statistics.mean(show_times)*1000:.2f} ms  show_std={statistics.stdev(show_times)*1000:.2f} ms "
                                + (f"mask_ready_avg={statistics.mean(mask_ready_times)*1000:.2f} ms" if mask_ready_times else ""))
                except Exception:
                    pass

                # Return buffer index to free pool now that display+measure completed
                try:
                    self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                except Exception:
                    # if we can't return it, keep going (producer will eventually timeout)
                    pass

        finally:
            # Signal producer to stop (if still running) and join briefly
            self.stop_producer_event.set()
            if self.producer_thread is not None:
                self.producer_thread.join(timeout=3.0)

            # Drain any leftover filled queue items and return their indices to free queue
            while True:
                try:
                    leftover = self._filled_buf_queue.get_nowait()
                except queue.Empty:
                    break
                if leftover is None:
                    break
                try:
                    if isinstance(leftover, (tuple, list)) and len(leftover) >= 1:
                        idx = leftover[0]
                    else:
                        idx = leftover
                    if isinstance(idx, int):
                        try:
                            self._free_buf_queue.put_nowait(idx)
                        except Exception:
                            pass
                except Exception:
                    pass

        return total_energy

    # ---------------------------------------------------
    # 4. Simulated Annealing Loop
    # ---------------------------------------------------

    def run_annealing(self, initial_temp, final_temp, cooling_rate, steps_per_temp):
        """
        Performs the simulated annealing algorithm.
        """
        current_spin_vector = np.random.choice([-1, 1], size=self.num_spins)
        
        print("Evaluating initial random spin configuration...")
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

                proposed_energy = self.evaluate_energy(proposed_spin_vector)


                
                delta_energy = proposed_energy - current_energy

                if proposed_energy - min_energy < 0:
                    min_energy = proposed_energy
                    min_spin_vector = proposed_spin_vector
                acceptance_prob = math.exp(-delta_energy / temp) if temp > 0 else 0
                is_accepted = delta_energy < 0 or random.random() < acceptance_prob
                # print(f"Step {step}: Delta E: {delta_energy:.4f} | Prob: {acceptance_prob:.4f} | Accepted: {is_accepted}")

                if is_accepted:
                    current_spin_vector = proposed_spin_vector
                    current_energy = proposed_energy
                    print(f"  Step {step+1}/{steps_per_temp} | New Energy Accepted: {current_energy:.4f}")
                
                energy_plot.append(current_energy)

            end_temp_time = time.time()
            print(f"  Temp step took {end_temp_time - start_temp_time:.2f}s")
            temp *= cooling_rate
            
        print("\nSimulated annealing finished.")
        total_time = time.time() - annealing_start_time
        print(f"Total time taken: {total_time:.2f} seconds")

        # Find the minimum energy and corresponding spin configuration
        energy_1 = self.evaluate_energy(min_spin_vector)
        energy_2 = self.evaluate_energy(current_spin_vector)
        if energy_1 < energy_2:
            print(f"Minimum energy found, final spin not optimum: {energy_1:.4f}")
            current_spin_vector = min_spin_vector
            current_energy = energy_1
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(energy_plot)
        plt.xlabel("Annealing Step")
        plt.ylabel("Measured Energy")
        plt.title("Photonic Annealing Energy Convergence")
        plt.show()

        return current_spin_vector, energy_plot

# -----------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------

if __name__ == '__main__':
    
    # --- 1. Define Problem ---
    NUM_SPINS = 40
    np.random.seed(42)
    J_random = np.random.randn(NUM_SPINS, NUM_SPINS)
    J_random = (J_random + J_random.T) / 2 # Symmetrize
    
    # --- 2. Define Experimental Parameters ---
    BEAM_SIGMA_X = 600
    BEAM_SIGMA_Y = 600
    
    # --- 3. Define Annealing Parameters ---
    INITIAL_TEMP = 1000.0
    FINAL_TEMP = 0.1
    COOLING_RATE = 0.95
    STEPS_PER_TEMP = 5 # Set low for testing, increase for real runs

    annealer = None # Initialize to None for the finally block
    try:
        # --- 4. Initialize Annealer (connects hardware, runs prep) ---
        annealer = PhotonicAnnealer(
            J=J_random,
            beam_sigma_x=BEAM_SIGMA_X,
            beam_sigma_y=BEAM_SIGMA_Y,
            serial_port='COM21' # [USER] Verify this port
        )
        
        # --- 5. Run the Annealing ---
        final_spins, energy_data = annealer.run_annealing(
            initial_temp=INITIAL_TEMP,
            final_temp=FINAL_TEMP,
            cooling_rate=COOLING_RATE,
            steps_per_temp=STEPS_PER_TEMP
        )
        
        print("\n--- Results ---")
        print(f"Final Spin Configuration: {final_spins}")
        print(f"Final Energy: {energy_data[-1]}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- 6. Safely Disconnect ---
        if annealer is not None:
            annealer.disconnect_hardware()
        else:
            # If annealer init failed, SDK might still be open
            HEDS.SDK.Exit()
            print("HEDS SDK closed (in cleanup).")