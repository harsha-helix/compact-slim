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
        self._buffer_pool_size = 4

        # Pre-allocate buffers once SLM size is known (after _connect_hardware() & prep())
        self._mask_buffers = [np.zeros((self.slm_height, self.slm_width), dtype=np.float32)
                            for _ in range(self._buffer_pool_size)]

        # Queues to manage buffer indices
        self._free_buf_queue = queue.Queue()
        self._filled_buf_queue = queue.Queue()

        # Populate free queue with indices
        for i in range(self._buffer_pool_size):
            self._free_buf_queue.put(i)


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

    def disconnect_hardware(self):
        """Safely disconnects from SLM and Serial port."""
        print("\nDisconnecting hardware...")
        if self.slm is not None:
            self.slm.showBlankScreen(0) # Blank the SLM
            err = self.slm.window().close()
            if err == HEDSERR_NoError:
                print("SLM window closed.")
            HEDS.SDK.Exit()
            print("HEDS SDK closed.")
        
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.serial_port} closed.")

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


    # ---------------------------------------------------
    # 2. Mask Generation & Preparation (Merged Logic)
    # ---------------------------------------------------
    
    def prep(self):
        """Runs all necessary setup calculations in the correct order."""
        print("Preparing model (layout, compensation, eigendecomposition)...")
        self._setup_layout()
        print(f"  - SLM Layout: {self._grid_rows} rows x {self._grid_cols} cols")
        print(f"  - Macropixel Size: {self._macro_pix_x} x {self._macro_pix_y} pixels")
        self._precompute_checkerboard()
        self._compute_compensation_factors()
        self._perform_eigendecomposition()
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

    def _precompute_checkerboard(self):
        """Pre-computes the base checkerboard pattern once."""
        lx = np.arange(self._macro_pix_x)
        ly = np.arange(self.slm_height // self._grid_rows) # Use floored height
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
            target_amplitudes = comp_factors * eigvecs[:, k]
            max_abs_val = np.max(np.abs(target_amplitudes))
            if max_abs_val < 1e-9:
                max_abs_val = 1.0
            normalized_amplitudes = np.clip(target_amplitudes / max_abs_val, -1.0, 1.0)
            alpha_ik = np.arccos(normalized_amplitudes)

            # Fill blocks
            for i in range(n):
                amplitude = alpha_ik[i]
                spin_phase = spin_phases[i]
                ys, xs = spin_slices[i]
                phi_block = spin_phase + (base_checkerboard * amplitude)
                # In-place assignment to the preallocated buffer
                buf[ys, xs] = phi_block % two_pi

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
        """Consumer: display each buffer provided by producer, measure PD, return energy."""
        # Reset/prepare
        self.stop_producer_event.clear()

        # Start producer (daemon thread)
        self.producer_thread = threading.Thread(
            target=self._producer_worker,
            args=(spin_vector,),
            daemon=True
        )
        self.producer_thread.start()

        total_energy = 0.0
        measurement_duration = 1.0 / 60.0
        # short timeouts to detect producer failure quickly
        filled_get_timeout = 2.0
        free_put_timeout = 1.0

        # Detect whether SDK supports background show flags or wait function:
        USE_BACKGROUND_FLAG = getattr(HEDS, 'HEDSSlmShowPhaseFlags', None) is not None
        has_wait_fn = hasattr(self.slm, 'waitForLastFrameDisplayed') or hasattr(self.slm, 'wait_for_frame')

        try:
            for k in range(self.num_spins):
                try:
                    buf_idx = self._filled_buf_queue.get(timeout=filled_get_timeout)
                except queue.Empty:
                    print("Error: timed out waiting for a filled buffer from producer.")
                    total_energy = float('inf')
                    break

                if buf_idx is None:
                    # producer signalled end
                    print("Warning: Producer finished before all masks were measured.")
                    break

                buf = self._mask_buffers[buf_idx]  # float32 preallocated buffer

                # Display: avoid any copy here (buf is already float32)
                # If SDK has background flag, use it; else rely on blocking showPhaseData
                if USE_BACKGROUND_FLAG:
                    # use SDK enum if available (this name may vary by SDK version)
                    try:
                        flags = HEDS.HEDSSlmShowPhaseFlags.SHOW_IN_BACKGROUND
                        err = self.slm.showPhaseData(buf, flags)
                    except Exception:
                        # fallback to simple call
                        err = self.slm.showPhaseData(buf)
                else:
                    err = self.slm.showPhaseData(buf)

                if err != HEDSERR_NoError:
                    print(f"Error displaying mask k={k}: {HEDS.SDK.ErrorString(err)}")
                    total_energy = float('inf')
                    # return buffer safely (best-effort)
                    try:
                        self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                    except Exception:
                        pass
                    break

                # If SDK is async and provides a wait function, wait until SLM consumed this frame
                if not hasattr(self.slm, 'showPhaseData') or (USE_BACKGROUND_FLAG and has_wait_fn):
                    # call wait if available (names vary by SDK)
                    wait_fn = getattr(self.slm, 'waitForLastFrameDisplayed', None) or getattr(self.slm, 'wait_for_frame', None)
                    if callable(wait_fn):
                        wait_fn()  # blocks until SLM finished reading/uploading frame

                # Measure PD
                measured_val = self._photodiode_measurement(duration=measurement_duration)
                if measured_val is None:
                    print(f"Error measuring mask k={k}.")
                    total_energy = float('inf')
                    # return buffer before breaking
                    try:
                        self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                    except Exception:
                        pass
                    break

                # accumulate
                total_energy += measured_val * self.eigvals[k]

                # Return buffer to free pool now that display+measure completed
                try:
                    self._free_buf_queue.put(buf_idx, timeout=free_put_timeout)
                except queue.Full:
                    # improbable: if free queue full, drop it (producer will timeout waiting for free)
                    pass

        finally:
            # Signal producer to stop (if still running) and join
            self.stop_producer_event.set()
            # join with timeout (avoid hang)
            if self.producer_thread is not None:
                self.producer_thread.join(timeout=3.0)

            # Drain any leftover filled buffers and return them to free pool
            while True:
                try:
                    item = self._filled_buf_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    break
                try:
                    self._free_buf_queue.put_nowait(item)
                except queue.Full:
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

        while temp > final_temp:
            print(f"\nCurrent Temperature: {temp:.4f}")
            start_temp_time = time.time()
            
            for step in range(steps_per_temp):
                idx = random.randint(0, self.num_spins - 1)
                
                proposed_spin_vector = np.copy(current_spin_vector)
                proposed_spin_vector[idx] *= -1

                proposed_energy = self.evaluate_energy(proposed_spin_vector)
                
                delta_energy = proposed_energy - current_energy
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
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