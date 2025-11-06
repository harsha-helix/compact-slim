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
# -----------------------------------------------------------------

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
        serial_port: str = 'COM8',
        serial_baud: int = 115200,
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
        self.ser: serial.Serial = None
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        
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
            self.ser = serial.Serial(self.serial_port, self.serial_baud, timeout=0.1)
            time.sleep(1.0) # Wait for Tiva to boot/reset
            self.ser.flushInput()
            print(f"Serial port {self.serial_port} connected.")
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to connect to {self.serial_port}: {e}")

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

    def _photodiode_measurement_fast(ser: serial.Serial, duration: float = 0.05,
                                    adc_ref: Optional[float] = 3.3, adc_bits: int = 12):
        """
        Fast binary reader for frames: [0xA5][lo][hi] (3 bytes per sample).
        Returns average voltage (if adc_ref provided) or average raw counts.
        """
        buf = bytearray()
        start_time = time.perf_counter()
        sample_sum = 0
        sample_count = 0
        FRAME_SIZE = 3
        START_BYTE = 0xA5

        # Pre-allocate a read buffer to reduce allocations
        read_chunk_size = 4096  # tune larger if you expect big bursts

        while time.perf_counter() - start_time < duration:
            chunk = ser.read(read_chunk_size)
            if chunk:
                buf.extend(chunk)

                # Fast parser: find start byte and parse if full frame available
                i = 0
                # Use while loop to avoid slicing overhead
                while i + FRAME_SIZE <= len(buf):
                    if buf[i] != START_BYTE:
                        # fast-skip to next possible start byte
                        # find next occurrence to avoid byte-by-byte increment
                        try:
                            nxt = buf.index(START_BYTE, i + 1)
                            i = nxt
                        except ValueError:
                            # no start byte found; drop processed bytes
                            # keep last two bytes in case start byte arrives split
                            del buf[:max(0, len(buf) - 2)]
                            i = 0
                            break
                    else:
                        # full frame available?
                        if i + FRAME_SIZE <= len(buf):
                            # parse uint16 little-endian from buf[i+1:i+3]
                            lo = buf[i+1]
                            hi = buf[i+2]
                            val = lo | (hi << 8)
                            sample_sum += val
                            sample_count += 1
                            i += FRAME_SIZE
                        else:
                            # incomplete frame — wait for more bytes
                            break

                # remove processed bytes from front
                if i > 0:
                    del buf[:i]
            else:
                # no data available right now - tiny sleep to avoid 100% CPU
                # Use a very short sleep to remain responsive
                time.sleep(0.0005)

        if sample_count == 0:
            # no data
            return None

        avg_count = sample_sum / sample_count
        if adc_ref is not None:
            max_counts = (1 << adc_bits) - 1
            return (avg_count / max_counts) * adc_ref
        else:
            return avg_count


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
        """
        [PRODUCER] This is a generator function that yields one mask at a time.
        This runs on the producer thread.
        """
        # Pre-compute spin phases once
        spin_phases = np.where(spin_vector == 1, np.pi / 2, 3 * np.pi / 2)
        
        # Pre-allocate mask array to be re-used
        phase_mask = np.zeros((self.slm_height, self.slm_width), dtype=np.float32)

        for k in range(self.num_spins):
            if self.stop_producer_event.is_set():
                return # Stop generating if event is set

            # --- Start Mask Computation (CPU-bound) ---
            # Clear only the parts that will be written (more efficient)
            # Or just re-use the array, it will be overwritten
            
            target_amplitudes = self._compensation_factors * self.eigvecs[:, k]
            max_abs_val = np.max(np.abs(target_amplitudes))
            if max_abs_val < 1e-9:
                max_abs_val = 1.0
            
            normalized_amplitudes = np.clip(target_amplitudes / max_abs_val, -1.0, 1.0)

            # Use the correct arccos logic
            alpha_ik = np.arccos(normalized_amplitudes)

            for i in range(self.num_spins):
                amplitude = alpha_ik[i]
                spin_phase = spin_phases[i]
                mask_slice = self._spin_slices[i]
                
                phi_block = spin_phase + (self._base_checkerboard * amplitude)
                phase_mask[mask_slice] = phi_block % (2 * np.pi)
            
            # --- End Mask Computation ---
            
            # Yield the completed mask to the queue
            yield phase_mask

    def _producer_worker(self, spin_vector: np.ndarray):
        """
        [PRODUCER THREAD] Target function for the producer thread.
        It runs the generator and 'puts' masks onto the queue.
        """
        try:
            mask_generator = self._generate_masks_streaming(spin_vector)
            for mask in mask_generator:
                if self.stop_producer_event.is_set():
                    break
                # This line will block if the queue is full (size 5),
                # waiting for the consumer to 'get' a mask.
                self.mask_queue.put(mask)
        finally:
            # Signal the end of the stream
            self.mask_queue.put(None) 

    def evaluate_energy(self, spin_vector: np.ndarray) -> float:
        """
        [CONSUMER] Evaluates the objective function for a given spin vector.
        This now manages the producer-consumer threads.
        """
        
        # --- 1. Start the Producer Thread ---
        # Clear any old stop events and create a fresh queue
        self.stop_producer_event.clear()
        self.mask_queue = queue.Queue(maxsize=5) 
        
        self.producer_thread = threading.Thread(
            target=self._producer_worker,
            args=(spin_vector,)
        )
        self.producer_thread.start()

        # --- 2. Run the Consumer (Main) Loop ---
        total_energy = 0.0
        measurement_duration = 0.05 # [USER] Tune this
        slm_wait_time = 0.05        # [USER] Tune this (SLM refresh time)

        for k in range(self.num_spins):
            # Get mask from queue (blocks until producer provides one)
            phase_mask = self.mask_queue.get()
            
            if phase_mask is None:
                # Producer finished early (shouldn't happen if k < num_spins)
                print("Warning: Producer finished before all masks were measured.")
                break

            # --- Display and Measure (I/O-bound) ---
            if not self._display_phase_mask(phase_mask):
                print(f"Error displaying mask k={k}. Stopping evaluation.")
                total_energy = float('inf') # Return high energy
                break
                
            time.sleep(slm_wait_time) # Wait for SLM to physically update
            
            measured_val = self._photodiode_measurement(duration=measurement_duration)
            
            if measured_val is None:
                print(f"Error measuring mask k={k}. Stopping evaluation.")
                total_energy = float('inf') # Return high energy
                break
                
            total_energy += measured_val * self.eigvals[k]
            # ---
            # While this thread was sleeping and measuring,
            # the producer thread was busy computing the *next* mask.
            # ---

        # --- 3. Cleanup ---
        # Tell the producer to stop (if it's not already done)
        self.stop_producer_event.set()
        
        # Drain the queue to unblock the producer if it's stuck on queue.put()
        while not self.mask_queue.empty():
            try:
                self.mask_queue.get_nowait()
            except queue.Empty:
                break
                
        self.producer_thread.join() # Wait for the producer thread to exit
        
        #print(f"Energy: {total_energy:.4f}")
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
            serial_port='COM8' # [USER] Verify this port
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