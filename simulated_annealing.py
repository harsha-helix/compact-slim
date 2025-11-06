# -*- coding: utf-8 -*-
"""
This script runs a simulated annealing algorithm on a spatial photonic Ising machine.
It uses the HOLOEYE SLM Display SDK to display phase masks corresponding to
Ising spin configurations and relies on an external measurement to guide the
annealing process.

The core algorithm operates on a 1D spin vector and is decoupled from the
hardware-specific display and measurement logic.

Instructions:
1. Ensure the holoeye_slmdisplaysdk library and its dependencies are correctly installed.
2. Implement the 'measure_objective_function' with your specific hardware code
   to read the energy/objective from your optical setup's detector.
"""

import numpy as np
import random
import math
import time
from beam_comp import *
import serial

# Import the SLM Display SDK
# Ensure the holoeye_slmdisplaysdk.py file and its dependencies are in the path
try:
    from holoeye_slmdisplaysdk import SLM, HEDSERR_NoError
    # Example: If you use a specific camera library, import it here.
    # from pypylon import pylon 
except ImportError:
    print("Error: Could not import HOLOEYE SLM Display SDK.")
    print("Please ensure 'holoeye_slmdisplaysdk.py' is in the Python path.")
    exit()

# --- USER IMPLEMENTATION REQUIRED ---

# --- Global hardware handles (optional, but good practice) ---
# Example: Initialize your camera once outside the measurement loop
# try:
#     camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#     camera.Open()
#     print("Camera initialized successfully.")
# except Exception as e:
#     print(f"Error initializing camera: {e}")
#     camera = None
# -----------------------------------------------------------
def connect_to_slm():
    """
    Initializes the connection to a HOLOEYE SLM and returns the device object.

    This function will open the HOLOEYE SLM Display SDK's device selection
    window by default if multiple devices are found.

    Returns:
        SLM: An initialized HOLOEYE SLM object if the connection is successful.
        None: If the connection fails or the user cancels the selection.
    """
    print("Initializing HOLOEYE SLM...")
    
    # This is the primary call to initialize the SDK and find an SLM
    slm_device = SLM.Init()

    # Check the error code to see if initialization was successful
    if slm_device.errorCode() == HEDSERR_NoError:
        print("SLM connected successfully.")
        print(f"  - Device: {slm_device.window().getDeviceName()}")
        print(f"  - Resolution: {slm_device.width_px()} x {slm_device.height_px()}")
        return slm_device
    else:
        print(f"Error: Could not connect to SLM. Error code: {slm_device.errorCode()}")
        return None

def display_phase_mask(slm_device, phase_mask_2d):
    """
    Displays a 2D NumPy array of phase values on an initialized SLM.

    Args:
        slm_device (SLM): An already initialized HOLOEYE SLM object.
        phase_mask_2d (numpy.ndarray): A 2D NumPy array where each value
                                       represents a phase in radians.
                                       The array should have a dtype of
                                       np.float32 for best performance.
    
    Returns:
        bool: True if the display was successful, False otherwise.
    """
    # Check if the SLM object is valid
    if not isinstance(slm_device, SLM) or slm_device.errorCode() != HEDSERR_NoError:
        print("Error: The provided SLM device is not valid or has an error.")
        return False

    # Check if the input is a NumPy array
    if not isinstance(phase_mask_2d, np.ndarray):
        print("Error: The provided phase mask must be a NumPy array.")
        return False

    print("Sending the 2D phase array to the SLM...")
    
    # The core function call to display the data
    error = slm_device.showPhaseData(phase_mask_2d)

    if error == HEDSERR_NoError:
        print("Phase mask displayed successfully!")
        return True
    else:
        print(f"An error occurred while displaying the phase mask: {error}")
        return False



def photodiode_measurement(port='COM3', baudrate=115200, duration=0.1):
    ser = serial.Serial(port, baudrate, timeout=0.1)
    time.sleep(1)  # Wait for Tiva to boot/reset

    adc_values = []
    start_time = time.time()

    while time.time() - start_time < duration:
        line = ser.readline().decode().strip()
        if line.isdigit():
            adc_values.append(int(line))

    ser.close()

    if adc_values:
        return sum(adc_values) / len(adc_values)
    else:
        return None  # or 0 or raise an exception



def evaluate_objective_function(spin_config, slm, interactions: CompensatedMattisInteractions):
    """
    This function should be implemented to evaluate the objective function
    based on the measured energy from the photodiode or detector.
    
    Returns:
        float: The measured energy value.
    """
    mattis_phase_masks = interactions.generate_phase_masks(spin_config)
    mattis_energies = []
    for i in range(len(mattis_phase_masks)):
        display = display_phase_mask(slm, mattis_phase_masks[i])
        if not display:
            print("Error displaying phase mask on SLM. Returning high energy value.")
            return float('inf')
        time.sleep(0.05)  # Allow time for the SLM to update
        mattis_energies.append(photodiode_measurement()*interactions.eigvals[i])
    # Measure the energy from the photodiode or detector

    energy = sum(mattis_energies) if mattis_energies else 0
    print(f"energy measured: {energy}")
    return energy

def run_photonic_annealing(num_spins, evaluate_spin_vector_func, initial_temp, final_temp, cooling_rate, steps_per_temp):
    """
    Performs a simulated annealing algorithm using a hardware objective function.
    This function is agnostic to the hardware details (SLM, camera, etc.).

    Args:
        num_spins (int): The total number of spins in the system.
        evaluate_spin_vector_func (function): A function that takes a 1D spin vector,
                                              sends it to the hardware, and returns a
                                              measured energy (float).
        initial_temp (float): The starting temperature.
        final_temp (float): The ending temperature.
        cooling_rate (float): The factor by which temperature is multiplied.
        steps_per_temp (int): The number of Monte Carlo steps at each temperature.

    Returns:
        numpy.ndarray: The final, low-energy 1D spin vector.
    """
    # Initialize a 1D spin vector with random spins (-1 or 1)
    current_spin_vector = np.random.choice([-1, 1], size=num_spins)

    # Get the initial energy measurement by evaluating the initial spin vector
    print("Evaluating initial random spin configuration...")
    current_energy = evaluate_spin_vector_func(current_spin_vector)
    print(f"Initial measured energy: {current_energy:.4f}")

    temp = initial_temp
    
    print("\nStarting photonic simulated annealing...")
    print(f"Initial Temp: {initial_temp}, Final Temp: {final_temp}, Cooling Rate: {cooling_rate}")

    while temp > final_temp:
        print(f"\nCurrent Temperature: {temp:.4f}")
        for step in range(steps_per_temp):
            # 1. Pick a random spin to consider flipping from the 1D vector
            idx = random.randint(0, num_spins - 1)
            
            # 2. Create a proposed new 1D vector with the flipped spin
            proposed_spin_vector = np.copy(current_spin_vector)
            proposed_spin_vector[idx] *= -1

            # 3. Evaluate the proposed state using the hardware function
            proposed_energy = evaluate_spin_vector_func(proposed_spin_vector)
            
            # 4. Calculate the change in energy
            delta_energy = proposed_energy - current_energy
            
            # 5. Decide whether to accept the flip using the Metropolis criterion
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
                # Accept the new state
                current_spin_vector = proposed_spin_vector
                current_energy = proposed_energy


            if step % 100 == 0:
                print(f"  Step {step}/{steps_per_temp} | Current Energy: {current_energy:.4f}")

        # Cool the system down
        temp *= cooling_rate

    print("\nSimulated annealing finished.")
    return current_spin_vector


def data_loader(filename):
    """
    Takes data from .npz and returns adjacency matrix.
    """
    import numpy as np
    import scipy.sparse as sp

    # Load the .npz file
    data = np.load(filename, allow_pickle=True)

    # Reconstruct the sparse adjacency matrix (CSR format)
    adj_sparse = sp.csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape=tuple(data['shape'])
    )

    # If you want a dense NumPy array (e.g., for neural nets)
    return adj_sparse.toarray()


if __name__ == "__main__":
    dscr = "data/graph.npz"
    adj_matrix = data_loader(dscr)