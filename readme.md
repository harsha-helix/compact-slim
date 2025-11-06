# Photonic Simulated Annealing for Ising Problems

This project implements a simulated annealing algorithm on a spatial photonic Ising machine. It uses a HOLOEYE Spatial Light Modulator (SLM) to encode Ising Hamiltonians onto a laser beam and measures the system's energy with a photodiode to find the ground state of the Ising model.

## Project Overview

The core of the project is to solve Ising problems, which are mathematical models of ferromagnetism in statistical mechanics. The Ising model consists of discrete variables representing magnetic dipole moments of atomic "spins" that can be in one of two states (+1 or -1). The energy of the system is a function of the spin configuration and the interaction matrix J, which defines the coupling strength between spins.

This project uses a photonic setup to find the ground state of the Ising model, which is the spin configuration that minimizes the energy. The setup consists of a laser, an SLM, and a photodiode. The SLM is used to modulate the phase of the laser beam to encode the Ising Hamiltonian. The photodiode measures the energy of the system for a given spin configuration. The simulated annealing algorithm is then used to find the spin configuration that minimizes the energy.

## Key Files

*   **`beam_comp.py`**: This script defines the `CompensatedMattisInteractions` class, which is responsible for generating the phase masks for the SLM. The phase masks are compensated for the Gaussian intensity profile of the laser beam to ensure uniform interaction strengths.

*   **`simulated_annealing.py`**: This script implements the simulated annealing algorithm. It uses the `CompensatedMattisInteractions` class to generate the phase masks and a photodiode to measure the energy of the system. The script then iteratively flips spins and accepts or rejects the new configuration based on the Metropolis criterion.

*   **`Photonic_SA.ipynb`**: This Jupyter notebook is used for testing and running the simulated annealing algorithm. It provides a step-by-step guide on how to connect to the SLM, load the interaction matrix, and run the simulation.

*   **`Intensity profile.ipynb`**: This notebook contains code to measure the intensity profile of the expanded beam using a Basler CMOS camera.

*   **`MC100.npz`**: This file contains the interaction matrix for a 100-spin Ising problem.

*   **`output.png`**: This image shows the result of solving the MaxCut problem for the `MC100.npz` graph.

## How to Run the Simulations

1.  **Install the necessary libraries**: Make sure you have the HOLOEYE SLM Display SDK and other required libraries like NumPy, Matplotlib, and PySerial installed.

2.  **Connect the hardware**: Connect the SLM and the photodiode to your computer.

3.  **Run the `Photonic_SA.ipynb` notebook**: This notebook will guide you through the process of connecting to the SLM, loading the interaction matrix, and running the simulated annealing algorithm.

## Results

The `output.png` image shows the result of solving the MaxCut problem for the `MC100.npz` graph. The algorithm found a cut of 1249, which is close to the best-known cut of 1287.
