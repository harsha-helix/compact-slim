# Intensity profile:
Code used to measure intensity profile of expanded beam using Basler CMOS camera using multiple images of different exposures.

# beam_comp.py:
Gives phase mapping for gaussian beam corrected mattis interactions on SLM

# simulated_annealing.py:
Helper functions to connect to SLM, display phasemasks for mattis hamiltonians and measure energy using photodiode and run simulated annealing loop

# Photonic_SA.ipynb:
Imports beam_comp.py and has same functions as simulated_annealing.py, used for testing.

# Output.png:
Result of MaxCut using MC100.npz graph (100 spins), obtained cut 1249, best cut 1287.