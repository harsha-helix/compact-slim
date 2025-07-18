import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple, List

class CompensatedMattisInteractions:
    """
    A class to calculate and display compensated phase masks for encoding Mattis-type
    Ising Hamiltonians on a Spatial Light Modulator (SLM).

    This implementation decomposes a given interaction matrix J into its eigenmodes
    (Mattis Hamiltonians) and computes the necessary phase patterns to realize them
    optically. It crucially compensates for an inhomogeneous (e.g., Gaussian)
    beam profile to ensure uniform interaction strengths.

    Attributes:
        slm_width (int): The width of the SLM in pixels.
        slm_height (int): The height of the SLM in pixels.
        num_spins (int): The number of spins in the Ising model.
        J (np.ndarray): The interaction matrix.
        beam_sigma_x (float): The standard deviation (sigma) of the Gaussian beam in x.
        beam_sigma_y (float): The standard deviation (sigma) of the Gaussian beam in y.
        eigvals (np.ndarray): Eigenvalues of the interaction matrix J.
        eigvecs (np.ndarray): Eigenvectors of the interaction matrix J.
    """

    def __init__(
        self,
        J: np.ndarray,
        beam_sigma_x: float,
        beam_sigma_y: float,
        slm_width: int = 1920,
        slm_height: int = 1080,
    ):
        """
        Initializes the model with the interaction matrix and system parameters.

        Args:
            J (np.ndarray): The n_spins x n_spins interaction matrix. Must be symmetric.
            beam_sigma_x (float): The standard deviation of the Gaussian beam in the x-direction.
            beam_sigma_y (float): The standard deviation of the Gaussian beam in the y-direction.
            slm_width (int): The width of the SLM in pixels.
            slm_height (int): The height of the SLM in pixels.
        """
        if not isinstance(J, np.ndarray) or J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square 2D numpy array.")
        if not np.allclose(J, J.T):
            max_diff = np.max(np.abs(J - J.T))
            raise ValueError(f"Interaction matrix J is not symmetric (max asymmetry={max_diff:.3e})")

        self.slm_width = slm_width
        self.slm_height = slm_height
        self.num_spins = J.shape[0]
        self.J = J
        self.beam_sigma_x = beam_sigma_x
        self.beam_sigma_y = beam_sigma_y

        # Attributes to be computed by the prep() method
        self._center_x: float = self.slm_width / 2
        self._center_y: float = self.slm_height / 2
        self.eigvals: np.ndarray = None
        self.eigvecs: np.ndarray = None
        self._grid_rows: int = 0
        self._grid_cols: int = 0
        self._macro_pix_x: int = 0
        self._macro_pix_y: int = 0
        self._intensity_map: np.ndarray = None
        self._compensation_factors: np.ndarray = None
        self._grid_offset_x: int = 0
        self._grid_offset_y: int = 0

    def _setup_layout(self):
        """
        Determines an optimal rectangular macropixel layout to fill the SLM area.
        This method maximizes the area of each macropixel to improve resolution.
        """
        best_layout = (0, 0)
        max_area = 0

        for rows in range(1, self.num_spins + 1):
            cols = math.ceil(self.num_spins / rows)
            if cols * self.slm_height > rows * self.slm_width: # Simple aspect ratio check
                continue

            pixel_width = self.slm_width // cols
            pixel_height = self.slm_height // rows
            area = pixel_width * pixel_height

            if area > max_area:
                max_area = area
                best_layout = (rows, cols)

        self._grid_rows, self._grid_cols = best_layout
        if self._grid_rows * self._grid_cols < self.num_spins:
             raise RuntimeError("Failed to find a macropixel grid that fits all spins.")

        self._macro_pix_x = self.slm_width // self._grid_cols
        self._macro_pix_y = self.slm_height // self._grid_rows

        total_width = self._grid_cols * self._macro_pix_x
        total_height = self._grid_rows * self._macro_pix_y

        self._grid_offset_x = (self.slm_width - total_width) // 2
        self._grid_offset_y = (self.slm_height - total_height) // 2

    def _compute_intensity_map(self):
        """
        Computes the 2D Gaussian intensity profile of the beam on the SLM.
        """
        x = np.arange(self.slm_width)
        y = np.arange(self.slm_height)
        X, Y = np.meshgrid(x, y)

        # Gaussian distribution formula
        self._intensity_map = np.exp(
            -(((X - self._center_x)**2) / (2 * self.beam_sigma_x**2) +
              ((Y - self._center_y)**2) / (2 * self.beam_sigma_y**2))
        )

    def _compute_compensation_factors(self):
        """
        Computes compensation factors (Ai) for each spin to counteract
        the intensity falloff of the Gaussian beam.
        """
        intensities_at_spins = np.zeros(self.num_spins)
        for i in range(self.num_spins):
            row = i // self._grid_cols
            col = i % self._grid_cols

            # Center of the macropixel for spin i
            x_pos = self._grid_offset_x + col * self._macro_pix_x + self._macro_pix_x / 2
            y_pos = self._grid_offset_y + row * self._macro_pix_y + self._macro_pix_y / 2

            # Intensity at the center of the macropixel
            intensities_at_spins[i] = np.exp(
                -(((x_pos - self._center_x)**2) / (2 * self.beam_sigma_x**2) +
                  ((y_pos - self._center_y)**2) / (2 * self.beam_sigma_y**2))
            )

        intensities_at_spins[intensities_at_spins < 1e-9] = 1e-9  # Avoid division by zero

        # The compensation factor is the sqrt of the normalized inverse intensity
        weights = np.max(intensities_at_spins) / intensities_at_spins
        compensation = np.sqrt(weights)
        self._compensation_factors = compensation / np.max(compensation) # Normalize to [0, 1]

    def _perform_eigendecomposition(self):
        """
        Performs eigendecomposition on the symmetric interaction matrix J to
        find the Mattis modes (eigenvectors) and their strengths (eigenvalues).
        """
        self.eigvals, self.eigvecs = np.linalg.eigh(self.J)

    def prep(self):
        """
        Runs all necessary setup calculations in the correct order. This method
        must be called before generating phase masks.
        """
        print("Preparing model...")
        self._setup_layout()
        print(f"  - SLM Layout: {self._grid_rows} rows x {self._grid_cols} cols")
        print(f"  - Macropixel Size: {self._macro_pix_x} x {self._macro_pix_y} pixels")
        self._compute_intensity_map()
        self._compute_compensation_factors()
        self._perform_eigendecomposition()
        print("Preparation complete.")

    def generate_phase_masks(
        self,
        spin_vector: List[int],
        display_limit: int = 5
    ):
        """
        Generates and displays the phase mask for each Mattis Hamiltonian (eigenmode k)
        based on a given spin configuration.

        Args:
            spin_vector (List[int]): A 1D list or array of {-1, 1} representing the spin state.
            display_limit (int): The maximum number of eigenmode plots to display. Set to 0 for no plots.
        """
        spin_vector = np.asarray(spin_vector)
        if spin_vector.shape != (self.num_spins,):
            raise ValueError(f"spin_vector must be a 1D array of length {self.num_spins}")
        if not np.all(np.isin(spin_vector, [-1, 1])):
            raise ValueError("spin_vector must only contain values of -1 or 1.")

        # Loop over each eigenmode (Mattis Hamiltonian)
        list_of_phase_masks = []
        for k in range(self.num_spins):
            phase_mask = np.zeros((self.slm_height, self.slm_width))

            # The phase modulation amplitude for each spin, compensated for beam intensity.
            # This is clipped to ensure the argument of arccos is valid.
            compensated_eigvec =  self.eigvecs[:, k] # self._compensation_factors *
            alpha_ik = np.arccos(np.clip(compensated_eigvec, -1.0, 1.0))

            # Populate the phase mask for each spin's macropixel
            for i in range(self.num_spins):
                spin_state = spin_vector[i]
                amplitude = alpha_ik[i]

                row = i // self._grid_cols
                col = i % self._grid_cols
                x0 = self._grid_offset_x + col * self._macro_pix_x
                y0 = self._grid_offset_y + row * self._macro_pix_y

                # Create the checkerboard pattern for diffraction
                lx = np.arange(self._macro_pix_x)
                ly = np.arange(self._macro_pix_y)
                lx_grid, ly_grid = np.meshgrid(lx, ly)

                checkerboard = ((-1)**(lx_grid + ly_grid)) * amplitude

                # The final phase depends on the spin state and the checkerboard
                # The spin state effectively shifts the phase of one half of the checkerboard
                # relative to the other, encoding the spin information.
                #phi_block = spin_state * checkerboard # spin_state *
                                # ... inside the loop ...
                checkerboard = ((-1)**(lx_grid + ly_grid)) * amplitude

                # 1. Determine the phase offset from the spin state
                if spin_state == 1:
                    spin_phase = np.pi / 2
                else: # spin_state == -1
                    spin_phase = 3 * np.pi / 2

                # 2. Add the spin's phase to the checkerboard pattern
                phi_block = spin_phase + checkerboard

                # 3. Apply modulo for SLM display
                phase_mask[y0:y0 + self._macro_pix_y, x0:x0 + self._macro_pix_x] = phi_block % (2 * np.pi)
                # Modulo 2*pi for SLM display
                phase_mask[y0:y0 + self._macro_pix_y, x0:x0 + self._macro_pix_x] = phi_block % (2 * np.pi)

            # if k < display_limit:
            list_of_phase_masks.append(phase_mask)
            #     self._display_mask(phase_mask, k)
        return list_of_phase_masks

    def _display_mask(self, phase_mask: np.ndarray, eigenmode_index: int):
        """Helper function to plot a generated phase mask."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})

        # Plot 1: Full phase mask on the SLM
        im1 = axes[0].imshow(phase_mask, cmap='twilight', aspect='auto', vmin=0, vmax=2*np.pi)
        axes[0].set_title(f'Full Phase Mask on SLM (Eigenmode k = {eigenmode_index})')
        axes[0].set_xlabel('SLM Pixel X')
        axes[0].set_ylabel('SLM Pixel Y')
        fig.colorbar(im1, ax=axes[0], label='Phase (radians)', shrink=0.8)

        # Plot 2: Zoomed-in view of the first macropixel
        spin_to_zoom = 0 # Zoom on the first spin
        row_zoom = spin_to_zoom // self._grid_cols
        col_zoom = spin_to_zoom % self._grid_cols
        x0_zoom = self._grid_offset_x + col_zoom * self._macro_pix_x
        y0_zoom = self._grid_offset_y + row_zoom * self._macro_pix_y

        zoomed_block = phase_mask[
            y0_zoom : y0_zoom + self._macro_pix_y,
            x0_zoom : x0_zoom + self._macro_pix_x
        ]

        im2 = axes[1].imshow(zoomed_block, cmap='twilight', interpolation='nearest', vmin=0, vmax=2*np.pi)
        axes[1].set_title(f'Zoom: Spin {spin_to_zoom}')
        axes[1].set_xlabel('Pixel Offset')
        axes[1].set_ylabel('Pixel Offset')

        fig.suptitle(f'Compensated Phase Mask for Mattis Hamiltonian k = {eigenmode_index}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def run(self, spin_vector: List[int]):
        """
        A convenience method to prepare the model, then generate and display phase masks.
        """
        #self.prep()
        self.generate_phase_masks(spin_vector)


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define the interaction matrix J for the spins
    NUM_SPINS = 40
    # Example: Ferromagnetic coupling (all spins want to align)
    J_ferro = np.ones((NUM_SPINS, NUM_SPINS))
    # Example: Random interaction matrix
    np.random.seed(42)
    J_random = np.random.randn(NUM_SPINS, NUM_SPINS)
    J_random = (J_random + J_random.T) / 2 # Ensure symmetry

    # 2. Define the experimental parameters
    BEAM_SIGMA_X = 100  # Gaussian beam width in pixels
    BEAM_SIGMA_Y = 100  # Gaussian beam height in pixels
    SLM_WIDTH = 1920
    SLM_HEIGHT = 1080

    # 3. Initialize the simulation
    mattis_model = CompensatedMattisInteractions(
        J=J_random,
        beam_sigma_x=BEAM_SIGMA_X,
        beam_sigma_y=BEAM_SIGMA_Y,
        slm_width=SLM_WIDTH,
        slm_height=SLM_HEIGHT
    )

    # 4. Define a spin configuration to encode
    # For this example, we'll use an alternating spin vector
    spin_config = [-1 if i % 2 else 1 for i in range(NUM_SPINS)]

    # 5. Run the process
    # This will prepare the model and then generate and display the phase masks.
    mattis_model.run(spin_vector=spin_config)
    # 100Pi lines