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
        
        # [REVIEW] Pre-computed values for optimization
        self._base_checkerboard: np.ndarray = None
        self._spin_slices: List[Tuple[slice, slice]] = []
        self._spin_coords_x: np.ndarray = np.zeros(self.num_spins)
        self._spin_coords_y: np.ndarray = np.zeros(self.num_spins)


    def _setup_layout(self):
        """
        Determines an optimal rectangular macropixel layout to fill the SLM area
        and pre-computes spin positions and slices.
        """
        best_layout = (0, 0)
        max_area = 0

        for rows in range(1, self.num_spins + 1):
            cols = math.ceil(self.num_spins / rows)
            
            # Check if this layout fits on the SLM at all
            if cols > self.slm_width or rows > self.slm_height:
                continue
                
            # Aspect ratio check to favor layouts matching SLM aspect
            slm_aspect = self.slm_width / self.slm_height
            grid_aspect = (cols * self.slm_width / cols) / (rows * self.slm_height / rows) # This logic is flawed
            # Let's stick to the user's original logic which maximizes area
            
            if cols * self.slm_height > rows * self.slm_width: # Original aspect check
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
             raise RuntimeError(f"Failed to find a macropixel grid for {self.num_spins} spins on a {self.slm_width}x{self.slm_height} SLM.")

        self._macro_pix_x = self.slm_width // self._grid_cols
        self._macro_pix_y = self.slm_height // self._grid_rows

        total_width = self._grid_cols * self._macro_pix_x
        total_height = self._grid_rows * self._macro_pix_y

        grid_offset_x = (self.slm_width - total_width) // 2
        grid_offset_y = (self.slm_height - total_height) // 2

        # [REVIEW] Vectorized pre-computation of spin positions and slices
        for i in range(self.num_spins):
            row = i // self._grid_cols
            col = i % self._grid_cols

            x0 = grid_offset_x + col * self._macro_pix_x
            y0 = grid_offset_y + row * self._macro_pix_y
            
            self._spin_coords_x[i] = x0 + self._macro_pix_x / 2
            self._spin_coords_y[i] = y0 + self._macro_pix_y / 2
            self._spin_slices.append((slice(y0, y0 + self._macro_pix_y), slice(x0, x0 + self._macro_pix_x)))

    def _precompute_checkerboard(self):
        """
        [REVIEW] Pre-computes the base checkerboard pattern once.
        """
        lx = np.arange(self._macro_pix_x)
        ly = np.arange(self._macro_pix_y)
        lx_grid, ly_grid = np.meshgrid(lx, ly)
        self._base_checkerboard = ((-1)**(lx_grid + ly_grid)).astype(np.float32)

    def _compute_intensity_map(self):
        """
        Computes the 2D Gaussian intensity profile of the beam on the SLM.
        """
        x = np.arange(self.slm_width)
        y = np.arange(self.slm_height)
        X, Y = np.meshgrid(x, y)

        self._intensity_map = np.exp(
            -(((X - self._center_x)**2) / (2 * self.beam_sigma_x**2) +
              ((Y - self._center_y)**2) / (2 * self.beam_sigma_y**2))
        )

    def _compute_compensation_factors(self):
        """
        [REVIEW] Computes compensation factors (1/sqrt(I)) for each spin
        in a vectorized manner.
        """
        # Calculate intensity at the center of each macropixel
        intensities_at_spins = np.exp(
            -(((self._spin_coords_x - self._center_x)**2) / (2 * self.beam_sigma_x**2) +
              ((self._spin_coords_y - self._center_y)**2) / (2 * self.beam_sigma_y**2))
        )
        
        intensities_at_spins[intensities_at_spins < 1e-9] = 1e-9  # Avoid division by zero

        # [REVIEW] Compensation factor is 1 / E_field, where E ~ sqrt(I)
        # We don't normalize this here; we normalize the *final* vector in generate_phase_masks
        self._compensation_factors = 1.0 / np.sqrt(intensities_at_spins)


    def _perform_eigendecomposition(self):
        """
        Performs eigendecomposition on the symmetric interaction matrix J.
        """
        self.eigvals, self.eigvecs = np.linalg.eigh(self.J)

    def prep(self):
        """
        Runs all necessary setup calculations in the correct order.
        """
        print("Preparing model...")
        self._setup_layout()
        print(f"  - SLM Layout: {self._grid_rows} rows x {self._grid_cols} cols")
        print(f"  - Macropixel Size: {self._macro_pix_x} x {self._macro_pix_y} pixels")
        
        self._precompute_checkerboard() # [REVIEW] Pre-compute checkerboard
        
        self._compute_intensity_map() # Good for visualization
        self._compute_compensation_factors() # [REVIEW] Now vectorized
        self._perform_eigendecomposition()
        print("Preparation complete.")

    def generate_phase_masks(
        self,
        spin_vector: List[int],
    ) -> List[np.ndarray]:
        """
        Generates the phase mask for each Mattis Hamiltonian (eigenmode k)
        based on a given spin configuration.

        Args:
            spin_vector (List[int]): A 1D list or array of {-1, 1} representing the spin state.
            
        Returns:
            List[np.ndarray]: A list of phase masks, one for each eigenmode.
        """
        spin_vector = np.asarray(spin_vector)
        if spin_vector.shape != (self.num_spins,):
            raise ValueError(f"spin_vector must be a 1D array of length {self.num_spins}")
        if not np.all(np.isin(spin_vector, [-1, 1])):
            raise ValueError("spin_vector must only contain values of -1 or 1.")

        # [REVIEW] Pre-compute spin phases
        spin_phases = np.where(spin_vector == 1, np.pi / 2, 3 * np.pi / 2)

        list_of_phase_masks = []
        for k in range(self.num_spins):
            phase_mask = np.zeros((self.slm_height, self.slm_width), dtype=np.float32)

            # [REVIEW] New normalization logic
            # 1. Calculate ideal target amplitudes (compensated)
            target_amplitudes = self._compensation_factors * self.eigvecs[:, k]
            
            # 2. Find the max amplitude required
            max_abs_val = np.max(np.abs(target_amplitudes))
            if max_abs_val < 1e-9:
                max_abs_val = 1.0 # Avoid 0/0 for zero-eigenvectors
            
            # 3. Normalize the *entire* vector so it fits in [-1, 1]
            # This scales the whole mode's strength, which is physically correct.
            normalized_amplitudes = target_amplitudes / max_abs_val

            alpha_ik = np.arccos(normalized_amplitudes) # Already clipped bw -1,1 by normalization

            # Populate the phase mask for each spin's macropixel
            for i in range(self.num_spins):
                amplitude = alpha_ik[i]
                spin_phase = spin_phases[i]
                mask_slice = self._spin_slices[i] # Get pre-computed slice

                # [REVIEW] Use pre-computed checkerboard
                checkerboard = self._base_checkerboard * amplitude
                
                phi_block = spin_phase + checkerboard

                phase_mask[mask_slice] = phi_block % (2 * np.pi)
            
            list_of_phase_masks.append(phase_mask)
            
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
        spin_to_zoom = 0 
        zoomed_block = phase_mask[self._spin_slices[spin_to_zoom]]

        im2 = axes[1].imshow(zoomed_block, cmap='twilight', interpolation='nearest', vmin=0, vmax=2*np.pi)
        axes[1].set_title(f'Zoom: Spin {spin_to_zoom}')
        axes[1].set_xlabel('Pixel Offset')
        axes[1].set_ylabel('Pixel Offset')

        fig.suptitle(f'Compensated Phase Mask for Mattis Hamiltonian k = {eigenmode_index}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def run(self, spin_vector: List[int], display_limit: int = 5):
        """
        A convenience method to prepare the model, then generate and display phase masks.
        """
        self.prep() # [REVIEW] Uncommented this line
        all_masks = self.generate_phase_masks(spin_vector)
        
        print(f"Generated {len(all_masks)} phase masks.")
        
        if display_limit > 0:
            print(f"Displaying first {min(display_limit, self.num_spins)} masks...")
            for k in range(min(display_limit, self.num_spins)):
                self._display_mask(all_masks[k], k)


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define the interaction matrix J for the spins
    NUM_SPINS = 40
    # Example: Random interaction matrix
    np.random.seed(42)
    J_random = np.random.randn(NUM_SPINS, NUM_SPINS)
    J_random = (J_random + J_random.T) / 2 # Ensure symmetry

    # 2. Define the experimental parameters
    BEAM_SIGMA_X = 600  # [REVIEW] Increased sigma to cover more spins
    BEAM_SIGMA_Y = 600  # (Original 100 was very narrow for a 1920-wide SLM)
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
    spin_config = [-1 if i % 2 else 1 for i in range(NUM_SPINS)]

    # 5. Run the process
    # [REVIEW] Call prep() and generate_phase_masks() explicitly
    # for better control, or just use the .run() method.
    
    # Option 1: Use the run() method
    mattis_model.run(spin_vector=spin_config, display_limit=3)

    # Option 2: Manual control (good for notebooks)
    # mattis_model.prep()
    # all_masks = mattis_model.generate_phase_masks(spin_vector=spin_config)
    # print(f"Generated {len(all_masks)} phase masks.")
    #
    # # Display the first 3 masks
    # for k in range(min(3, NUM_SPINS)):
    #     mattis_model._display_mask(all_masks[k], k)