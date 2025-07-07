import numpy as np
import matplotlib.pyplot as plt

class compensated_mattis_interactions:
    def __init__(self, J, beam_sigma_x, beam_sigma_y, SLM_WIDTH = 1920, SLM_HEIGHT = 1080):
        self.SLM_WIDTH = SLM_WIDTH
        self.SLM_HEIGHT = SLM_HEIGHT
        self.n_spins = np.shape(J)[0]
        self.J = J
        self.beam_sigma_x = beam_sigma_x
        self.beam_sigma_y = beam_sigma_y
        self.center_x = SLM_WIDTH / 2
        self.center_y = SLM_HEIGHT / 2
        self._choose_macropixel_size()
        self._compute_intensity_map()
        self._compute_compensation_factors()
        self.eigvals = None
    def _choose_macropixel_size(self):
        tile_x = self.SLM_WIDTH // self.n_spins
        tile_y = self.SLM_HEIGHT // self.n_spins
        min_tile = min(tile_x, tile_y)
        if min_tile < 2:
            raise ValueError(
                f"SLM too small for {self.n_spins} spins: max square block size = {min_tile}x{min_tile}."
            )
        self.MACRO_PIX = min_tile
        self.MACRO_PIX_X = min_tile
        self.MACRO_PIX_Y = min_tile

    def _compute_intensity_map(self):
        xs = np.arange(self.SLM_WIDTH)
        ys = np.arange(self.SLM_HEIGHT)
        X, Y = np.meshgrid(xs, ys)
        self.I = np.exp(-2 * (((X - self.center_x)**2 / self.beam_sigma_x**2) +
                              ((Y - self.center_y)**2 / self.beam_sigma_y**2)))

    def _compute_compensation_factors(self):
        Ii = np.zeros(self.n_spins)
        for i in range(self.n_spins):
            x_start = i * self.MACRO_PIX_X
            x_end = x_start + self.MACRO_PIX_X
            y_start = i * self.MACRO_PIX_Y
            y_end = y_start + self.MACRO_PIX_Y
            block = self.I[y_start:y_end, x_start:x_end]
            Ii[i] = block.mean()
        wi = np.max(Ii) / Ii
        ci = np.sqrt(wi)
        self.Ai = ci / np.max(ci)


    def get_mattis_interactions(self):
        if not np.allclose(self.J, self.J.T):
            max_diff = np.max(np.abs(self.J - self.J.T))
            raise ValueError(
                f"Interaction matrix is not symmetric (max asymmetry={max_diff:.3e}). Please provide a symmetric matrix."
            )
        print("\nInput Interaction Matrix J (symmetric):")
        print(self.J)

        self.eigvals, self.eigvecs = np.linalg.eigh(self.J)
