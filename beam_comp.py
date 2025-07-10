import numpy as np
import matplotlib.pyplot as plt

class CompensatedMattisInteractions:
    def __init__(self, J, beam_sigma_x, beam_sigma_y, SLM_WIDTH=1920, SLM_HEIGHT=1080):
        self.SLM_WIDTH = SLM_WIDTH
        self.SLM_HEIGHT = SLM_HEIGHT
        self.n_spins = J.shape[0]
        self.J = J
        self.beam_sigma_x = beam_sigma_x
        self.beam_sigma_y = beam_sigma_y
        self.center_x = SLM_WIDTH / 2
        self.center_y = SLM_HEIGHT / 2

        self.eigvals = None
        self.eigvecs = None
        self.grid_cols = None
        self.grid_rows = None

        self._choose_macropixel_size()
        self._compute_intensity_map()
        self._compute_compensation_factors()

    def _choose_macropixel_size(self):
        tile_x = self.SLM_WIDTH // self.n_spins
        tile_y = self.SLM_HEIGHT // self.n_spins
        min_tile = max(2, min(tile_x, tile_y))

        self.MACRO_PIX_X = min_tile
        self.MACRO_PIX_Y = min_tile

    def _compute_intensity_map(self):
        xs = np.arange(self.SLM_WIDTH)
        ys = np.arange(self.SLM_HEIGHT)
        X, Y = np.meshgrid(xs, ys)
        self.I = np.exp(-(((X - self.center_x) ** 2) / (2 * self.beam_sigma_x ** 2) +
                          ((Y - self.center_y) ** 2) / (2 * self.beam_sigma_y ** 2)))

    def _compute_compensation_factors(self):
        self.grid_cols = int(np.ceil(np.sqrt(self.n_spins)))
        self.grid_rows = int(np.ceil(self.n_spins / self.grid_cols))

        Ii = np.zeros(self.n_spins)
        for idx in range(self.n_spins):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            x0 = col * self.MACRO_PIX_X
            y0 = row * self.MACRO_PIX_Y
            block = self.I[y0:y0 + self.MACRO_PIX_Y, x0:x0 + self.MACRO_PIX_X]
            Ii[idx] = block.mean()

        wi = np.max(Ii) / Ii
        ci = np.sqrt(wi)
        self.Ai = ci / np.max(ci)

    def get_mattis_interactions(self):
        if not np.allclose(self.J, self.J.T):
            max_diff = np.max(np.abs(self.J - self.J.T))
            raise ValueError(f"Interaction matrix is not symmetric (max asymmetry={max_diff:.3e})")

        self.eigvals, self.eigvecs = np.linalg.eigh(self.J)

    def encode_and_display(self, spin_vector):
        spin_vector = np.asarray(spin_vector)
        if spin_vector.ndim != 1 or spin_vector.shape[0] != self.n_spins:
            raise ValueError(f"spin_vector must be a 1D array of length {self.n_spins}")
        if not np.all(np.isin(spin_vector, [-1, 1])):
            raise ValueError("spin_vector must only contain -1 or 1")

        for k in range(self.n_spins):
            alpha_ik = np.arccos(np.clip(self.Ai * self.eigvecs[:, k], -1.0, 1.0))

            phase_mask = np.zeros((self.grid_rows * self.MACRO_PIX_Y, self.grid_cols * self.MACRO_PIX_X))

            for i in range(self.n_spins):
                spin = spin_vector[i]
                a = alpha_ik[i]

                row = i // self.grid_cols
                col = i % self.grid_cols
                x0 = col * self.MACRO_PIX_X
                y0 = row * self.MACRO_PIX_Y

                for ly in range(self.MACRO_PIX_Y):
                    for lx in range(self.MACRO_PIX_X):
                        phi_raw = spin * (np.pi / 2) + ((-1) ** (lx + ly)) * a
                        phi_slm = phi_raw % (2 * np.pi)
                        phase_mask[y0 + ly, x0 + lx] = phi_slm

            plt.figure(figsize=(10, 6))
            plt.imshow(phase_mask, cmap='twilight', aspect='auto')
            plt.colorbar(label='Phase (radians)')
            plt.title(f'Compensated Phase Mask for Mattis Hamiltonian k = {k}')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')
            plt.tight_layout()
            plt.show()

            plt.imshow(phase_mask[y0:y0+self.MACRO_PIX_Y, x0:x0+self.MACRO_PIX_X], cmap='twilight', interpolation='nearest')
            plt.title(f'Macroblock for spin {i}, k={k}')
            plt.colorbar()
            plt.show()



    def prep(self):
        self._choose_macropixel_size()
        self._compute_intensity_map()
        self._compute_compensation_factors()
        self.get_mattis_interactions()

    def run(self, spin_vector):
        self.encode_and_display(spin_vector)


if __name__ == "__main__":
    J = np.array([[1, 0.5], [0.5, 1]])
    beam_sigma_x = 300
    beam_sigma_y = 300
    spins = np.array([1, -1])

    model = CompensatedMattisInteractions(J, beam_sigma_x, beam_sigma_y)
    model.prep()
    model.run(spins)
