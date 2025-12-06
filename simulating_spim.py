import diffractsim
from diffractsim import MonochromaticField, ApertureFromImage, mm, nm, cm
import numpy as np
from PIL import Image

def create_gaussian_amplitude(size_px, sigma_px):
    """Generates a Gaussian amplitude profile image."""
    x = np.linspace(-size_px//2, size_px//2, size_px)
    y = np.linspace(-size_px//2, size_px//2, size_px)
    X, Y = np.meshgrid(x, y)
    # Gaussian function: exp(-(x^2 + y^2) / (2*sigma^2))
    # Note: Amplitude is sqrt(Intensity), but often we map 0-255 linearly to transmission.
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma_px**2))
    # Normalize to 0-255
    im_data = (gaussian * 255).astype(np.uint8)
    return Image.fromarray(im_data)

def create_slm_phase_pattern(size_px):
    """Generates a dummy SLM phase pattern (e.g., a vortex or grating)."""
    x = np.linspace(-size_px//2, size_px//2, size_px)
    y = np.linspace(-size_px//2, size_px//2, size_px)
    X, Y = np.meshgrid(x, y)
    # Example: Spiral phase (Vortex)
    phase = np.arctan2(Y, X) 
    # Normalize -pi to pi -> 0 to 255
    im_data = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    return Image.fromarray(im_data)

# --- Simulation Setup ---

# 1. Parameters
WAVELENGTH = 632.8 * nm
GRID_SIZE = 10 * mm
RESOLUTION = 1024  # pixels

# 2. Create the optical field
F = MonochromaticField(
    wavelength=WAVELENGTH,
    extent_x=GRID_SIZE,
    extent_y=GRID_SIZE,
    Nx=RESOLUTION,
    Ny=RESOLUTION
)

# 3. Prepare Masks (Gaussian Source + SLM Phase)
# Create a Gaussian spot with sigma = 1/6th of the grid
gaussian_img = create_gaussian_amplitude(RESOLUTION, RESOLUTION / 6)
gaussian_img.save("gaussian_amp.png")

# Create a Phase Mask (simulating the SLM)
slm_img = create_slm_phase_pattern(RESOLUTION)
slm_img.save("slm_phase.png")

# 4. Apply the SLM and Illumination to the Field
# We load the Gaussian image as the 'amplitude' (transmittance)
# and the SLM image as the 'phase' (delay).
F.add(ApertureFromImage(
    amplitude_mask_path="gaussian_amp.png",  # Defines the Gaussian Beam shape
    phase_mask_path="slm_phase.png",        # Defines the SLM pattern
    image_size=(GRID_SIZE, GRID_SIZE),
    simulation=F
))

# 5. Propagate and Visualize
print("Propagating...")
F.propagate(50 * cm)  # Propagate 50 cm

print("Computing colors...")
rgb = F.get_colors()
F.plot_colors(rgb)