import numpy as np
import matplotlib.pyplot as plt
from typing import List

def convert_phase_mask_to_8bit(phase_mask_radians: np.ndarray) -> np.ndarray:
    """
    Converts a phase mask from radians to an 8-bit integer representation.

    This function takes a 2D NumPy array where each value represents a phase
    in radians (expected range [0, 2*pi]) and maps it to an integer value
    in the range [0, 255].

    Args:
        phase_mask_radians (np.ndarray): A 2D NumPy array containing phase values
                                         in radians.

    Returns:
        np.ndarray: A 2D NumPy array of dtype=uint8 with values from 0 to 255.
    """
    if not isinstance(phase_mask_radians, np.ndarray) or phase_mask_radians.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")

    # 1. Ensure all values are within the [0, 2*pi) range using modulo.
    #    This handles cases where input values might be slightly outside the range.
    normalized_mask_radians = phase_mask_radians % (2 * np.pi)

    # 2. Normalize the range [0, 2*pi] to [0, 1] by dividing.
    normalized_mask_unit = normalized_mask_radians / (2 * np.pi)
    
    # 3. Scale to [0, 255], round to the nearest integer, and cast to uint8.
    phase_mask_8bit = (normalized_mask_unit * 255).round().astype(np.uint8)
    
    return phase_mask_8bit

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Create a sample phase mask in radians.
    #    This simulates the output you would get from the previous class.
    #    Here, we create a simple gradient from 0 to 2*pi.
    width, height = 1920, 1080
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    sample_radian_mask = (X + Y) % (2 * np.pi)

    print("--- Input Mask (Radians) ---")
    print(f"Data type: {sample_radian_mask.dtype}")
    print(f"Shape: {sample_radian_mask.shape}")
    print(f"Minimum value: {sample_radian_mask.min():.4f}")
    print(f"Maximum value: {sample_radian_mask.max():.4f}")

    # 2. Use the standalone function to convert the mask.
    print("\nConverting mask to 8-bit...")
    converted_8bit_mask = convert_phase_mask_to_8bit(sample_radian_mask)
    print("Conversion complete.")

    # 3. Verify the properties of the converted mask.
    print("\n--- Output Mask (8-bit) ---")
    print(f"Data type: {converted_8bit_mask.dtype}")
    print(f"Shape: {converted_8bit_mask.shape}")
    print(f"Minimum value: {converted_8bit_mask.min()}")
    print(f"Maximum value: {converted_8bit_mask.max()}")

    # 4. Display the original and converted masks for comparison.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Display original radian mask
    im1 = axes[0].imshow(sample_radian_mask, cmap='twilight', vmin=0, vmax=2*np.pi)
    axes[0].set_title('Original Mask (Radians)')
    axes[0].set_xlabel('Pixel X')
    axes[0].set_ylabel('Pixel Y')
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Phase (radians)')

    # Display converted 8-bit mask
    im2 = axes[1].imshow(converted_8bit_mask, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Converted Mask (8-bit Integer)')
    axes[1].set_xlabel('Pixel X')
    axes[1].set_ylabel('Pixel Y')
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Phase (8-bit integer value)')

    plt.tight_layout()
    plt.show()
