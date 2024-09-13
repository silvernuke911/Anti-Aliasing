import numpy as np
import cv2
from scipy.ndimage import convolve

def gaussian_kernel(size: int, sigma: float = 1.0):
    """Generate a Gaussian kernel."""
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1d[i] = np.exp(-(kernel_1d[i]**2) / (2 * sigma**2))
    kernel_1d /= np.sum(kernel_1d)
    
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= np.sum(kernel_2d)
    
    return kernel_2d

def apply_antialiasing(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0):
    """Apply anti-aliasing to an image using Gaussian convolution."""
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Apply the convolution
    if len(image.shape) == 2:  # Grayscale image
        result = convolve(image, kernel)
    else:  # Color image
        result = np.zeros_like(image)
        for i in range(3):  # Apply to each channel
            result[:, :, i] = convolve(image[:, :, i], kernel)
    
    return result

# Example usage:
image_path = r"C:\Users\verci\Downloads\telescope1.png"
image = cv2.imread(image_path)

# Apply anti-aliasing
anti_aliased_image = apply_antialiasing(image, kernel_size=4, sigma=2)

# Save or display the result
cv2.imwrite('anti_aliased_image7.png', anti_aliased_image)
cv2.imshow('Anti-Aliased Image', anti_aliased_image)
cv2.waitKey(0)
cv2.destroyAllWindows()