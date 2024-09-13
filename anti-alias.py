import numpy as np
import cv2
import os
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
filename = 'telescope1_anti-aliased.png'
# Apply anti-aliasing
anti_aliased_image = apply_antialiasing(image, kernel_size=4, sigma=2)


def check_and_prompt_overwrite(filename):
    
    def get_user_input(prompt, valid_responses, invalid_input_limit=3):
        attempts = 0
        while attempts < invalid_input_limit:
            response = input(prompt).lower().strip()
            if response in valid_responses:
                return response
            print("Invalid input. This is in handle user")
            attempts += 1
        print("Exceeded maximum invalid input limit. Operation aborted.")
        return 'ABORT'

    def handle_file_exists(filename):
        while True:
            response = get_user_input(f"{filename} already exists, do you want to overwrite it? (Y/N): ", ['yes', 'y', 'no', 'n'],5)
            if response in ['yes', 'y']:
                print("Proceeding with overwrite...")
                return True, filename
            elif response in ['no', 'n']:
                return handle_rename(filename)
            elif response == 'ABORT':
                return False, filename

    def handle_rename(filename):
        while True:
            rename_response = get_user_input('Would you like to rename it? (Y/N): ', ['yes', 'y', 'no', 'n'],3)
            if rename_response in ['yes', 'y']:
                return get_new_filename()
            elif rename_response in ['no', 'n']:
                print('Operation aborted.')
                return False, filename
            elif rename_response == 'ABORT':
                return False, filename

    def get_new_filename():
        while True:
            new_filename = input('Input the new name of the file: ').strip() + '.gif'
            if new_filename == 'ABORT.gif':
                print('Operation aborted.')
                return False, new_filename
            if not os.path.isfile(new_filename):
                print(f'Proceeding with creation of {new_filename}')
                return True, new_filename
            print(f'{new_filename} already exists. Please put another file name.')

    if os.path.isfile(filename):
        return handle_file_exists(filename)
    return True, filename

# Save or display the result
overwrite, filename = check_and_prompt_overwrite(filename)

if overwrite == True:
    cv2.imwrite(filename, anti_aliased_image)
    cv2.imshow('Anti-Aliased Image', anti_aliased_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()