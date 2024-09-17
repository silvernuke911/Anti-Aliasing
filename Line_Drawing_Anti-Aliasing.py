import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def plt_show(image):
    plt.imshow(image)
    plt.show()

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
            response = get_user_input(f"{filename} already exists, do you want to overwrite it? (Y/N): ", ['yes', 'y', 'no', 'n'], 5)
            if response in ['yes', 'y']:
                print                       ('\nx---------------------WARNING---------------------x')
                sure_response = get_user_input("Are you really sure you want to OVERWRITE it? (Y/N): ", ['yes', 'y', 'no', 'n'], 3)
                if sure_response in ['yes', 'y']:
                    print("Proceeding with overwrite...")
                    return True, filename
                elif sure_response in ['no', 'n']:
                    print('Operation aborted.')
                    return False, filename
                elif sure_response == 'ABORT':
                    return False, filename
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

# Function to increase contrast and darken the dark areas
def adjust_contrast_and_darkness(image, alpha=1.5, beta=-30, gamma=1.2):
    """
    Adjusts the contrast and darkens dark parts of the image.
    :param image: Input image (numpy array)
    :param alpha: Contrast control (1.0 - 3.0)
    :param beta: Brightness control (-100 to 100)
    :param gamma: Darkens darker areas (use >1 to darken)
    :return: Image with adjusted contrast and darkened dark parts
    """
    # Increase contrast and adjust brightness
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Apply gamma correction to darken the dark parts
    # Normalize image to [0, 1], apply gamma correction, and rescale back to [0, 255]
    gamma_corrected = np.power(adjusted / 255.0, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)

    return gamma_corrected

# Function to adjust brightness based on the condition
def adjust_brightness(image, threshold=200):
    """
    Adjusts the brightness of the image. If the brightness of a pixel is higher than
    the threshold, it rounds it up to 255.
    :param image: Input image (numpy array)
    :param threshold: Brightness threshold for rounding to 255
    :return: Image with adjusted brightness
    """
    # Convert the image to grayscale to compute the brightness
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the pixels with brightness greater than the threshold and set them to 255
    adjusted = np.where(grayscale > threshold, 255, grayscale)

    # Convert back to BGR to keep the output in the same format as the input
    adjusted_bgr = cv2.merge([adjusted, adjusted, adjusted])

    return adjusted_bgr

input_path = 'teletype.png'
output_path = f'{input_path}_anti-aliased.png'

if __name__ == '__main__':
    input_image = cv2.imread(input_path) 
    if input_image is None:
        print("Error loading image!")
    else:
        # Checking for name duplication
        overwrite, output_path = check_and_prompt_overwrite(output_path)
        output_image = input_image
        if overwrite == True:
            plt_show(output_image)
            # Gaussian blurring pass
            output_image = apply_antialiasing(output_image, kernel_size=3, sigma=2)
            print('Gaussian blurring pass')
            plt_show(output_image)
            # Contrasting pass
            print('Contrasting pass')
            output_image = adjust_contrast_and_darkness(output_image, alpha=1.05, beta=-30, gamma=2)
            plt_show(output_image)
            # Brightening pass
            print('Brightening pass')
            output_image = adjust_brightness(output_image, threshold=200)
            plt_show(output_image)

            # Display the original and adjusted images
            print('Final Display')
            plt_show(input_image)
            plt_show(output_image)

            # Save the result
            print('Result saved!')
            cv2.imwrite(output_path, output_image)
        
