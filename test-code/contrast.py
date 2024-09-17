import cv2
import numpy as np
import matplotlib.pyplot as plt

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

input_path = 'test2.png'
output_path= f'{input_path}_contrasted.png'
# Load an image from file
input_image = cv2.imread(input_path)  # Change 'input_image.jpg' to your image path

def plt_show(image):
    plt.imshow(image)
    plt.show()
# Check if the image was loaded successfully
if input_image is None:
    print("Error loading image!")
else:
    # Adjust contrast and darken dark parts
    output_image = adjust_contrast_and_darkness(input_image, alpha=1.05, beta=-30, gamma=2)

    # Display the original and adjusted images
    plt_show(input_image)
    plt_show(output_image)
    
    # Save the result
    cv2.imwrite(output_path, output_image)  # Save the adjusted image

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# kernel = 3, sigma = 2
# alpha =1.05
# beta = -30
# gamma = 2
