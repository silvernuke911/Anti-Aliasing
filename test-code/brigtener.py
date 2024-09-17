import cv2
import numpy as np

import matplotlib.pyplot as plt
def plt_show(image):
    plt.imshow(image)
    plt.show()

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

input_path = 'test2.png_contrasted.png'
output_path = f'{input_path}_brigtened.png'

# Load an image
input_image = cv2.imread(input_path)  # Change 'input_image.jpg' to your image path

# Check if the image was loaded successfully
if input_image is None:
    print("Error loading image!")
else:
    # Apply the brightness adjustment
    output_image = adjust_brightness(input_image, threshold=200)

    # Display the original and adjusted images
    plt_show(input_image)
    plt_show(output_image)

    # Save the result
    cv2.imwrite(output_path, output_image)

