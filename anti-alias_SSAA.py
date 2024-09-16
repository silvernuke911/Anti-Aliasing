from PIL import Image
import os

def supersample_antialiasing(input_image_path, output_image_path, supersample_factor=2):
    # Open the original image
    img = Image.open(input_image_path)

    # If the image is in RGBA mode (with alpha channel), convert it to RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Get original dimensions
    original_width, original_height = img.size

    # Calculate supersampled dimensions
    supersample_width = original_width * supersample_factor
    supersample_height = original_height * supersample_factor

    # Resize the image to the supersampled dimensions (higher resolution)
    supersampled_img = img.resize((supersample_width, supersample_height), Image.NEAREST)

    # Downscale it back to original size with antialiasing
    antialiased_img = supersampled_img.resize((original_width, original_height), Image.LANCZOS)

    # Save the final antialiased image
    antialiased_img.save(output_image_path)

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


# Example usage
input_image_path = "telescope1.png"
output_image_path = "tel1_antialiased.png"
supersample_factor = 4  # You can change this to 3, 4, etc.

overwrite, filename = check_and_prompt_overwrite(output_image_path)

if overwrite == True:
    supersample_antialiasing(input_image_path, output_image_path, supersample_factor)
    print('\nImage saved!')
