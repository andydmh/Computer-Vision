import os
import glob

# Requrement: imagemagic
# For mac run: brew install imagemagick
# For Debian or Ubuntu run: sudo apt-get install imagemagick

def convert_single_image(input_image_path, output_image_path):
    command = 'convert '
    command += input_image_path + ' '
    command += output_image_path
    print(command)
    os.system(command)

def convert_dir(input_dir, output_dir, orig_ext, target_ext):
    all_image_paths = glob.glob(input_dir + '/*.' + orig_ext)

    for image_path in all_image_paths:
        base=os.path.basename(image_path)
        name = os.path.splitext(base)[0]
        output_image_path = output_dir + '/' + name + '.' + target_ext
        
        convert_single_image(image_path, output_image_path)

def main():
    input_dir = 'test_in'
    output_dir = 'test_out'
    orig_ext = 'tif'
    target_ext = 'jpg'

    convert_dir(input_dir, output_dir, orig_ext, target_ext)

if __name__ == '__main__':
    main()
