import os
from argparse import ArgumentParser





def main():
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to the folder containing vidoes')
    parser.add_argument('--annotation_folder', type=str, required=True, help='Path to the folder where the annotation files are saved')
    args = parser.parse_args()

    # List all the subfolders in the annotation folder
    trail_folders = [f.path for f in os.scandir(args.annotation_folder) if f.is_dir()]

    #the dataset folder contains trail folders. Copy the dataset folder contents to the annotation folder

    for trail_folder in trail_folders:
        trail_name = os.path.basename(trail_folder)
        #uppercase
        trail_name = trail_name.upper()
        print('Copying trail folder: {}'.format(trail_name))

        # Copy the files from the dataset folder to the annotation folder
        os.system('cp -r {} {}'.format(os.path.join(args.dataset_folder, trail_name), args.annotation_folder))

if __name__ == '__main__':
    main()