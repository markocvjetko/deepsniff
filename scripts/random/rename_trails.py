import os

def rename_subdirs(root_dir):
    # Iterate over subdirectories in the root directory
    for index, subdir in enumerate(os.listdir(root_dir)):
        # Get the full path of the subdirectory
        subdir_path = os.path.join(root_dir, subdir)
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Create the new name based on parent directory name and index
            new_name = f"{os.path.basename(root_dir)}_{index}"
            # Construct the new path
            new_path = os.path.join(root_dir, new_name)
            # Rename the subdirectory
            os.rename(subdir_path, new_path)
            print(f"Renamed '{subdir}' to '{new_name}'")

# Provide the root directory path
root_directory = "/scratch-local/users/markoc-haeslerlab/sniff-labeled/"


#list absolut paths of all subdirectories in root_directory
subdirs = [os.path.join(root_directory, name) for name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, name))]
for subdir in subdirs:
    rename_subdirs(subdir)

# Call the function to rename subdirectories
#rename_subdirs(root_directory)