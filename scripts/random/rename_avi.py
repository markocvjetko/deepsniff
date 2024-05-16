import os

def rename_avi_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.avi'):
                parent_dir = os.path.basename(root)
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, parent_dir + '.avi')
                os.rename(old_path, new_path)
                print(f"Renamed '{old_path}' to '{new_path}'")

# Replace 'directory_path' with the path of the directory where you want to start the search.
directory_path = '/scratch-local/users/markoc-haeslerlab/sniff-labeled/'
rename_avi_files(directory_path)