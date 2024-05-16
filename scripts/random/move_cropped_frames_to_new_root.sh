#!/bin/bash


#t_start
t_start=$(date +%s)

# Define the root directory
root_dir="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain"

# Define the destination directory
new_root="/scratch-local/users/markoc-haeslerlab/sniff-pretrain"

# Create the destination directory if it doesn't exist
mkdir -p "$new_root"

# Find all the cropped_frames dirs and copy their relative paths to the new root (the resulting path should be the same as if
#accessing them from root dir)

# Find all the cropped_frames directories in the root directory
counter=0
find "$root_dir" -type d -name "cropped_frames" | while read -r cropped_frames_dir; do
    # Extract the relative path of the cropped_frames directory (without cropped_frames in the path)
    relative_path=$(dirname "${cropped_frames_dir/$root_dir/}")

    # Define the destination directory path
    dest_dir="$new_root/$relative_path"
    # Create the destination directory
    mkdir -p "$dest_dir"
    
    #print path and counter
    echo "$counter: $cropped_frames_dir"
    echo "$counter: $dest_dir"
    ((counter++))

    # Move the cropped_frames directory to the destination directory
    cp -r "$cropped_frames_dir" "$dest_dir"
done

#t_end
t_end=$(date +%s)
echo "Time elapsed: $((t_end - t_start)) seconds"