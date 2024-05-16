#!/bin/bash

# Root directory containing subdirectories
root_dir="/scratch-local/users/markoc-haeslerlab/sniff-pretrain-large"

# Path to your Python script
python_script='avi_to_png.py'

# count number of subdirectories
n_subdirs=$(find "$root_dir" -type d | wc -l)
echo "Number of subdirectories: $n_subdirs"
i=0

# Iterate over subdirectories in the root directory

for subdir in $(find "$root_dir" -type d); do
    #echo "Processing subdirectory: $subdir"
    if [ -d "$subdir" ]; then
    
        #check if subdirectory contains avi files (not recursive)
        avi_file=$(find "$subdir" -maxdepth 1 -type f -name "*.avi" | head -n 1)
        #echo "AVI file: $avi_file"
    
        #if avi files doesnt exist, skip
        if [ -z "$avi_file" ]; then
            echo "No AVI file found in $subdir"
            i=$((i+1))
            continue
        fi

        dest_dir="$subdir/frames"
        echo "Creating directory: $dest_dir"
        #echo "Processing AVI file: $avi_file"

        # Call your Python script with input and output directory as arguments
        python3 "$python_script" "$avi_file" "$dest_dir"
    fi
    i=$((i+1))
    echo "Processed $i out of $n_subdirs subdirectories"

done