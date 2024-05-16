#!/bin/bash

# Root directory containing subdirectories

#root dirs are /mnt/haeslerlab/haeslerlabwip2020/raw_data/FLIR and /mnt/haeslerlab/haeslerlabwip2023/raw_data/FLIR

root_dirs=("/mnt/haeslerlab/haeslerlabwip2020/raw_data/FLIR" "/mnt/haeslerlab/haeslerlabwip2023/raw_data/FLIR")

output_dir="/scratch-local/users/markoc-haeslerlab/sniff-pretrain-large/"
# Path to your Python script
python_script='seq_to_avi.py'

# Iterate over subdirectories in the root directory
for root_dir in "${root_dirs[@]}"; do
    for subdir in "$root_dir"/*; do
        echo "Processing $subdir"
        if [ -d "$subdir" ]; then
            # Call your Python script with input and output directory as arguments
            python3 "$python_script" "$subdir" "$output_dir" --num_videos 10
        fi
    done
done