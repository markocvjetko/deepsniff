#!/bin/bash

# Define root directory and new root directory
root_dir="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain"
new_root_dir="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain-test"

# Create new root directory if it doesn't exist
mkdir -p "$new_root_dir"

# Get a list of subdirectories in the root directory
subdirs=("$root_dir"/*)

# Shuffle the list of subdirectories randomly
shuf -e "${subdirs[@]}" | head -n 10 | while read -r subdir; do
    # Copy the subdirectory to the new root directory
    cp -r "$subdir" "$new_root_dir"
done

echo "Subdirectories copied successfully."