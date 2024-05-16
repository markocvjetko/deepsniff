#!/bin/bash

# Define root directory
root_dir="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain"

# Initialize a counter for directories with "dlc" subdirectory
dlc_counter=0

# Iterate over subdirectories in the root directory
for subdir in "$root_dir"/*/; do
    # Check if the current subdirectory contains a "dlc" subdirectory
    if [ -d "${subdir}dlc" ]; then
        dlc_counter=$((dlc_counter + 1))
    fi
done

echo "Number of subdirectories with 'dlc' subdirectory: $dlc_counter"