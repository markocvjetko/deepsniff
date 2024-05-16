#!/bin/bash

root_dir="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain"

# Iterate over subdirectories in the root directory
for subdir in "$root_dir"/*/; do
    # Count the number of CSV files in the subdirectory
    num_csv_files=$(find "$subdir" -maxdepth 1 -type f -name "*.csv" | wc -l)
    
    # Check if there are more than one CSV files
    if [ "$num_csv_files" -gt 1 ]; then
        # Extract the subdirectory name from its path
        subdir_name=$(basename "$subdir")
        echo "$subdir_name"
    fi
done