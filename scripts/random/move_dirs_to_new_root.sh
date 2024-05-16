#!/bin/bash

# List of substrings to search for
substrings=(
    "RDP136" "RDP154" "RDP167" "LK050" "LK002"
    "KK089" "KK061" "KK036" "FN005"
    "ES010" "CA301" "CA303" "ES015"
)
root_dir="/scratch-local/users/markoc-haeslerlab/sniff-pretrain-large"
# Function to move subdirectories containing the given substrings to the root directory
move_directories() {
    for substring in "${substrings[@]}"; do
        find "$root_dir" -mindepth 1 -maxdepth 1 -type d -name "*$substring*" -exec mv -t "$1" {} +
    done
}

# Replace 'root_dir' with the directory path where you want to perform this operation
move_directories "/scratch-local/users/markoc-haeslerlab/sniff-labeled-test"