#!/bin/bash

# This script randomly selects one experiment directory from each session directory in the source root

source_roots=("/mnt/haeslerlab/haeslerlabwip2020/raw_data/FLIR" \
              "/mnt/haeslerlab/haeslerlabwip2023/raw_data/FLIR" \
              "/mnt/haeslerlab/haeslerlabwip2024/raw_data/FLIR")

destination_root="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_root"

# Loop through each session directory in the source root

for source_root in "${source_roots[@]}"; do
    echo "Processing source root: $source_root"

    for session_dir in "$source_root"/*; do
        if [ -d "$session_dir" ]; then
            # Get the name of the session directory
            session_name=$(basename "$session_dir")

            # skip session if it is empty
            if [ ! "$(ls -A $session_dir)" ]; then
                echo "Skipping empty session: $session_name"
                continue
            fi

            # Create the corresponding session directory in the destination root
            destination_session_dir="$destination_root/$session_name"
            mkdir -p "$destination_session_dir"

            # Randomly choose one experiment directory from the session
            experiments=("$session_dir"/*)
            random_experiment="${experiments[RANDOM % ${#experiments[@]}]}"

            # Copy the randomly chosen experiment directory to the destination session directory
            cp "$random_experiment" "$destination_session_dir"
        fi
    done
done