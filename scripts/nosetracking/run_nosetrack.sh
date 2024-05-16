#take as argument a root dir, iterate over its subdirectories, and call a Python script with the subdir as input

# Path: scripts/video-conversions/seq_to_avi.sh
#!/bin/bash

# load singularity
module load singularity/3.5

# # Root directory containing subdirectories
root_dir="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain"

# # Path to your Python script
python_script='nosetrack_deeplabcut_analysis.py'

# # Iterate over subdirectories in the root directory
for subdir in "$root_dir"/*; do

    if [ -d "$subdir" ]; then
        echo "Processing subdirectory: $subdir"
        # # Call Python script
        singularity exec --nv --bind /mnt:/mnt \
        /home/michielc-haeslerlab/scripts/flirsniff/singularity/nose_track.sif \
        python3 "$python_script" "$subdir"
    fi
done