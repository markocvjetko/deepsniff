
#!/bin/bash

# # Root directory containing subdirectories
root_dir="/scratch-local/users/markoc-haeslerlab/sniff-pretrain-large"

# # Path to your Python script
python_script='crop_frames.py'

#t_start
t_start=$(date +%s)

# # Iterate over subdirectories in the root directory
for mice_dir in "$root_dir"/*; do
    echo "Processing mice directory: $mice_dir"
    #iterate over subdirs in mice dir
    for subdir in "$mice_dir"/*; do
        if [ -d "$subdir" ]; then
            echo "Processing subdirectory: $subdir"
            # # Call Python script
            python3 "$python_script" "$subdir"
        fi
    done
done

#t_end
t_end=$(date +%s)

# # Calculate the time it took to process all subdirectories
echo "Processing all subdirectories took $(($t_end - $t_start)) seconds"
