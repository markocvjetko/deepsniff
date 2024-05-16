directories=(
    "211126_RDP037"
    "211127_RDP037"
    "211130_RDP037"
    "211213_RDP039"
    "211215_RDP039"
    "211216_RDP039"
    "220122_RDP041"
    "220124_RDP041"
    "220125_RDP041"
    "220203_RDP042"
    "220204_RDP042"
    "220206_RDP042"
    "220304_RDP044"
    "220306_RDP044"
    "220307_RDP044"
    "220423_RDP052"
    "220425_RDP052"
    "220426_RDP052"
    "221116_RDP084"
    "240128_KK087"
)

# Root directory where the directories are located
root_dir="/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain"

# Iterate over each directory and delete it
for dir in "${directories[@]}"; do
    dir_path="${root_dir}/${dir}"
    if [ -d "$dir_path" ]; then
        echo "Deleting directory: $dir_path"
        rm -r "$dir_path"
    else
        echo "Directory does not exist: $dir_path"
    fi
done