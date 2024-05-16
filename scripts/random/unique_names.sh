
# Define a function to extract unique directory substrings
extract_unique_dir_substrings() {
    local root_dir=$1
    # Loop through each directory in the current directory
    for dir in "$root_dir"/*; do
        # Extract the second piece of the directory name using '_' as delimiter
        substring=$(echo "$dir" | cut -d'_' -f2)
        
        # Print the substring if it's not empty and unique
        if [ -n "$substring" ]; then
            echo "$substring"
        fi
    #shuffle
    done | sort -u # | wc -l
}

# Call the function to extract and print unique directory substrings
extract_unique_dir_substrings "/scratch-local/users/markoc-haeslerlab/sniff-pretrain-large"