import os
import argparse

def get_unique_substrings(directory):
    unique_substrings = set()
    
    # Iterate through all files in the directory (not subdirectories) 
    for filename in os.listdir(directory):
        
        try:
            # Extract mouse name 
            substring = filename.split("_")[1]

            if substring == "":
                print(f"Invalid filename: {filename}, skipping...")
                continue
            
            # Add each substring to the set
            unique_substrings.add(substring)
            
        #invalid name error
        except IndexError:
            print(f"Invalid filename: {filename}, skipping...")
        
    return unique_substrings

if __name__ == "__main__":

    #arg parse for dir
    parser = argparse.ArgumentParser(description='Get unique substrings from filenames in a directory')
    parser.add_argument('directory', default='.', type=str, help='Directory containing files')
    args = parser.parse_args()


    unique_substrings = get_unique_substrings(args.directory)
    for substring in unique_substrings:
        print(substring)

    print(len(unique_substrings))