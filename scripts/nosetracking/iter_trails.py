import os

root_dir = '/scratch-local/users/markoc-haeslerlab/sniff-pretrain-large'

#find all .avi files in root dir

avi_files = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".avi"):
            avi_files.append(os.path.join(root, file))
            print(os.path.join(root, file))

