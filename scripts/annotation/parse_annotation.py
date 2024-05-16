import xml.etree.ElementTree    as ET
import os 
import numpy as np
from argparse import ArgumentParser
# Parse the XML file
tree = ET.parse('/scratch-local/users/markoc-haeslerlab/to_be_labeled/annotations/task_211130_es010_6.avi_annotations_2024_05_13_16_04_44_cvat for video 1.1/annotations.xml')

# Get the root element
# root = tree.getroot()

# print(root[1][0][2].text)
# #get size
# #   <meta>
# #     <task>
# #       <id>43</id>
# #       <name>211130_ES010_6.avi</name>
# #       <size>422</size>
# video_size = root[0][0][2].text
# print(video_size)

# breathing_onsets = []
# # Iterate over child elements
# for child in root:
#     if child.tag == 'track':
#         if len(child) == 2:
#             breathing_onsets.append(child[0].attrib['frame'])

# print(breathing_onsets)



def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    video_size = root[1][0][2].text
    breathing_onsets = []
    for child in root:
        if child.tag == 'track':
            if len(child) == 2:
                breathing_onsets.append(int(child[0].attrib['frame']))
            else: #len is one
                breathing_onsets.append(int(child[0].attrib['frame']))
    return video_size, breathing_onsets


def main():
    parser = ArgumentParser()
    parser.add_argument('--xml_folder', type=str, required=True, help='Path to the folder containing the XML files')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder where the output files will be saved')
    args = parser.parse_args()


    #list all subfolders in xml_folder
    trail_folders = [f.path for f in os.scandir(args.xml_folder) if f.is_dir()]
    
    for trail_folder in trail_folders:
        xml_file = os.path.join(trail_folder, 'annotations.xml')
        print('Parsing annotation file: {}'.format(xml_file))
        
        #parses the trail name from auto-generated folder name
        trail_name = '_'.join(os.path.basename(trail_folder).split('_')[1:4])[:-4].upper()
        video_size, breathing_onsets = parse_annotation(xml_file)
        output_file = os.path.join(args.output_folder, trail_name, 'breathing_onsets.txt')  
        #output_file uppercase
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(breathing_onsets)
        #transform array of timestamps (2, 3, 15, 27...) to array where each frame has 0 or 1, depending on the timestamp
        breathing_onsets = np.array([1 if i in breathing_onsets else 0 for i in range(int(video_size))])
        np.savetxt(output_file, breathing_onsets, fmt='%s')

if __name__ == '__main__':
    main()