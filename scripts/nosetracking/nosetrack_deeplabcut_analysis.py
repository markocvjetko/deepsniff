
import os
import argparse
from datetime import datetime
import glob
import json

import tensorflow
from tensorflow.python.client import device_lib

import deeplabcut
from pathlib import Path

raw_data_folder = Path("/mnt/haeslerlab/haeslerlabwip2023/raw_data/nose_track")
parser = argparse.ArgumentParser(description= 'Run deeplabcut')

parser.add_argument('directory', type=str,
                    help='Directory (absolute path) containing avi files from which to extract keypoints')

# args = parser.parse_args()
# input_directory = Path(args.directory)

# #get absolute paths of all subdirs in input_directory, not recusrive
# subdirs = [x for x in input_directory.iterdir() if x.is_dir()]
# #print(subdirs)
# #list avi files in each subdir, absolute paths, and put in single list
# videos = [str(x) for subdir in subdirs for x in subdir.glob('*.avi')]
# #print(*videos, sep = '\n')
# print(len(videos))
#outdir = input_directory / "dlc" #raw_data_folder / input_directory.stem

#if not outdir.exists():
 #   outdir.mkdir()

root_dir = '/scratch-local/users/markoc-haeslerlab/sniff-pretrain-large'

#find all .avi files in root dir

videos = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".avi"):
            videos.append(os.path.join(root, file))
            #print(os.path.join(root, file))

print(videos[:5])

if __name__=='__main__':

    InputYmlFile        = os.getenv("CONFIG_LOCATION")
    #videos              = glob.glob(str((input_directory / "*.avi").absolute()), recursive=True)
   
    shuffle             = 1
        
    AnalyzeVideo        = {
                    "dynamic": [False, 0.5, 12],
                    "save_as_csv": True,
                    "trainingsetindex": 0,
                }
    FilterPredictions   = {"ARdegree" : 6, "MAdegree": 1.8}
    CreateLabeledVideo  = {"CreateLabeledVideo": {"filtered": False}}
    PlotTrajectories    = {"filtered": False}
              
    if not os.path.isfile(InputYmlFile):
        raise IOError('File {0} does not exist. You might want to consider give the full path of your file '
                      .format(InputYmlFile))
        
        
    print("================Starting Analyze Video  =====================================")
        
    dynamic= tuple(AnalyzeVideo.get('dynamic',[False,.5,12]))
    
    deeplabcut.analyze_videos(InputYmlFile,videos,
                              shuffle           = shuffle,
                              videotype         = AnalyzeVideo.get('videotype','.avi'),
                              save_as_csv       = AnalyzeVideo.get('save_as_csv',True),
                              gputouse          = AnalyzeVideo.get('gputouse',None),
                              trainingsetindex  = AnalyzeVideo.get('trainingsetindex',0),
                              destfolder        = AnalyzeVideo.get('destfolder',None),
                              dynamic           = dynamic )
                              
    
    print("================Starting Filter Predictions  =====================================")

    
    #deeplabcut.filterpredictions(InputYmlFile, videos,
     #                            shuffle        = shuffle,
                                 #videotype      = FilterPredictions.get('videotype','.mp4'),
      #                           filtertype     = FilterPredictions.get('filtertype','arima'),
       #                          ARdegree       = FilterPredictions.get('ARdegree',5),
        #                         MAdegree       = FilterPredictions.get('MAdegree',2) )
    
#    deeplabcut.create_labeled_video(InputYmlFile, videos,
    #                                filtered = CreateLabeledVideo.get('filtered',True) )
    

 #   deeplabcut.plot_trajectories(InputYmlFile,videos,
   #                              filtered   = PlotTrajectories.get('filtered',True) )
                                
    
    print("ENDING  at: ",datetime.today().strftime('%d-%m-%Y-%H:%M:%S'))
