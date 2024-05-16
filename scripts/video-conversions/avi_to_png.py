import cv2
import os
import argparse

def avi_to_pngs(avi_path, dest_dir):
    '''
    Converts an avi video to a directory of png images
    :param avi_path: path to the avi video
    :param dest_dir: directory to save the png images
    '''
    cap = cv2.VideoCapture(avi_path)
    try:
        frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    except AttributeError:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #make dir if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    for i in range(frame_count):
        flag, frame = cap.read()  
        cv2.imwrite(os.path.join(dest_dir, str(i) + '.png'), frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert avi video to png images')
    parser.add_argument('avi_path', type=str, help='Path to the avi video')
    parser.add_argument('dest_dir', type=str, help='Directory to save the png images')
    args = parser.parse_args()

    avi_to_pngs(args.avi_path, args.dest_dir)