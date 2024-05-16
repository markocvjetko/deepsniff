import pandas as pd
import cv2
from pathlib import Path
import argparse
import logging
import concurrent.futures

def process_image(i, min_x, max_x, min_y, max_y, frames_path, output_frames_path):
    img_path = frames_path / f'{i}.png'
    img = cv2.imread(str(img_path))

    img = img[max(0, min_y[i]-10):max_y[i]+20, max(0, min_x[i]-20):max_x[i]+20]

    try:
        cv2.imwrite(str(output_frames_path / f'{i}.png'), img)
    except Exception as e:
        logging.error('Failed to write image %s: %s', i, e)
        logging.error('min_x: %s, max_x: %s, min_y: %s, max_y: %s', min_x[i], max_x[i], min_y[i], max_y[i])
        logging.error('Image shape: %s', img.shape)


def crop_frames(dir):
    dir = Path(dir)

    keypoints_path = next(dir.glob('*.csv'), None)
    if keypoints_path is None:
        logging.error('No csv file found in directory %s', dir)
        return

    frames_path = dir / 'frames'
    output_frames_path = dir / 'cropped_frames'
    output_frames_path.mkdir(exist_ok=True)

    print('frames_path:', frames_path)
    print('output_frames_path:', output_frames_path)

    df = pd.read_csv(keypoints_path, header=[2], index_col=0)

    min_x = df[['x', 'x.1', 'x.2', 'x.3']].min(axis=1).astype(int)
    max_x = df[['x', 'x.1', 'x.2', 'x.3']].max(axis=1).astype(int)
    min_y = df[['y', 'y.1', 'y.2', 'y.3']].min(axis=1).astype(int)
    max_y = df[['y', 'y.1', 'y.2', 'y.3']].max(axis=1).astype(int)

    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_image, range(len(df)), [min_x]*len(df), [max_x]*len(df), [min_y]*len(df), [max_y]*len(df), [frames_path]*len(df), [output_frames_path]*len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Crop frames based on keypoints')
    parser.add_argument('dir', type=str, help='Directory containing the avi files and keypoints')
    args = parser.parse_args()

    crop_frames(args.dir)