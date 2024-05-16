import os
import numpy
from multiprocessing import Pool


try:
    import cv2
    cv2_avail = True
except ImportError:
    cv2_avail = False


def bytes_to_array(bytestr, coding="le", nbytes=2):
    dtype_str = ">" if coding == "be" else "<"
    dtype_str += f"u{nbytes}"
    return numpy.frombuffer(bytestr, dtype=dtype_str)


class SEQFile:
    MAGIC_BYTES = b"FFF\x00"  # These occur at start of each frame
    WIDTH = 320  # width of each frame
    HEIGHT = 240  # height of each frame
    BYTES_PER_PIXEL = 2  # Number of bytes dedicated to each pixel

    def __init__(self, filename="", coding="le", width=320, height=240):
        self.BYTES = None
        if os.path.isfile(filename):
            self.FILENAME = filename
            if not self._check_file():
                print("Magic bytes do not match, sure this is a seq file?")
        else:
            print(f"File {filename} not found")
        if coding in ("le", "be"):
            self.CODING = coding
        else:
            raise Exception(f"Coding {coding} invalid, choose one of 'le', 'be'")
        self.WIDTH = width
        self.HEIGHT = height

    @classmethod
    def load(cls, filename):
        # shortcut if all defaults are ok
        return cls(filename=filename)

    @property
    def bytes_per_frame(self):
        return self.BYTES_PER_PIXEL * self.HEIGHT * self.WIDTH

    def _check_file(self):
        # Check if file starts with the 'magic bytes'
        return open(self.FILENAME, 'rb').read(len(self.MAGIC_BYTES)) == self.MAGIC_BYTES

    def _load(self):
        # Load file contents into memory
        self.BYTES = open(self.FILENAME, 'rb').read()
        self.MAGIC_BYTES = self.BYTES[:10]   # TODO: decent file type checking
        return self.BYTES

    def split_frames(self):
        if self.BYTES is None:
            self._load()
        return [x for x in self.BYTES.split(self.MAGIC_BYTES) if x]

    def __len__(self):
        return len(self.split_frames())

    def asarray(self):
        # Convert bytes to frames x rows x columns numpy array
        frames = numpy.asarray([bytes_to_array(x[-self.bytes_per_frame:],
                                               nbytes=self.BYTES_PER_PIXEL,
                                               coding=self.CODING) for x in self.split_frames()])
        frames = frames.reshape((-1, self.HEIGHT, self.WIDTH))
        #cut first and last 2 minutes
        frames = frames[120:-120]
        return frames.reshape((-1, self.HEIGHT, self.WIDTH))

    def timestamps(self):
        # Timestamps associated with each frame
        # Indices of s and ms timestamps found manually
        # TODO: check multiple videos to see if we can use regexp pattern match
        second_byte = -1560
        ms_byte = -1556
        headers = [x[:-self.bytes_per_frame] for x in self.split_frames()]
        timestamp = numpy.array([h[second_byte] +
                                 int.from_bytes(h[ms_byte:ms_byte+2], "little") / 1000 for h in headers])
        return timestamp - timestamp[0]

    def _save_avi(self, filename, framerate):
        if not cv2_avail:
            raise ImportError("cv2 not installed")
        img = self.asarray()
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
        img = img.astype("uint8")

        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        writer = cv2.VideoWriter(filename, fourcc,
                                 framerate, img.shape[1:][::-1], False)

        for i in range(img.shape[0]):
            writer.write(img[i, :, :])
        writer.release()

    def save(self, filename, framerate=60):
        if filename.endswith(".avi"):
            os.makedirs(os.path.dirname(filename.split(".avi")[0]), exist_ok=True)
            self._save_avi(filename, framerate)
        else:
            numpy.save(filename, self.asarray())

def process_file(f, destination):
    #get name of file without extension
    dir_name = f.split(".")[0].split("/")[-2]
    file_name = f.split(".")[0].split("/")[-1] + '.avi'
    output_name = os.path.join(destination, dir_name, file_name[:-4], file_name)
    if not os.path.isfile(output_name):
        #print("saving", output_name)
        s = SEQFile(f)
        s.save(output_name, framerate=60)


def main():
    from pathlib import Path
    import argparse
    import glob
    import os
    ap = argparse.ArgumentParser(description="Convert seq files to avi")
    ap.add_argument('directory', nargs='+', type=str, help="Directory containing seq file to be converted to avi")
    ap.add_argument('destination', type=str, help="Directory to save the avi files")
    ap.add_argument('--num_videos', default=10, type=int, help="Number of videos to be sampled from. -1 means all videos are sampled.")
    args = ap.parse_args()
    dirs = args.directory
    if isinstance(dirs, str):
        dirs = [dirs]
    dirs = [Path(d) for d in dirs]
    for d in dirs:
        files = glob.glob(os.path.join(str(d.absolute()), "*.seq"), recursive=True)
        
        #sort files by name
        files.sort()
        #take five files taking uniformly from the start, middle and end
        if args.num_videos > 0:
            files = files[::max(1, len(files)//args.num_videos)]
        
        #print("Converting", len(files), "files")
        
        os.makedirs(args.destination, exist_ok=True)
        with Pool() as p:
                    p.starmap(process_file, [(f, args.destination) for f in files])


if __name__ == "__main__":
    main()
