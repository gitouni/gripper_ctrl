from utils.video import VideoReader
from pathlib import Path
# import os
import argparse
import shutil
# from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--video_file",type=str,default="video/grasp4.mp4")
parser.add_argument("--output_dir",type=str,default="video_imgs/grasp4")
parser.add_argument("--max_fps",type=float,default=30)
parser.add_argument("--if_resize",action="store_true", default=True)
parser.add_argument("--resize",type=int, nargs=2, default=[480, 270])
args = parser.parse_args()
output_dir = Path(args.output_dir)
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)
reader = VideoReader(args.max_fps)
reader.video_to_frames(args.video_file, output_dir, args.if_resize, args.resize)