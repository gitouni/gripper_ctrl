from utils.video import VideoReader
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir",type=str,default="video")
parser.add_argument("--output_dir",type=str,default="raw_video_imgs")
parser.add_argument("--max_fps",type=float,default=30)
parser.add_argument("--if_resize",action="store_true")
parser.add_argument("--resize",type=int, nargs=2, default=[480,270])
args = parser.parse_args()
video_dir = Path(args.video_dir)
video_files = sorted(os.listdir(video_dir))
output_dir = Path(args.output_dir)
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)
for video_file in tqdm(video_files, 'video to frames'):
    video_path = video_dir.joinpath(video_file)
    reader = VideoReader(args.max_fps)
    reader.video_to_frames(str(video_path), output_dir.joinpath(os.path.splitext(video_file)[0]), args.if_resize, list(reversed(args.resize)))