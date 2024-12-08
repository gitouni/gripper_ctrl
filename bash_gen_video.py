import argparse
from utils.video import VideoWriter
import yaml
from typing import Dict

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="cfg/video.yml")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    config:Dict = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    for name, argv in config.items():
        for key in ['markerless', 'ann','vision']:
            print("writing {} {}".format(name, key))
            subargv = argv[key]
            video_writer = VideoWriter(subargv['fps'], subargv['height'], subargv['width'])
            video_writer.frames_to_video(**subargv)