import argparse
from utils.video import VideoWriter

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str,default="video_imgs/grasp2")
    parser.add_argument("--output_video",type=str,default="gripper_res_video/vision_01.mp4")
    parser.add_argument("--fps",type=float,default=30)  # 8.4375 vs 30
    parser.add_argument("--height",type=int,default=480)
    parser.add_argument("--width",type=int,default=448)
    parser.add_argument("--start_index",default=160)
    parser.add_argument("--end_index",default=None)
    parser.add_argument("--repeat_num",type=int,default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    video_writer = VideoWriter(args.fps, args.height, args.width)
    video_writer.frames_to_video(args.input_dir, args.output_video, repeat_num=args.repeat_num)