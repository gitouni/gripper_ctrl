import argparse
from utils.video import ConcatVideoWriter

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir_list",type=str,nargs="+",default=['raw_video_imgs/grasp1','gripper_res/00/markerless','gripper_res/00/marker_motion_pic'])
    parser.add_argument("--output_file",type=str,default="gripper_res_video/concat_1.mp4")
    parser.add_argument("--start_indices", default=[None, None, None])
    parser.add_argument("--end_indices", default=[None, None, None])
    parser.add_argument("--xy_list",default=[[0,0],[540,0],[540,480]])
    parser.add_argument("--wh_list",default=[[540, 960], [640, 480], [640, 480]])
    parser.add_argument("--fps",default=30.0)
    parser.add_argument("--title_list",type=str,default=['Grasping External View','Markerless (raw)','Marker (generated)'])
    parser.add_argument("--total_wh",type=int,default=[1180, 960])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    concat_videowrtier = ConcatVideoWriter(args.img_dir_list, args.start_indices, args.end_indices,
            args.wh_list, args.xy_list, args.title_list, args.total_wh, args.fps)
    concat_videowrtier.frames_to_video(args.output_file)