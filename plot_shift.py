import numpy as np
import argparse
from utils import plot_marker_delta, refresh_dir
import os
from PIL import Image
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",type=str,default="gripper_res/05/markerless")
    parser.add_argument("--marker_shift_dir",type=str,default="gripper_res/05/marker_motion")
    parser.add_argument("--output_dir",type=str,default="gripper_res/05/marker_motion_pic")
    parser.add_argument("--pt_color",type=int, nargs=3,default=[0,0,0])
    parser.add_argument("--arrow_color",type=int, nargs=3,default=[255,246,3])
    parser.add_argument("--arrow_scale",type=float,default=14)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    markerless_files = sorted(os.listdir(args.img_dir))
    markershift_files = sorted(os.listdir(args.marker_shift_dir))
    refresh_dir(args.output_dir)
    for img_file, shift_file in tqdm(zip(markerless_files, markershift_files),total=len(markerless_files)):
        img = Image.open(os.path.join(args.img_dir, img_file))
        kpt_shift = np.loadtxt(os.path.join(args.marker_shift_dir, shift_file))
        kpt_coords = kpt_shift[:, :2].astype(np.int32)
        end_coords = (kpt_shift[:, :2] + kpt_shift[:, 2:]).astype(np.int32)
        Image.fromarray(plot_marker_delta(img, kpt_coords, kpt_shift[:,2:], args.pt_color, args.arrow_color, scale=args.arrow_scale)).save(os.path.join(args.output_dir, img_file))