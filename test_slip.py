import cv2
from models import get_deeplabv3
import argparse
from typing import Union, Tuple
from torchvision import transforms as tf
import numpy as np
import torch
from utils import plot_marker_delta, refresh_dir
import os



def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/slip_v2/ex5/markerless/%04d.png")
    parser.add_argument("--model_path",type=str,default="checkpoint/model_40.pth")
    parser.add_argument("--marker_center",type=str,default="ref_marker_center.txt")
    parser.add_argument("--draw_dir",type=str,default="res/ex5/")
    parser.add_argument("--draw",type=bool,default=True)
    parser.add_argument("--img_mean",type=float,nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--img_std", type=float,nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--pt_color",type=int,nargs=3,default=[0,0,0])
    parser.add_argument("--arrow_color",type=int,nargs=3,default=[255,246,3])
    parser.add_argument("--scale",type=float,default=3.0)
    return parser.parse_args()

def np2tensor(arr:np.ndarray, mean:Tuple[float, float, float], std:Tuple[float,float,float]) -> torch.Tensor:
    transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean=mean, std=std,inplace=True)
    ]
    )
    tensor = transform(arr)
    return torch.unsqueeze(tensor, dim=0).float()  # 1,3,H,W

def cal_norm(force:np.ndarray):
    return np.sqrt(np.sum(force**2, axis=1))

if __name__ == "__main__":
    args = options()
    video = cv2.VideoCapture(args.input)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\033[33;1mWarning: Model running on CPU\033[0m")
    model = get_deeplabv3(2)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model.to(device)
    marker_center = np.loadtxt(args.marker_center).astype(np.int32)
    marker_x, marker_y = marker_center[:,0], marker_center[:, 1]
    prev_marker_motion = None
    cum_marker_motion = np.zeros(marker_center.shape[0])
    cnt = 0
    refresh_dir(args.draw_dir)
    while True:
        ret, frame = video.read()
        if not ret:
            print("Stream Finished")
            break
        x = np2tensor(frame, args.img_mean, args.img_std).to(device)  # flip channel
        out = model(x)['out'].cpu().detach().squeeze(0).numpy()
        shift = out[:,marker_y, marker_x].T
        debug = plot_marker_delta(frame, marker_center + shift, shift, pt_color=args.pt_color, arrow_color=args.arrow_color, scale=args.scale)
        debug = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.draw_dir, "%04d.png"%cnt),debug)
        if prev_marker_motion is None:
            prev_marker_motion = shift
        else:
            curr_marker_motion = shift
            cnt += 1
            delta_marker_motion = cal_norm(prev_marker_motion - curr_marker_motion)
            cum_marker_motion += delta_marker_motion
            max_varation = np.max(delta_marker_motion)
            print("Max variation:{} mean variation: {} pixels in frame {}".format(max_varation, np.max(cum_marker_motion)/cnt, cnt))

