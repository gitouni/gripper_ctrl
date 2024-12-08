import cv2
from models import get_deeplabv3
import argparse
from typing import Tuple
from torchvision import transforms as tf
import numpy as np
import torch
from utils import plot_marker_delta, refresh_dir
import os
import yaml
from robotiq_gripper import RobotiqGripper
import urx

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=2)
    parser.add_argument("--config",type=str, default="cfg/grasp_params.yml")
    parser.add_argument("--model_path",type=str,default="checkpoint/model_40.pth")
    parser.add_argument("--marker_center",type=str,default="ref_marker_center.txt")
    parser.add_argument("--draw_dir",type=str,default="res/realtime5/")
    parser.add_argument("--draw",type=bool,default=True)
    parser.add_argument("--markerless_dir",type=str,default="gripper_res/04/markerless")
    parser.add_argument("--marker_motion_dir",type=str,default="gripper_res/04/marker_motion")
    parser.add_argument("--img_mean",type=float,nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--img_std", type=float,nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--pt_color",type=int,nargs=3,default=[0,0,0])
    parser.add_argument("--arrow_color",type=int,nargs=3,default=[255,246,3])
    parser.add_argument("--scale",type=float,default=9.0)
    parser.add_argument("--threshold",type=float,default=3.25)
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
    config = yaml.load(open(args.config, 'r'), yaml.SafeLoader)
    ip = config['ip']
    port = config['port']
    bottom_pos = config['bottom_pos']
    top_pos = config['top_pos']
    init_gripper = config['init_gripper']
    delta_gripper = config['delta_gripper']
    refresh_dir(args.markerless_dir)
    refresh_dir(args.marker_motion_dir)
    video = cv2.VideoCapture(args.input)
    gripper = RobotiqGripper()
    gripper.connect(ip, port)
    rob = urx.Robot(ip)
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))
    gripper.move(init_gripper, 1, 1)
    rob.movel(bottom_pos, acc=0.3, vel=0.05)
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
    cnt = 1
    refresh_dir(args.draw_dir)
    rob.movel(top_pos, vel=0.008, wait=False)
    flag = False
    curr_pos = rob.getl()[2]
    tgt_pos = top_pos[2]
    while abs(curr_pos - tgt_pos) > 0.001:
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
        marker_motion_data = np.hstack((marker_center, shift))
        cv2.imwrite(os.path.join(args.markerless_dir, "%04d.png"%cnt), frame)
        np.savetxt(os.path.join(args.marker_motion_dir, "%04d.txt"%cnt), marker_motion_data, fmt="%.4f")
        if prev_marker_motion is None:
            prev_marker_motion = shift
        else:
            curr_marker_motion = shift
            delta_marker_motion = cal_norm(prev_marker_motion - curr_marker_motion)
            max_varation = np.max(delta_marker_motion)
            print("max variation:{}".format(max_varation))
            if max_varation > args.threshold:
                if not flag:
                    gripper.grasp_tighter(delta_gripper)
                    print("\033[33;1mSlip detected:\033[0m current ({}) > threshold ({}) case:{}.".format(max_varation, args.threshold,cnt))
                    flag = True
                    max_idx = np.argmax(delta_marker_motion)
                    debug = plot_marker_delta(debug, (marker_center + shift)[[max_idx]], shift[[max_idx]], pt_color=(0,0,0), arrow_color=(0,0,255), scale=args.scale)
                    cv2.imwrite(os.path.join(args.draw_dir, "%04d.png"%cnt), debug)
                print("max_idx:{}".format(max_idx))
        curr_pos = rob.getl()[2]
        cnt += 1
    rob.close()

