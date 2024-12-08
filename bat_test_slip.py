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


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg/te_slip.yml")
    parser.add_argument("--model_path",type=str,default="checkpoint/model_40.pth")
    parser.add_argument("--marker_center",type=str,default="ref_marker_center.txt")
    parser.add_argument("--save",type=str,default="slip_res.yml")
    parser.add_argument("--draw",type=bool,default=False)
    parser.add_argument("--draw_root",type=str,default="res")
    parser.add_argument("--res_root",type=str,default="motion_res")
    parser.add_argument("--img_mean",type=float,nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--img_std", type=float,nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--pt_color",type=int,nargs=3,default=[0,0,0])
    parser.add_argument("--arrow_color",type=int,nargs=3,default=[255,246,3])
    parser.add_argument("--scale",type=float,default=14.0)
    parser.add_argument("--max_threshold1",type=float,default=3.25)
    parser.add_argument("--max_threshold2",type=float,default=2.5)
    parser.add_argument("--mean_threshold",type=float,default=2.0)
    parser.add_argument("--win_size",type=int,default=5)
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
    config = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\033[33;1mWarning: Model running on CPU\033[0m")
    model = get_deeplabv3(2)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model.to(device)
    marker_center = np.loadtxt(args.marker_center).astype(np.int32)
    marker_x, marker_y = marker_center[:,0], marker_center[:, 1]
    res_dict = dict()
    acc = 0
    for subdir, label in config['res'].items():
        video = cv2.VideoCapture(os.path.join(config['root'],subdir,config['stream_fmt']))
        prev_marker_motion = None
        marker_motion_list = []
        cnt = 0
        res_dir = os.path.join(args.res_root, subdir)
        refresh_dir(res_dir)
        if args.draw:
            draw_dir = os.path.join(args.draw_root, subdir)
            refresh_dir(draw_dir)
        flag = 0
        while True:
            ret, frame = video.read()
            if not ret:
                print("Stream Finished")
                break
            x = np2tensor(frame, args.img_mean, args.img_std).to(device)  # flip channel
            out = model(x)['out'].cpu().detach().squeeze(0).numpy()
            shift = out[:,marker_y, marker_x].T
            if args.draw:
                debug = plot_marker_delta(frame, marker_center + shift, shift, pt_color=args.pt_color, arrow_color=args.arrow_color, scale=args.scale)
            marker_motion_data = np.hstack((marker_center, shift))  
            np.savetxt(os.path.join(res_dir, "%04d.png"%cnt), marker_motion_data)  
            if prev_marker_motion is None:
                prev_marker_motion = shift
                if args.draw:
                    cv2.imwrite(os.path.join(draw_dir, "%04d.png"%cnt),debug)
            else:
                curr_marker_motion = shift
                cnt += 1
                delta_marker_motion = cal_norm(prev_marker_motion - curr_marker_motion)
                marker_motion_list.append(delta_marker_motion)
                max_varation = np.max(delta_marker_motion)
                if max_varation > args.max_threshold1:
                    flag = 1
                    max_idx = np.argmax(delta_marker_motion)
                    print("max variation: {} > {}".format(max_varation, args.max_threshold1))
                    if args.draw:
                        debug = plot_marker_delta(debug, (marker_center + shift)[[max_idx]], shift[[max_idx]], pt_color=(0,0,0), arrow_color=(255,0,0), scale=args.scale)
                        debug = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(draw_dir, "%04d.png"%cnt),debug)
                    break
                # if len(marker_motion_list) >= args.win_size and max_varation > args.max_threshold2:
                #     variation = np.concatenate(marker_motion_list[-5:], axis=0)
                #     mean_variation = np.max(np.mean(variation, axis=0))
                #     if mean_variation > args.mean_threshold:
                #         flag = 1
                #         print("mean variation: {} > {}".format(mean_variation, args.mean_threshold))
                #         break
                if args.draw:
                    debug = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(draw_dir, "%04d.png"%cnt),debug)
        res_dict[subdir] = flag
        if flag == label:
            acc += 1
        print("{} : {}".format(subdir, flag))
    print("Acc: {:0.2%}".format(acc / len(config['res'])))
    yaml.dump(res_dict, open(args.save,'w'), yaml.SafeDumper)
        

