import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Iterable, Optional

class FeatureExtractor:
    def __init__(self, num_features:int=2048, pad_if_not_divisible:bool=True):
        self.device = K.utils.get_cuda_or_mps_device_if_available()
        self.model = KF.DISK.from_pretrained("depth").to(self.device)
        self.num_features = num_features
        self.pad = pad_if_not_divisible

    def load_img(self, img_path:str):
        return K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=self.device)[None, ...]
    
    @torch.inference_mode()
    def extract_features(self, path:str):
        img = self.load_img(path)
        feat = self.model(img, self.num_features, pad_if_not_divisible=self.pad)[0]
        kps, descs = feat.keypoints, feat.descriptors
        lafs = KF.laf_from_center_scale_ori(kps[None], torch.ones(1, len(kps), 1, 1, device=self.device))
        return kps, descs, lafs, img.shape[-2:]
    
    @torch.inference_mode()
    def extract_list_features(self, path_list:Iterable[str]):
        img_list = [self.load_img(path) for path in path_list]
        img = torch.cat(img_list, dim=0)
        feat_list = self.model(img, self.num_features, self.pad)
        kpt_list = []
        desc_list = []
        laf_list = []
        for feat in feat_list:
            kps, descs = feat.keypoints, feat.descriptors
            lafs = KF.laf_from_center_scale_ori(kps[None], torch.ones(1, len(kps), 1, 1, device=self.device))
            kpt_list.append(kps)
            desc_list.append(descs)
            laf_list.append(lafs)
        return kpt_list, desc_list, laf_list, img.shape[-2:]
    
class FeatureMatcher:
    def __init__(self):
        self.device = K.utils.get_cuda_or_mps_device_if_available()
        self.model = KF.LightGlueMatcher("disk").eval().to(self.device)

    @torch.inference_mode()
    def match(self, params:Dict[str, torch.Tensor], mask:Optional[torch.Tensor]=None):
        if mask is None:
            dists, idxs = self.model(params['desc1'], params['desc2'], params['lafs1'], params['lafs2'], params['hw1'], params['hw2'])
            return dists, idxs
        else:
            assert 'kpt1' in params, "params must include 'kpt1' to extract mask"
            kpt1 = params['kpt1'].to(torch.long)
            rev = mask[kpt1[:,1], kpt1[:,0]]
            dists, idxs = self.model(params['desc1'][rev], params['desc2'][rev], params['lafs1'][rev], params['lafs2'][rev], params['hw1'], params['hw2'])
            return dists, idxs, rev