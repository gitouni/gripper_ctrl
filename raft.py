import torch
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, Raft_Small_Weights, raft_small
import torchvision.transforms.functional as F
from torchvision.ops import Conv2dNormActivation
import numpy as np
from PIL import Image
from typing import Literal, Union
from matplotlib import pyplot as plt
import cv2
import os
from tqdm import tqdm

def load_raft_large(in_chan:int, device:str):
	model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
	if in_chan != 3:
		model.feature_encoder.convnormrelu = Conv2dNormActivation(
			in_chan, 64, norm_layer=nn.InstanceNorm2d, kernel_size=7, stride=2, bias=True
		)
		model.context_encoder.convnormrelu = Conv2dNormActivation(
			in_chan, 64, norm_layer=nn.BatchNorm2d, kernel_size=7, stride=2, bias=True
		)
	return model.to(device)

def load_raft_small(in_chan:int, device:str):
	model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
	if in_chan != 3:
		model.feature_encoder.convnormrelu = Conv2dNormActivation(
			in_chan, 32, norm_layer=nn.InstanceNorm2d, kernel_size=7, stride=2, bias=True
		)
		model.context_encoder.convnormrelu = Conv2dNormActivation(
			in_chan, 32, norm_layer=None, kernel_size=7, stride=2, bias=True
		)
	return model.to(device)

class RAFT:
	def __init__(self, in_chan:int=3, model_type:Literal['large','small']='large') -> None:
		weights = Raft_Large_Weights.DEFAULT
		self.transforms = weights.transforms()
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		if model_type == 'large':
			self.model = load_raft_large(in_chan, device=self.device)
		elif model_type == 'small':
			self.model = load_raft_small(in_chan, device=self.device)
		self.model.eval()

	def preprocess(self, img1_batch:torch.Tensor, img2_batch:torch.Tensor):
		if img1_batch.shape[1] == 3:
			img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
			img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
			return self.transforms(img1_batch, img2_batch)
		else:
			return img1_batch, img2_batch

	@torch.inference_mode()
	def predict_flow(self, img1:Union[np.ndarray, Image.Image], img2:Union[np.ndarray, Image.Image], img_shape) -> np.ndarray:
		H, W = img_shape
		src_tensor = F.to_tensor(img1)[None,...]
		tgt_tensor = F.to_tensor(img2)[None,...]
		src_batch, tgt_batch = self.preprocess(src_tensor, tgt_tensor)
		list_of_flows = self.model(src_batch.to(self.device), tgt_batch.to(self.device))
		predicted_flows = F.resize(list_of_flows[-1], (H,W))
		cam_flow = predicted_flows.squeeze(0).cpu()
		return cam_flow  # 2, H, W
	
	def predict_batchflow(self, img1:torch.Tensor, img2:torch.Tensor) -> np.ndarray:
		H, W = img1.shape[-2:]
		src_batch, tgt_batch = self.preprocess(img1, img2)
		list_of_flows = self.model(src_batch.to(self.device), tgt_batch.to(self.device))
		predicted_flows = F.resize(list_of_flows[-1], (H,W))
		cam_flow = predicted_flows.detach().cpu().numpy()
		return cam_flow  # B, 2, H, W
	
if __name__ == "__main__":
	in_chan = 3
	raft = RAFT(in_chan=in_chan, model_type='large')
	# input_dir = "data/slip_v2/ex2/markerless"
	# output_dir = "flow/ex2/markerless"
	# files = sorted(os.listdir(input_dir))
	# for i in tqdm(range(len(files) -1)):
	# 	prev_img = Image.open(os.path.join(input_dir, files[i]))
	# 	curr_img = Image.open(os.path.join(input_dir, files[i+1]))
	# 	flow = raft.predict_flow(prev_img, curr_img, (curr_img.height, curr_img.width))
	# 	flow_img = flow_to_image(flow)
	# 	flow_img:Image.Image = F.to_pil_image(flow_img.squeeze())
	# 	flow_img.save(os.path.join(output_dir, files[i-1]))
	prev_img = Image.open("tmp/Out_0040.png").convert('BGR')
	curr_img = Image.open("tmp/Out_0041.png").convert('BGR')
	flow = raft.predict_flow(prev_img, curr_img, (curr_img.height, curr_img.width)).numpy()
	plt.subplot(2,2,1)
	plt.title('x flow')
	plt.imshow(flow[0,...])
	plt.subplot(2,2,2)
	plt.title('y flow')
	plt.imshow(flow[1,...])
	plt.subplot(2,2,3)
	plt.title('prev img')
	plt.imshow(prev_img)
	plt.subplot(2,2,4)
	plt.title('curr img')
	plt.imshow(curr_img)
	plt.tight_layout()
	plt.savefig("debug_raft.png")