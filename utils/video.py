import cv2
from typing import Optional, Tuple, Optional, List, Union
from pathlib import Path
from functools import partial
import os, shutil
import numpy as np
#获得视频的格式

class VideoReader:
    def __init__(self, max_fps:float):
        self.max_fps = max_fps
    def video_to_frames(self, video_path:str, save_dir:Path, resize:bool, size:Optional[Tuple[int, int]]=None):
        save_dir.mkdir(exist_ok=True, parents=True)
        videoCapture = cv2.VideoCapture(video_path)
        img_process_func = partial(cv2.resize, dsize=[size[1],size[0]], interpolation=cv2.INTER_AREA) if resize else lambda x:x
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        t = 0
        success, frame = videoCapture.read()
        if not success:
            videoCapture.release()
            raise RuntimeError("Fail to open Video:{}".format(video_path))
        frame = img_process_func(frame)
        cnt = 0
        cv2.imwrite(str(save_dir.joinpath("%04d.png"%cnt)), frame)
        cnt += 1
        while success :
            success, frame = videoCapture.read() #获取下一帧
            t += 1.0 / fps
            if success and t > 1.0 / self.max_fps:
                cv2.imwrite(str(save_dir.joinpath("%04d.png"%cnt)), img_process_func(frame))
                cnt += 1
                t -= 1.0 / self.max_fps
        videoCapture.release()

class VideoWriter:
    def __init__(self, fps:float, height:int, width:int):
        self.fourcc = cv2.VideoWriter_fourcc(*'mpv4')  # mp4 formatted, but does not fit H264 Encoding
        self.fps = fps
        self.wh = (width, height)
    
    def frames_to_video(self, img_dir:str, output_file:str, start_index:Optional[int]=None, end_index:Optional[int]=None, repeat_num:int=0, **argv):
        files = sorted(os.listdir(img_dir))
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(files)
        files = files[start_index:end_index]
        videowriter = cv2.VideoWriter(output_file, self.fourcc, self.fps, self.wh, isColor=True)
        for img_file in files:
            frame = cv2.imread(os.path.join(img_dir, img_file))
            if frame is None:
                print(f"Warning: {img_file} could not be loaded.")
                continue
            resized_frame = cv2.resize(frame, self.wh)
            videowriter.write(resized_frame)
        for _ in range(repeat_num):  # repeatly add the final frame
            videowriter.write(resized_frame)
        videowriter.release()

class ConcatVideoWriter:
    def __init__(self, img_dir_list:List[str],
         start_indices:List[Union[int, None]],
            end_indices:List[Union[int, None]],
            wh_list:List[Tuple[int,int]],
            xy_list:List[Tuple[int,int]],
            title_list:List[Union[str, None]],
            total_wh:Tuple[int,int],
            fps:float):
        self.img_dir_list = img_dir_list
        img_files_list = []
        for img_dir, start_index, end_index in zip(img_dir_list, start_indices, end_indices):
            img_files = sorted(os.listdir(img_dir))
            if start_index is None:
                start_index = 0
            if end_index is None or end_index < 0:
                end_index = len(img_files)
            img_files = img_files[start_index:end_index]
            img_files_list.append(img_files)
        self.zipped_img_files = list(zip(*img_files_list))
        self.zipped_wh = wh_list
        self.title_list = title_list
        self.xy_list = xy_list
        self.total_wh = total_wh
        self.wh = total_wh
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 formatted, but does not fit H264 Encoding
        self.fps = fps

    def frames_to_video(self, output_file:str):
        videowriter = cv2.VideoWriter(output_file, self.fourcc, self.fps, self.wh, isColor=True)
        merged_img = np.zeros((self.wh[1],self.wh[0],3),dtype=np.uint8)
        for img_files in self.zipped_img_files:
            for img_file, xy, wh, img_dir, title in zip(img_files, self.xy_list, self.zipped_wh, self.img_dir_list, self.title_list):
                img = cv2.imread(os.path.join(img_dir, img_file), cv2.IMREAD_COLOR)
                img = cv2.resize(img, wh)
                if title is not None:
                    img = cv2.putText(img, title, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                merged_img[xy[1]:xy[1]+wh[1], xy[0]:xy[0]+wh[0], :] = img
            videowriter.write(merged_img)
        videowriter.release()

if __name__ == "__main__":
    # debug
    # videoCapture = cv2.VideoCapture('video/camera_up.mp4')
    
    # #获得码率及尺寸
    # fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
    #         int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    
    # #读帧
    # success, frame = videoCapture.read()
    # while success :
    #     cv2.imshow('fps: {}, size: {},{}, nums: {}'.format(fps, size[0], size[1], fNUMS), frame) #显示
    #     cv2.waitKey(1000 // int(fps)) #延迟
    #     success, frame = videoCapture.read() #获取下一帧
    
    # videoCapture.release()
    video_reader = VideoReader(3)
    video_name = "video/camera_down.mp4"
    image_size = (256, 256)
    save_dir = Path("data/down")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)
    video_reader.video_to_frames(video_name, save_dir, True, image_size)