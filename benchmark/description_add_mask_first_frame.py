import json
import torch
import pycocotools.mask as maskUtils
import numpy as np
from decord import VideoReader, cpu
import bisect
import os
import cv2
from tqdm import tqdm
import shutil
from matplotlib import pyplot as plt
import re
from PIL import Image
from visualizer import Visualizer
import argparse

def singleMask2rle(mask):
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def frame_sample(duration, video_frames=16):
    num_frames = 32
    seg_size = float(duration - 1) / num_frames

    raw_frame_ids = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        raw_frame_ids.append((start + end) // 2)

    sampled_frame_ids = []
    seg_size = float(num_frames - 1) / video_frames
    for i in range(video_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        sampled_frame_ids.append(raw_frame_ids[(start + end) // 2])
    return sampled_frame_ids
       
def annToMask(rle):
    m = maskUtils.decode(rle)
    return m


parser = argparse.ArgumentParser(description='Data construction', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--video_frames', default=16) 
parser.add_argument('--json_path', required=True)
parser.add_argument('--video_path', required=True)
parser.add_argument('--save_dir', required=True)
args = parser.parse_args()  

benchmark_qa = json.load(open(args.json_path))

for ii,data in tqdm(enumerate(benchmark_qa)):
    video_path = os.path.join(args.video_path, data['video'])
    vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames_of_video = len(vreader)
    frame_id_list = frame_sample(num_frames_of_video, video_frames=16)
    
    video_data = [Image.fromarray(frame) for frame in vreader.get_batch(frame_id_list).asnumpy()]
    is_masked = False
    save_dir = os.path.join(args.save_dir, str(ii))
    os.makedirs(save_dir, exist_ok=True)

    for idx, i in enumerate(frame_id_list):
        image = video_data[idx]
        if is_masked:
            image.save(os.path.join(save_dir,str(i).zfill(5)+'.jpg'))
            continue
        visualizer = Visualizer(image)
        
        ann = data['annotation'][0]
        if str(i) in ann and ann[str(i)]['segmentation'] is not None:
            mask = annToMask(ann[str(i)]['segmentation'])
            is_masked = True
            masked_img = visualizer.draw_binary_mask_with_number(mask, edge_color='red', color='red', alpha=0.1, anno_mode=['Mask', 'Mark'])
            masked_img.save(os.path.join(save_dir,str(i).zfill(5)+'.jpg'))
        else:
            image.save(os.path.join(save_dir,str(i).zfill(5)+'.jpg'))
            
