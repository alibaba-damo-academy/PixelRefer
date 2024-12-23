import math
import os
import argparse
import json
import warnings
from tqdm import tqdm
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
import sys
sys.path.append('./')
from videorefer import model_init, mm_infer
from pycocotools import mask as maskUtils
import numpy as np
from videorefer.mm_utils import process_video
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
from videorefer.utils import disable_torch_init        
import pycocotools.mask as maskUtils

NUM_FRAMES = 16

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    # num_new_tokens = tokenizer.add_tokens('<region>', special_tokens=True)
    for m in model.modules():
        m.tokenizer = tokenizer

    model = model.to(device='cuda', dtype=torch.float16)

    video_folder = args.video_folder

    data = json.load(open(args.question_file))

    final_data = []
    for d in tqdm(data):
        video_name = d['video']
        d['Question'] = d['Question'].replace('<region>', '[<region>]')

        question = d['Question'] +' ' + ' '.join(d['options']) + '. Answer with the option\'s letter from the given choices directly.'
        annotations = d['annotation']
        
        frame_idx = str(d['frame_idx'])
        annotations_single = []
        for ann in annotations:
            annotations_single.append({frame_idx: ann[frame_idx]})
        annotations = annotations_single
        
        ann_indices = []
        all_frames = set()
        for ann in annotations:
            all_frames.update(list(ann.keys()))
        all_frames = list(all_frames)
        frame_nums = len(all_frames)
        for ann in annotations:
            frame_list = list(ann.keys())
            indices = []
            # print("####frame list")
            # print(frame_list)
            for frame in frame_list:
                indices.append(all_frames.index(frame))
            # print(indices)
            ann_indices.append(indices)

        ann_indices=[ann_indices]
        frame_nums=[frame_nums]
        all_frames = [int(f) for f in all_frames]
        
        print(ann_indices)
        print(frame_nums)
        print(all_frames)

        video_path = os.path.join(video_folder, video_name)

        num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES


        # video_tensor, frame_data, height, width = processor(video_path)
        video_tensor, frame_data, height, width = process_video(video_path, processor=processor, aspect_ratio='square', frame_idx=all_frames)

        masks = []
        for anns in annotations:
            for ann_idx in anns.keys():
                if anns[ann_idx]['segmentation'] is None:
                    mask = np.zeros((height, width))
                else:
                    mask = annToMask(anns[ann_idx]['segmentation'], height, width)
                masks.append(mask)
        masks = np.array(masks)
        masks = torch.Tensor(masks)
        masks = masks.unsqueeze(0).cuda()
        print(masks.shape)

        output = mm_infer(
            video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            masks=masks,
            frame=frame_data,
            ann_indices=ann_indices,
            frame_nums=frame_nums,
        )
        print(output)
        d.pop('annotation')
        d['answer'] = output
        final_data.append(d)
    b = json.dumps(final_data)
    f2 = open(args.output_file, 'w')
    # f2 = open('/mnt/workspace/workgroup/yuanyq/video_data_construction/benchmark/davis/results/mevis_1111_multi.json', 'w')
    f2.write(b)
    f2.close()  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()

    run_inference(args)
