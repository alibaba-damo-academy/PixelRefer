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
from torch.utils.data import Dataset, DataLoader

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

class VideoRefer_Bench_D(Dataset):
    def __init__(self, video_folder, data_list, processor, mode):
        self.video_folder = video_folder
        self.data_list = data_list
        self.processor = processor
        self.mode = mode
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_path = os.path.join(self.video_folder, line['video'])
    
        question = 'Please give a detailed description of the highlighted object [<region>] in the video.'
        video_name = line['video']
        annotations = line['annotation']
        
        if self.mode=='single':
            frame_idx = str(line['frame_idx'])
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
            for frame in frame_list:
                indices.append(all_frames.index(frame))
            ann_indices.append(indices)

        ann_indices=[ann_indices]
        frame_nums=[frame_nums]
        all_frames = [int(f) for f in all_frames]

        video_path = os.path.join(self.video_folder, video_name)

        video_tensor, frame_data, height, width = process_video(video_path, processor=self.processor, aspect_ratio='square', frame_idx=all_frames)

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
        masks = masks.unsqueeze(0)
        
        return {
            'video_name': line['video'],
            'video': video_tensor,
            'masks': masks,
            'question': question,
            'frame': frame_data,
            'ann_indices': ann_indices,
            'frame_nums': frame_nums,
            'caption': line['caption'],
        }

def collate_fn(batch):
    vin = [x['video_name'] for x in batch]
    vid = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    fr = [x['frame'] for x in batch]
    aid = [x['ann_indices'] for x in batch]
    fn = [x['frame_nums'] for x in batch]
    cap = [x['caption'] for x in batch]
    return vin, vid, msk, qs, fr, aid, fn, cap

def build_videorefer_bench_d_eval(args, processor):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoRefer_Bench_D(args.video_folder, questions, processor, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    for m in model.modules():
        m.tokenizer = tokenizer

    model = model.to(device='cuda', dtype=torch.float16)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    val_loader = build_videorefer_bench_d_eval(args, processor)
    
    final_data = []
    for i, (video_names, video, masks_, questions, frames, ann_ids, framenums, captions) in enumerate(tqdm(val_loader)):
        video_name = video_names[0]
        video_tensor = video[0]
        masks = masks_[0]
        question = questions[0]
        frame_data = frames[0]
        ann_indices = ann_ids[0]
        frame_nums = framenums[0]
        caption = captions[0]
        
        output = mm_infer(
            video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            masks=masks.cuda(),
            frame=frame_data,
            ann_indices=ann_indices,
            frame_nums=frame_nums,
        )
        record = {
            'video': video_name,
            'caption': caption,
            'pred': output,
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
