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
from videollama3 import disable_torch_init, model_init, get_model_output
from pycocotools import mask as maskUtils
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
import pycocotools.mask as maskUtils
from torch.utils.data import Dataset, DataLoader
from videollama3.mm_utils import load_video

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

class VideoRefer_Bench_Q(Dataset):
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
        
        line['Question'] = line['Question']
        question = line['Question'] +' ' + ' '.join(line['options']) + '. Answer with the option\'s letter from the given choices directly.'
        video_name = line['video']
        annotations = line['annotation']
        frame_ids = [int(line['frame_idx'])]
        mask_ids = []
        masks = []
      
        for ann in annotations:
          mask_ids.append(0)
          mask = annToMask(ann[line['frame_idx']]['segmentation'])
          masks.append(mask)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
 
        video_tensor = load_video(video_path, fps=1, max_frames=768, frame_ids=frame_ids)

        return {
            'video_name': line['video'],
            'video': video_tensor,
            'masks': masks,
            'question': question,
            'mask_ids': mask_ids,
            'answer': line['Answer'],
            'types': line['type']
        }

def collate_fn(batch):
    vin = [x['video_name'] for x in batch]
    vid = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    mid = [x['mask_ids'] for x in batch]
    ans = [x['answer'] for x in batch]
    tps = [x['types'] for x in batch]
    return vin, vid, msk, qs, mid, ans, tps

def build_videorefer_bench_q_eval(args, processor):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoRefer_Bench_Q(args.video_folder, questions, processor, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    val_loader = build_videorefer_bench_q_eval(args, processor)
    
    final_data = []
    for i, (video_names, video, masks_, questions, mask_ids, answers, types) in enumerate(tqdm(val_loader)):
        video_name = video_names[0]
        video_tensor = video[0]
        masks = masks_[0]
        question = questions[0]
        mask_ids = mask_ids[0]
        answer = answers[0]
        type_ = types[0]
        
        output = get_model_output(
            video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            masks=masks,
            mask_ids=mask_ids
        )
        print(output)
        record = {
            'video': video_name,
            'Answer': answer,
            'pred': output,
            'type': type_,
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', default='DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B')
    parser.add_argument('--video-folder', help='Directory containing video files.', default='video_data')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='videorefer-bench/final/VideoRefer-Bench-Q.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='./eval/videorefer-bench-q/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
