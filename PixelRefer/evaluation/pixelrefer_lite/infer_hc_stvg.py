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
from pixelreferlite import disable_torch_init, model_init, mm_infer
from pycocotools import mask as maskUtils
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
import pycocotools.mask as maskUtils
from torch.utils.data import Dataset, DataLoader
from pixelreferlite.mm_utils import load_video, load_video_from_specific_ids

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

class HC_STVG(Dataset):
    def __init__(self, video_folder, data_list, processor, mode):
        self.video_folder = video_folder
        self.data_list = data_list
        self.processor = processor
        self.mode = mode
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_path = os.path.join(self.video_folder, line['video'][0])
        
        question = "There is one object in the video: <region>.\nPlease give a description of the highlighted object in the video. Using only one sentense."
        # question = "Please give a description of the highlighted object <region> in the video. Output in one sentense."
        # question = 'Please give a detailed description of the highlighted object <region> in the video.'
        frame_ids = []
        mask_ids = []
        masks = []
        mask_ids_raw = []

        for ann in line['masks']:
            for k in ann.keys():
                if int(k) not in frame_ids:
                    frame_ids.append(int(k))
        frame_ids.sort()

        for ann in line['masks']:
            mask_ids_raw_ = []
            for k in ann.keys():
                mask = annToMask(ann[k])
                if mask.sum()>0:
                    mask_ids_raw_.append(frame_ids.index(int(k)))
                    masks.append(mask)
            mask_ids_raw.append(mask_ids_raw_)
            mask_ids+=mask_ids_raw_
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        
        video_tensor = load_video_from_specific_ids(video_path, frame_ids=frame_ids)

        return {
            'video_name': line['video'],
            'video': video_tensor,
            'masks': masks,
            'question': question,
            'mask_ids': mask_ids,
            'answer': line['conversations'][1]['value'],
            'mask_ids_raw': mask_ids_raw
        }

def collate_fn(batch):
    vin = [x['video_name'] for x in batch]
    vid = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    mid = [x['mask_ids'] for x in batch]
    ans = [x['answer'] for x in batch]
    mir = [x['mask_ids_raw'] for x in batch]
    return vin, vid, msk, qs, mid, ans, mir

def build_HC_STVG_eval(args, processor):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = HC_STVG(args.video_folder, questions, processor, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    val_loader = build_HC_STVG_eval(args, processor)
    tps_all = 0
    gen_time_all = 0
    total_samples = 0
    
    final_data = []
    for i, (video_names, video, masks_, questions, mask_ids, answers, mask_ids_raw) in enumerate(tqdm(val_loader)):
        video_name = video_names[0]
        video_tensor = video[0]
        masks = masks_[0]
        question = questions[0]
        mask_ids = mask_ids[0]
        answer = answers[0]
        mask_ids_raw = mask_ids_raw[0]

        output, tps, gen_time = mm_infer(
            video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            masks=masks,
            mask_ids=mask_ids,
            mask_ids_raw=mask_ids_raw,
            return_time=True
        )
        tps_all += tps
        gen_time_all += gen_time
        total_samples += 1
        print(f"Avg tps: {tps_all/total_samples}, Avg gen time: {gen_time_all/total_samples}")
        print(output)
        record = {
            'video': video_name,
            'Answer': answer,
            'pred': output,
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', default='work_dirs/stage1_v0_region')
    parser.add_argument('--video-folder', help='Directory containing video files.', default='benchmarks/videorefer-bench/final/Panda-70M-part')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='benchmarks/hc-stvg-val-allframe.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='./eval/videorefer-bench-d/stage1-output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
