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
from videollama3.mm_utils import load_images

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

class REFCOCOG(Dataset):
    def __init__(self, image_folder, data_list, processor, mode, question_file):
        self.image_folder = image_folder
        self.question_file = question_file
        self.data_list = []
        for data in data_list:
            image = data['image'].split('/')[-1].split('_')[-1]
            height = data['height']
            width = data['width']
            for i in range(len(data['cat'])):
                cat = data['cat'][i]
                ann = data['masks'][i]
                try:
                    region_num = data['region_num'][i]
                except:
                    region_num = 0
                dic = {
                    'image': image,
                    'height': height,
                    'width': width,
                    'category': cat,
                    'mask': ann,
                    'region_num': region_num
                }
                self.data_list.append(dic)
            
        self.processor = processor
        self.mode = mode
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        if os.path.exists(os.path.join(self.image_folder, line['image'])):
            image_file = os.path.join(self.image_folder, line['image'])
        else:
            image_file = os.path.join('/mnt/damovl/MEDIA/IMAGE/COCO/val2017', line['image'])
        
        question = '<image>\nCan you provide a brief description of the region marked by <region> in the picture?'
        image_name = line['image']
        mask = line['mask']
        mask_ids = [0]
        masks = []
        mask = annToMask(mask, line['height'], line['width'])
        masks.append(mask)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        images = load_images(image_file)
        return {
            'image_name': line['image'],
            'images': images,
            'masks': masks,
            'question': question,
            'mask_ids': mask_ids,
            'answer': line['category'],
            'region_num': line['region_num']
        }

def collate_fn(batch):
    vin = [x['image_name'] for x in batch]
    vid = [x['images'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    mid = [x['mask_ids'] for x in batch]
    ans = [x['answer'] for x in batch]
    rn = [x['region_num'] for x in batch]
    return vin, vid, msk, qs, mid, ans, rn

def build_REFCOCOG_eval(args, processor):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = REFCOCOG(args.image_folder, questions, processor, args.mode, args.question_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    val_loader = build_REFCOCOG_eval(args, processor)
    
    final_data = []
    for i, (image_names, images, masks_, questions, mask_ids, answers, region_nums) in enumerate(tqdm(val_loader)):
        image_name = image_names[0]
        image = images[0]
        masks = masks_[0]
        question = questions[0]
        mask_ids = mask_ids[0]
        answer = answers[0]
        region_num = region_nums[0]

        output = get_model_output(
            image,
            question,
            model=model,
            tokenizer=tokenizer,
            masks=masks,
            mask_ids=mask_ids,
            modal='image',
            image_downsampling=1,
        )
        print(output)
        record = {
            'image': image_name,
            'Answer': answer,
            'pred': output,
            'region_num': region_num,
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', default='DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B')
    parser.add_argument('--image-folder', help='Directory containing video files.', default='COCO/train2017')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='refcocog_val.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='./eval/refcocog/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
