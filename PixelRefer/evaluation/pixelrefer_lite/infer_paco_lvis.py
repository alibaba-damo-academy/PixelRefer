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
from pixelreferlite.mm_utils import load_images

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

class PACO_LVIS(Dataset):
    def __init__(self, image_folder, data_list, processor, mode, question_file):
        self.image_folder = image_folder
        self.question_file = question_file
        self.data_list = []
        for data in data_list:
            image = data['file_name']
            height = data['height']
            width = data['width']
            for i in range(len(data['categories'])):
                cat = data['categories'][i]
                ann = data['annotations'][i]['segmentation']
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
        if os.path.exists(os.path.join(self.image_folder, 'train2017', line['image'])):
            image_file = os.path.join(self.image_folder, 'train2017', line['image'])
        else:
            image_file = os.path.join(self.image_folder, 'val2017', line['image'])
        
        if 'paco' in self.question_file:
            question = "What is the category of <region>? It maybe a subpart of an object. If it is a subpart, output in the format of 'category:subcategory', else just output the category."
        else:
            question = 'What is the category of <region>? Using only one word or phrase.'
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

def build_PACO_LVIS_eval(args, processor):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = PACO_LVIS(args.image_folder, questions, processor, args.mode, args.question_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    val_loader = build_PACO_LVIS_eval(args, processor)
    
    final_data = []
    for i, (image_names, images, masks_, questions, mask_ids, answers, region_nums) in enumerate(tqdm(val_loader)):
        image_name = image_names[0]
        image = images[0]
        masks = masks_[0]
        question = questions[0]
        mask_ids = mask_ids[0]
        answer = answers[0]
        region_num = region_nums[0]

        output = mm_infer(
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

    parser.add_argument('--model-path', help='', default='work_dirs/stage2_v0_region')
    parser.add_argument('--image-folder', help='Directory containing video files.', default='data/coco/')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='benchmarks/lvis_val_1k_category.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='./eval/lvis/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
