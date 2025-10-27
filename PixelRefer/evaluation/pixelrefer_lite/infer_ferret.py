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
from videollama3 import disable_torch_init, model_init, mm_infer
from pycocotools import mask as maskUtils
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
import pycocotools.mask as maskUtils
from torch.utils.data import Dataset, DataLoader
from videollama3.mm_utils import load_images
import cv2
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

class Ferret(Dataset):
    def __init__(self, image_folder, data_list, processor, mode, question_file):
        self.image_folder = image_folder
        self.question_file = question_file
        self.data_list = []
        idx = 0
        for data in data_list:
            image = data['image']
            if data['category']=='reason':
                question = 'There is 1 region in the image: <region> ' + data['text'].replace('<region>', '')
            else:
                question = data['text']
            ann = data['annotation']['segmentation']
            dic = {
                'question_id': data['question_id'],
                'image': image,
                'question': question,
                'mask': ann,
                'category': data['category']
            }
            idx+=1
            self.data_list.append(dic)
            
        self.processor = processor
        self.mode = mode
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        if os.path.exists(os.path.join(self.image_folder, 'train2017', line['image'])):
            image_file = os.path.join(self.image_folder, line['image'])
        else:
            image_file = os.path.join(self.image_folder, 'val2017', line['image'])
        
        question_id = line['question_id']
        question = line['question']
        image_name = line['image']
        mask = line['mask']
        mask_ids = [0]
        masks = []
        image = cv2.imread(image_file)
        height, width = image.shape[:2]
        mask = annToMask(mask, height, width)
        masks.append(mask)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        images = load_images(image_file)
        return {
            'question_id': question_id,
            'image_name': line['image'],
            'images': images,
            'masks': masks,
            'question': question,
            'mask_ids': mask_ids,
            'category': line['category']
        }

def collate_fn(batch):
    qid = [x['question_id'] for x in batch]
    vin = [x['image_name'] for x in batch]
    vid = [x['images'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    mid = [x['mask_ids'] for x in batch]
    cat = [x['category'] for x in batch]
    return qid, vin, vid, msk, qs, mid, cat

def build_Ferret_eval(args, processor):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = Ferret(args.image_folder, questions, processor, args.mode, args.question_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    val_loader = build_Ferret_eval(args, processor)
    
    final_data = []
    for i, (question_ids, image_names, images, masks_, questions, mask_ids, cats) in enumerate(tqdm(val_loader)):
        question_id = question_ids[0]
        image_name = image_names[0]
        image = images[0]
        masks = masks_[0]
        question = questions[0]
        mask_ids = mask_ids[0]
        cat = cats[0]

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
            'question_id': question_id,
            'image': image_name,
            'pred': output,
            'category': cat
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', default='work_dirs/stage2_v0_region')
    parser.add_argument('--image-folder', help='Directory containing video files.', default='data/coco/')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='ferret-bench/box_refer_caption.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='./eval/osprey_desc/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
