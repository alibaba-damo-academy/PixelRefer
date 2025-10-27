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
from pycocotools.coco import COCO

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

class DAM(Dataset):
    def __init__(self, image_folder, processor, question_file, num_chunks, chunk_id):
        self.image_folder = image_folder
        self.question_file = question_file
        coco = COCO(os.path.join(question_file))

        def get_mask(ann_id):
            anns = coco.loadAnns([ann_id])
            mask = coco.annToMask(anns[0])

            return mask

        def select_ann(img_id, area_min=None, area_max=None):
            cat_ids = coco.getCatIds()
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)

            if area_min is not None:
                ann_ids = [ann_id for ann_id in ann_ids if coco.anns[ann_id]['area'] >= area_min]

            if area_max is not None:
                ann_ids = [ann_id for ann_id in ann_ids if coco.anns[ann_id]['area'] <= area_max]
            
            return ann_ids
        data_list = []
        img_ids = list(coco.imgs.keys())
        for img_id in img_ids:
            ann_ids = select_ann(img_id)
            img_info = coco.loadImgs(img_id)[0]
            for i, ann_id in enumerate(ann_ids):
                mask = get_mask(ann_id)
                dic = {
                    'id': str(ann_id),
                    'image': img_info['file_name'],
                    'mask': mask,
                }
                data_list.append(dic)

        self.data_list = get_chunk(data_list, num_chunks, chunk_id)
        
        self.processor = processor
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        image_file = os.path.join(self.image_folder, line['image'])
       
        
        # question = 'Please describe <region> in detail.'
        # question = 'Please describe the region <region> in the image in detail.'
        question = 'Please describe the region <region> in the image in detail. Do not describe anything else.'
        image_name = line['image']
        mask = line['mask']
        mask_ids = [0]
        masks = []
        masks.append(mask)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        images = load_images(image_file)
        return {
            'image_name': line['image'],
            'images': images,
            'id': line['id'],
            'masks': masks,
            'question': question,
            'mask_ids': mask_ids,
        }

def collate_fn(batch):
    vin = [x['image_name'] for x in batch]
    vid = [x['images'] for x in batch]
    imgid = [x['id'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    mid = [x['mask_ids'] for x in batch]
    return vin, vid, imgid, msk, qs, mid

def build_DAM_eval(args, processor):
    # convert parquet to json
    dataset = DAM(args.image_folder, processor, args.question_file, args.num_chunks, args.chunk_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    val_loader = build_DAM_eval(args, processor)
    
    final_data = []
    for i, (image_names, images, img_ids, masks_, questions, mask_ids) in enumerate(tqdm(val_loader)):
        image_name = image_names[0]
        image = images[0]
        img_id = img_ids[0]
        masks = masks_[0]
        question = questions[0]
        mask_ids = mask_ids[0]
 
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
            'id': img_id,
            'image': image_name,
            'pred': output,
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', default='work_dirs/stage2_v0_region')
    parser.add_argument('--image-folder', help='Directory containing video files.', default='benchmarks/DLC-Bench/')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='benchmarks/DLC-Bench/annotations.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='./eval/lvis/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    run_inference(args)

    # DAM(1,1,'/mnt/damovl/yuanyq/benchmarks/DLC-Bench/annotations.json',1,0)
