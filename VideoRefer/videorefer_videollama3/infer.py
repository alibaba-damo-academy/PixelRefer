import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os
import torch
import sys
sys.path.append('./')
from videorefer_videollama3 import disable_torch_init, model_init, mm_infer, get_model_output
from videorefer_videollama3.mm_utils import load_video

import numpy as np
from PIL import Image

def infer_image(model, tokenizer):
    image_path = 'demo/images/1.jpg'
    image = Image.open(image_path)
    image_data = np.array(image)

    question = '<image>\nPlease describe the <region> in the image in detail.'

    mask = np.load('demo/masks/demo0.npy')
    masks = []
    masks.append(mask)
    masks = np.array(masks)
    masks = torch.from_numpy(masks).to(torch.uint8)

    mask_ids = [0]*len(masks)

    output = get_model_output(
        [image_data],
        question,
        model=model,
        tokenizer=tokenizer,
        masks=masks,
        mask_ids=mask_ids,
        modal='image',
        image_downsampling=1,
    )
    print(output)

def infer_video(model, tokenizer):
    video_path = 'demo/videos/1.mp4'
    question = '<video>\nPlease describe the <region> in the video in detail.'

    frame_idx = 0 # mask from the first frame
    video_tensor = load_video(video_path, fps=1, max_frames=768, frame_ids=[frame_idx])

    mask = np.load('demo/masks/demo1.npy')
    masks = []
    masks.append(mask)
    masks = np.array(masks)
    masks = torch.from_numpy(masks).to(torch.uint8)

    mask_ids = [0]*len(masks)

    output = get_model_output(
        video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        masks=masks,
        mask_ids=mask_ids,
        modal='video',
    )
    print(output)

def main():
    disable_torch_init()

    # fill in the model path here
    model_path = '/mnt/workspace/workgroup/yuanyq/code/videollama3/ProjectX_region/work_dirs/stage2_v1_10_token'
    model, processor, tokenizer = model_init(model_path)
    
    # image
    infer_image(model, tokenizer)

    # viideo
    infer_video(model, tokenizer)


if __name__=='__main__':
    main()