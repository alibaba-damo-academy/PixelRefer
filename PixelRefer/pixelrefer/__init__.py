import os
import copy
import math
import warnings
import shutil
from functools import partial

import torch
import torch.nn.functional as F

from .model import load_pretrained_model
from .model.processor import PixelreferProcessor
from .mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria, resize_image_mask, load_video_from_ids
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, STREAM_START_TOKEN, STREAM_END_TOKEN
from pixelrefer.constants import REGION_TOKEN
import time
from transformers import TextIteratorStreamer
from threading import Thread

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    

def model_init(model_path=None, **kwargs):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    aspect_ratio = model.config.image_aspect_ratio if hasattr(model.config, "image_aspect_ratio") else "pad"
    image_size = model.config.image_size if hasattr(model.config, "image_size") else 384
    # NOTE: If num_frames is None, the frame sampling mode is "fps". If num_frames is not None, the frame sampling mode is "uniform". 
    # num_frames = model.config.num_frames
    model.config.region_token_index = tokenizer.convert_tokens_to_ids(REGION_TOKEN)
    processor = {
        'image': load_images,
        'video': load_video_from_ids,
        'text':  None
    }

    return model, processor, tokenizer


def mm_infer(images_or_videos, instruct, model, tokenizer, modal='video', **kwargs):
    """inference api of PixelRefer for video understanding.

    Args:
        model: PixelRefer model.
        images_or_videos (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    mask_ids = kwargs.pop('mask_ids', None)
    masks = kwargs.pop('masks', None)
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
        images = images_or_videos
        additional_frames = images.copy()
        timestamps = None
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
        images, timestamps, additional_frames = images_or_videos
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    vlprocessor = PixelreferProcessor(model.get_vision_encoder().image_processor, tokenizer)
    vlprocessor.tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN], special_tokens=True)

    model.config.image_token_index = vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

    if masks is not None:
        additional_frames, masks, mask_nums, box_params = resize_image_mask(additional_frames, masks, mask_ids, max_tokens=model.config.mask_num)
        for idx in range(len(mask_nums)):
            instruct = instruct.replace('<region>', "["+REGION_TOKEN*mask_nums[idx]+"]", 1)

        additional_images_dict = vlprocessor._process_image(additional_frames, image_downsampling=1) 
        additional_images = additional_images_dict['images']
        additional_images_thws = additional_images_dict['grid_thws']
        additional_images = (additional_images, additional_images_thws)


    else:
        additional_images = []
        mask_nums = []
        box_params = None
    
    # 1. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        messages = [{'role': 'user', 'content': instruct}]
    elif isinstance(instruct, list):
        messages = copy.deepcopy(instruct)
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if all(not modal_token in message["content"] for message in messages):
        warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
        messages[0]["content"] = modal_token + messages[0]["content"]

    converted_messages = []
    for message in messages:
        chunks = message["content"].split(modal_token)
        converted_messages.append({
            "role": "user",
            "content": []
        })

        for chunk_idx in range(1, 2 * len(chunks)):
            if chunk_idx % 2 == 1:
                chunk = chunks[chunk_idx // 2].strip()
                converted_messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
            else:
                if modal == 'image':
                    converted_messages[-1]["content"].append({"type": "image"})
                elif modal == 'video':
                    converted_messages[-1]["content"].append({"type": "video", "num_frames": len(images), "time": timestamps})

    messages = converted_messages

    # 2. vision preprocess (load & transform image or video).
    if model.config.model_type in ['pixelrefer_mistral', 'pixelrefer_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    image_downsampling = kwargs.get('image_downsampling', model.config.spatial_merge_size)
    # TODO: attention mask?

    messages = system_message + messages
    data_dict = vlprocessor(
        images=images,
        text=messages,
        image_downsampling=image_downsampling,
        return_tensors="pt",
    )

    torch_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.float16

    images = [x.to(torch_dtype).cuda(non_blocking=True) for x in data_dict["images"]]
    grid_thws = [x.cuda(non_blocking=True) for x in data_dict["grid_thws"]]

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, data_dict["input_ids"])

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            # input_ids,
            # attention_mask=attention_masks,
            # images=images,
            data_dict["input_ids"].cuda(),
            attention_mask=data_dict["attention_mask"].cuda(),
            images=[(modal, images, grid_thws)],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            additional_images=[additional_images],
            masks=[masks],
            box_params=[box_params]
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs



def get_model_output_streaming(images_or_videos, instruct, model, tokenizer, modal='video', **kwargs):
    """inference api of PixelRefer for video understanding.

    Args:
        model: PixelRefer model.
        images_or_videos (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    mask_ids = kwargs.pop('mask_ids', None)
    masks = kwargs.pop('masks', None)
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
        images = images_or_videos
        additional_frames = images.copy()
        timestamps = None
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
        images, timestamps, additional_frames = images_or_videos
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    vlprocessor = PixelreferProcessor(model.get_vision_encoder().image_processor, tokenizer)
    vlprocessor.tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN], special_tokens=True)

    model.config.image_token_index = vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

    if masks is not None:
        additional_frames, masks, mask_nums, box_params = resize_image_mask(additional_frames, masks, mask_ids, max_tokens=model.config.mask_num)
        for idx in range(len(mask_nums)):
            instruct = instruct.replace('<region>', "["+REGION_TOKEN*mask_nums[idx]+"]", 1)

        additional_images_dict = vlprocessor._process_image(additional_frames, image_downsampling=1) 
        additional_images = additional_images_dict['images']
        additional_images_thws = additional_images_dict['grid_thws']
        additional_images = (additional_images, additional_images_thws)


    else:
        additional_images = []
        mask_nums = []
        box_params = None
    
    # 1. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        messages = [{'role': 'user', 'content': instruct}]
    elif isinstance(instruct, list):
        messages = copy.deepcopy(instruct)
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if all(not modal_token in message["content"] for message in messages):
        warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
        messages[0]["content"] = modal_token + messages[0]["content"]

    converted_messages = []
    for message in messages:
        chunks = message["content"].split(modal_token)
        converted_messages.append({
            "role": "user",
            "content": []
        })

        for chunk_idx in range(1, 2 * len(chunks)):
            if chunk_idx % 2 == 1:
                chunk = chunks[chunk_idx // 2].strip()
                converted_messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
            else:
                if modal == 'image':
                    converted_messages[-1]["content"].append({"type": "image"})
                elif modal == 'video':
                    converted_messages[-1]["content"].append({"type": "video", "num_frames": len(images), "time": timestamps})

    messages = converted_messages

    # 2. vision preprocess (load & transform image or video).
    if model.config.model_type in ['pixelrefer_mistral', 'pixelrefer_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    image_downsampling = kwargs.get('image_downsampling', model.config.spatial_merge_size)
    # TODO: attention mask?

    messages = system_message + messages
    data_dict = vlprocessor(
        images=images,
        text=messages,
        image_downsampling=image_downsampling,
        return_tensors="pt",
    )

    torch_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.float16

    images = [x.to(torch_dtype).cuda(non_blocking=True) for x in data_dict["images"]]
    grid_thws = [x.cuda(non_blocking=True) for x in data_dict["grid_thws"]]

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, data_dict["input_ids"])

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)
    stop_str = tokenizer.eos_token
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) 
    generation_kwargs = dict(
        inputs=data_dict["input_ids"].cuda(),
        attention_mask=data_dict["attention_mask"].cuda(),
        images=[(modal, images, grid_thws)],
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        pad_token_id=tokenizer.eos_token_id,
        additional_images=[additional_images],
        masks=[masks],
        box_params=[box_params],
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        if stop_str in generated_text:
            generated_text = generated_text[:generated_text.find(stop_str)]
            break
        yield new_text
    
    thread.join()
   