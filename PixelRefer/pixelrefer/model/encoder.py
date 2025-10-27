import os

import torch
import torch.nn as nn
from transformers import (CLIPImageProcessor, CLIPVisionConfig,
                          CLIPVisionModel, SiglipImageProcessor,
                          SiglipVisionConfig, SiglipVisionModel)

from .qwen2vl_encoder import (Qwen2VisionTransformerPretrainedModel,
                              Qwen2VLImageProcessor, Qwen2VLVisionConfig)

from .damovl_encoder  import (DAMOVLImageProcessor, DAMOVLVisionModel)


class CLIPVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = CLIPVisionModel.from_pretrained(self.vision_encoder_name,
                                                            attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = SiglipVisionModel.from_pretrained(self.vision_encoder_name,
                                                              attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class Qwen2VLVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model(args)
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = Qwen2VLVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self, args):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        # merge_size is set to 1 by default, because STAGE1, STAGE1.5, STAGE2 are trained with merge_size=1
        # for stage 3, the merge_size is set to 2 by argments. 
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.vision_encoder_name)
        self.image_processor.merge_size = args.spatial_merge_size
        # NOTE: The maximum number of vision tokens is 8192 by default.
        mm_max_length = args.mm_max_length if hasattr(args, 'mm_max_length') else 9477 // (args.spatial_merge_size**2)
        self.image_processor.max_pixels = mm_max_length * (args.spatial_merge_size**2 * self.image_processor.patch_size**2)
        self.image_processor.size["max_pixels"] = self.image_processor.max_pixels

        # merge_size is fixed to 1 for STAGE1, STAGE1.5, STAGE2, STAGE3 in encoder and can be modified in connector.
        self.cfg_only = Qwen2VLVisionConfig.from_pretrained(self.vision_encoder_name)
        self.cfg_only.spatial_merge_size = args.spatial_merge_size

        self.vision_encoder = Qwen2VisionTransformerPretrainedModel.from_pretrained(
            self.vision_encoder_name,
            config=self.cfg_only,
            torch_dtype=args.torch_dtype,
            attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def forward(self, images, grid_thws, strides, **kwargs):
        images    = [image    for sub_images in images for image in sub_images]
        grid_thws = [grid_thw for sub_grid_thws in grid_thws for grid_thw in sub_grid_thws]
        strides = [stride for sub_strides in strides for stride in sub_strides]

        images = torch.cat(images, dim=0)
        grid_thws = torch.cat(grid_thws, dim=0)

        image_features = self.vision_encoder(images, grid_thws, strides=strides)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return -1

    @property
    def num_patches_per_side(self):
        return -1

    @property
    def image_size(self):
        return 14 * self.vision_encoder.config.spatial_merge_size


class DAMOVLVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.args = args

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model(self.args)
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = DAMOVLVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self, args):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        # merge_size is set to 1 by default, because STAGE1, STAGE1.5, STAGE2 are trained with merge_size=1
        # for stage 3, the merge_size is set to 2 by argments. 
        self.image_processor = DAMOVLImageProcessor.from_pretrained(self.vision_encoder_name)
        self.image_processor.merge_size = args.spatial_merge_size
        # NOTE: The maximum number of vision tokens is 8192 by default.
        mm_max_length = args.mm_max_length if hasattr(args, 'mm_max_length') else 9477 // (args.spatial_merge_size**2)
        self.image_processor.max_pixels = mm_max_length * (args.spatial_merge_size**2 * self.image_processor.patch_size**2)
        self.image_processor.size["max_pixels"] = self.image_processor.max_pixels

        # merge_size is fixed to 1 for STAGE1, STAGE1.5, STAGE2, STAGE3 in encoder and can be modified in connector.
        self.cfg_only = Qwen2VLVisionConfig.from_pretrained(self.vision_encoder_name)
        self.cfg_only.spatial_merge_size = args.spatial_merge_size

        self.vision_encoder = DAMOVLVisionModel.from_pretrained(
            self.vision_encoder_name,
            spatial_merge_size=args.spatial_merge_size,
            torch_dtype=args.torch_dtype,
            attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def forward(self, images, grid_thws, strides, **kwargs):
        images    = [image    for sub_images in images for image in sub_images]
        grid_thws = [grid_thw for sub_grid_thws in grid_thws for grid_thw in sub_grid_thws]
        strides = [stride for sub_strides in strides for stride in sub_strides]

        images = torch.cat(images, dim=0)
        grid_thws = torch.cat(grid_thws, dim=0)

        image_features = self.vision_encoder(images, grid_thws, strides)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return -1

    @property
    def num_patches_per_side(self):
        return -1

    @property
    def image_size(self):
        return 14 * self.vision_encoder.config.spatial_merge_size


def build_vision_encoder(vision_encoder_cfg, **kwargs):
    vision_encoder = getattr(vision_encoder_cfg, 'mm_vision_encoder', getattr(vision_encoder_cfg, 'vision_encoder', None))

    if  'clip' in vision_encoder:
        vision_encoder = CLIPVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    elif 'siglip' in vision_encoder:
        vision_encoder = SiglipVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    elif 'qwen2vl' in vision_encoder:
        vision_encoder = Qwen2VLVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    elif 'damovl' in vision_encoder or 'SigLIP-NaViT' in vision_encoder:
        vision_encoder = DAMOVLVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision encoder: {vision_encoder}')

    return vision_encoder
