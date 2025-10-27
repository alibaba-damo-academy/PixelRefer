# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import math
from abc import ABC, abstractmethod

import einops
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

from ..constants import IGNORE_INDEX, MODAL_INDEX_MAP, NUM_FRAMES
from .encoder import build_vision_encoder
from .projector import build_vision_projector, load_mm_projector
from .region_encoder import build_region_encoder
from ..mm_utils import reshape_images_to_raw_grid, split_images, get_image_split_length

def spatial_downsampling(features, grid_thws, strides):
    n, c = features.shape

    flatten_grid_thws = torch.cat([grid_thw for batch_grid_thws in grid_thws for grid_thw in batch_grid_thws])
    split_sizes = [grid_thw.prod() for grid_thw in flatten_grid_thws]
    features = torch.split(features, split_sizes)
    flatten_strides = [stride for batch_strides in strides for stride in batch_strides]

    new_features = []
    for feature, grid_thw, stride in zip(features, flatten_grid_thws, flatten_strides):
        # NOTE: adapted for reshape in image processor 
        feature = feature.view(grid_thw[0], grid_thw[1] // stride, grid_thw[2] // stride, stride, stride,  c).permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(grid_thw[0], grid_thw[1], grid_thw[2], c).permute(0, 3, 1, 2)
        # NOTE: previous version model is align_corners=True
        new_feature = torch.nn.functional.interpolate(feature, (math.ceil(grid_thw[1] / stride), math.ceil(grid_thw[2] / stride)), mode='bilinear')
        # new_feature = nn.functional.avg_pool2d(feature, stride)
        # new_feature = nn.functional.max_pool2d(feature, stride)
        new_features.append(new_feature.permute(0, 2, 3, 1).view(-1, c))
    new_features = torch.cat(new_features)

    return new_features


class PixelreferliteMetaModel:

    def __init__(self, config):
        super(PixelreferliteMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_encoder"):
            self.vision_encoder = build_vision_encoder(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
        self.region_encoder = build_region_encoder(config, self.vision_encoder.hidden_size) 

    def get_vision_encoder(self):
        vision_encoder = getattr(self, 'vision_encoder', None)
        if type(vision_encoder) is list:
            vision_encoder = vision_encoder[0]
        return vision_encoder

    def get_mm_projector(self):
        return self.mm_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_encoder = model_args.vision_encoder
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_projector = model_args.pretrain_mm_projector

        self.config.mm_vision_encoder = vision_encoder

        if self.get_vision_encoder() is None:
            vision_encoder = build_vision_encoder(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_encoder = [vision_encoder]
            else:
                self.vision_encoder = vision_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_encoder = self.vision_encoder[0]
            else:
                vision_encoder = self.vision_encoder
            # NOTE: only compatible with delay_load encoder
            # vision_encoder.load_model(vision_encoder.cfg_only)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_encoder.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_projector is not None:
            if os.path.exists(pretrain_mm_projector):
                is_local = True
                if os.path.isdir(pretrain_mm_projector):
                    mm_projector_weights = load_mm_projector(pretrain_mm_projector)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_projector, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_projector = pretrain_mm_projector.replace('mm_projector.bin', '')
                pretrain_mm_projector = pretrain_mm_projector.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_projector)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


class PixelreferliteMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def spatial_merge_size(self):
        if hasattr(self.config, 'spatial_merge_size'):
            return self.config.spatial_merge_size
        else:
            return 1

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(self,images, grid_thws, strides):
        """
            images shape [b c h w]
        """
        images_features = self.get_model().get_vision_encoder()(images, grid_thws=grid_thws, strides=strides)
        # images_features = spatial_downsampling(images_features, grid_thws, stride=self.config.spatial_merge_size)
        mm_features = spatial_downsampling(images_features, grid_thws, strides=strides)
        images_features = self.get_model().mm_projector(mm_features)

        return images_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, position_ids=None, masks=None, masks_raw=None, additional_images = None, additional_images_raw = None, local_images_raw = None, mm_type=None, new_token_types=None, box_params=None
    ):
        if self.config.use_token_compression:
            return self.prepare_inputs_labels_for_multimodal_with_compression(input_ids, attention_mask, past_key_values, labels, images, position_ids, masks, masks_raw, additional_images, additional_images_raw, local_images_raw, mm_type, new_token_types, box_params)

        # # images shape (modal, tensor, flag)
        # vision_encoder = self.get_vision_encoder()
        # # NOTE: text-only situation
        # if vision_encoder is None or images is None or input_ids.shape[1] == 1:
        #     return input_ids, attention_mask, past_key_values, None, labels, position_ids

        # # NOTE: Equvialent to the following code:
        # # images_tensor = [image      for modal, image, image_flag, grid_thw in images]
        # # images_flag   = [image_flag for modal, image, image_flag, grid_thw in images]
        # # grid_thws     = [grid_thw   for modal, image, image_flag, grid_thw in images]
        # modals, images, grid_thws = zip(*images)

        # images_flag = []
        # strides = []
        # for modal, grid_thw in zip(modals, grid_thws):
        #     grid_thw = torch.cat(grid_thw)
        #     stride = self.config.spatial_merge_size if modal == "video" else 1
        #     num_patches = grid_thw.prod(dim=-1).sum().div(stride**2).long()
        #     image_flag = torch.full((num_patches, ), 0 if modal == 'text' else 1)
        #     images_flag.append(image_flag)
        #     strides.append([stride] * grid_thw.size(0))
        # images_flag_tensor = torch.cat(images_flag)

        # mm_features = self.encode_images(images, grid_thws, strides)
        # mm_features = mm_features[images_flag_tensor.to(mm_features.device) == 1].to(input_ids.device)
        
        # additional_images_list = []
        # additional_images_thw = []
        # additional_images_strides = []
        
        # for i in range(len(additional_images)):
        #     additional_images_list.append(torch.from_numpy(np.array(additional_images[0][0])).to(mm_features.dtype).to(mm_features.device))
        #     additional_images_thw.append(torch.tensor(additional_images[0][1][0]).to(mm_features.device))
        #     additional_images_strides.append([1]*len(additional_images[0][1][0]))
            

        # image_selected = (input_ids == self.config.image_token_index)
        # audio_selected = (input_ids == MODAL_INDEX_MAP['<audio>'])
        # input_ids[image_selected] = 0
        # input_ids[audio_selected] = 0

        # input_embeds = self.get_model().embed_tokens(input_ids).clone()

        # B, N, C = input_embeds.shape
        # input_embeds = input_embeds.reshape(B * N, C).to(input_ids.device)
        # image_selected = image_selected.reshape(B * N)
        # audio_selected = audio_selected.reshape(B * N)

        # input_embeds[image_selected] = input_embeds[image_selected] * 0.0 + mm_features.reshape(-1, C)

        # # replace region token
        # mask_selected = (input_ids == self.config.region_token_index)
        # if mask_selected.sum()>0:
            
        #     additional_images_features = self.get_model().get_vision_encoder()(additional_images_list, grid_thws=[additional_images_thw], strides=additional_images_strides)
        #     reshaped_features = reshape_images_to_raw_grid(additional_images_features, additional_images_thw)
        #     mask_additional_image_features = []
        #     for idx in mask_ids:
        #         mask_additional_image_features.append(reshaped_features[idx])
        #     mask_feats = self.model.region_encoder(mask_additional_image_features, masks)
        #     input_embeds[mask_selected] = input_embeds[mask_selected]*0.0 + mask_feats

        
        # input_embeds = input_embeds.reshape(B, N, C)

        # return None, attention_mask, past_key_values, input_embeds, labels, position_ids

    def prepare_inputs_labels_for_multimodal_with_compression(
        self, input_ids, attention_mask, past_key_values, labels, images, position_ids=None, masks=None, masks_raw=None, additional_images = None, additional_images_raw = None, local_images_raw=None, mm_type=None, new_token_types=None, box_params=None
    ):

        # images shape (modal, tensor, flag)
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or additional_images is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels, position_ids


        # NOTE: Equvialent to the following code:
        # images_tensor = [image      for modal, image, image_flag, grid_thw in images]
        # images_flag   = [image_flag for modal, image, image_flag, grid_thw in images]
        # grid_thws     = [grid_thw   for modal, image, image_flag, grid_thw in images]
        # modals, images, grid_thws = zip(*images)

        images_flag = []
        # visual_masks = []
        strides = []
        # visual_trunc_masks = []
        
        # for modal, image, grid_thw in zip(modals, images, grid_thws):
        #     grid_thw = torch.cat(grid_thw)
        #     stride = self.config.spatial_merge_size if modal == "video" else 1
        #     num_patches = grid_thw.prod(dim=-1).sum().div(stride**2).long()
        #     image_flag = torch.full((num_patches, ), 0 if modal == 'text' else 1)
        #     images_flag.append(image_flag)
        #     strides.append([stride] * grid_thw.size(0))

            # if modal == "image" or (modal == "video" and len(image) == 1):
                # visual_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))
            #     visual_trunc_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

            # elif modal == "video":
                # NOTE: video frame compressor
                # n, h, w = len(image), grid_thw[0][1], grid_thw[0][2]
                # image = torch.stack(image, dim=0).view(n, (h // stride) * (w // stride), -1)

                # threshold = 0.1
                # min_tokens = 1
                # pixel_diff = image[1:] - image[:-1]
                # pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                # pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                # # if dist.get_rank() == 0:
                # #     print(pixel_diff.shape, image.shape)
                # mask = pixel_diff > threshold
                # padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                # # mask[padding_ids, torch.randperm(min_tokens)] = 1
                # mask[padding_ids, :min_tokens] = 1
                # visual_masks.append(mask.flatten())
            #     visual_trunc_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

            # elif modal == "text":
            #     visual_trunc_masks.append(torch.ones((0,), dtype=torch.bool, device=input_ids.device))

        # images_flag_tensor = torch.cat(images_flag)
        # mm_features = self.encode_images(images, grid_thws, strides)
   
        # # mm_features = mm_features[images_flag_tensor.to(mm_features.device) == 1]

        # img_start_pos, img_end_pos = get_image_split_length(grid_thws)
     
        # # slow tokens
        # try:
        #     mm_features_split = split_images(mm_features, grid_thws)
        # except:

        # slow_image_features = mm_features


        # if len(mm_features_new)>0:
        #     mm_features_new = torch.cat(mm_features_new, dim=0)
        # else:
        #     mm_features_new = None


        additional_images_list = []
        additional_images_thw = []
        additional_images_strides = []

        for i in range(len(additional_images)):
            for img_idx in range(len(additional_images[i][0])):
                additional_images_list.append([additional_images[i][0][img_idx].bfloat16()])
                additional_images_thw.append([additional_images[i][1][img_idx]])
                additional_images_strides.append([1]*len(additional_images[i][1][img_idx]))

        additional_images = additional_images_raw
        additional_images_list_raw = []
        additional_images_thw_raw = []
        additional_images_strides_raw = []
        for i in range(len(additional_images)):
            for img_idx in range(len(additional_images[i][0])):
                additional_images_list_raw.append([additional_images[i][0][img_idx].bfloat16()])
                additional_images_thw_raw.append([additional_images[i][1][img_idx]])
                additional_images_strides_raw.append([1]*len(additional_images[i][1][img_idx]))

        additional_images = local_images_raw
        local_images_list_raw = []
        local_images_thw_raw = []
        local_images_strides_raw = []
        for i in range(len(additional_images)):
            for img_idx in range(len(additional_images[i][0])):
                local_images_list_raw.append([additional_images[i][0][img_idx].bfloat16()])
                local_images_thw_raw.append([additional_images[i][1][img_idx]])
                local_images_strides_raw.append([1]*len(additional_images[i][1][img_idx]))

        B, N = input_ids.shape
        # C = mm_features.shape[-1]

        assert B == 1, "Only support batch flattening for now"
        input_ids = input_ids.view(B * N)
        image_selected = (input_ids == self.config.image_token_index)
        audio_selected = (input_ids == MODAL_INDEX_MAP['<audio>'])

        # if len(visual_masks) > 0:
        #     # if dist.get_rank() == 0:
        #     #     print(grid_thws, [x.shape for x in visual_masks])
        #     visual_masks = torch.cat(visual_masks)
        #     # print((visual_masks == 1).sum(), (visual_masks == 0).sum())

        #     mm_features = mm_features[visual_masks]
        #     # text_masks = torch.zeros_like(input_ids, dtype=torch.bool)
        #     # text_masks[~image_selected] = True
        #     text_masks = torch.logical_not(image_selected)

        #     try:
        #         text_masks[image_selected] = visual_masks
        #     except Exception as e:
        #         assert position_ids is not None, "Position ids must be provided when shapes mismatch"
        #         print(
        #             f'warning: {e}, text_masks[image_selected].shape={text_masks[image_selected].shape},',
        #             f'visual_masks.shape={visual_masks.shape}'
        #         )

        #         seq_end_indices = torch.nonzero(position_ids.view(B * N) == 0)[:, 0]
        #         seq_end_indices = seq_end_indices[seq_end_indices > 0]
        #         seq_end_indices = seq_end_indices.tolist()+ [len(input_ids)]
        #         seq_start_indices = [0] + seq_end_indices[:-1]
        #         num_visual_tokens = [
        #             input_ids[start:end].eq(self.config.image_token_index).sum()
        #             for start, end in zip(seq_start_indices, seq_end_indices)
        #         ]

        #         for n, mask in zip(num_visual_tokens, visual_trunc_masks):
        #             if len(mask) > 0:
        #                 mask[n:] = False
        #         visual_trunc_masks = torch.cat(visual_trunc_masks)

        #         text_masks[image_selected] = visual_masks[visual_trunc_masks]
        #         mm_features = mm_features[visual_trunc_masks[visual_masks]]

        # else:
        text_masks = torch.ones_like(input_ids, dtype=torch.bool)

        input_ids = input_ids[text_masks]
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)[text_masks].reshape(1, -1)
        if labels is not None:
            labels = labels.view(B * N)[text_masks].reshape(1, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([torch.arange(end - start, device=input_ids.device) for start, end in zip(pos_start, pos_end)])
            position_ids = position_ids.reshape(1, -1)

        # image_selected = (input_ids == self.config.image_token_index)
        # audio_selected = (input_ids == MODAL_INDEX_MAP['<audio>'])
        # input_ids[image_selected] = 0
        # input_ids[audio_selected] = 0

        input_embeds = self.get_model().embed_tokens(input_ids).clone()
        C = input_embeds.shape[-1]

        # if image_selected.sum()>0:
        #     input_embeds[image_selected] = input_embeds[image_selected] * 0.0 + mm_features_new.reshape(-1, C)
        
        # replace region token
        mask_selected = (input_ids == self.config.region_token_index)
        
        # try:
        if mask_selected.sum()>0:

            additional_images_features = self.get_model().get_vision_encoder()(additional_images_list, grid_thws=additional_images_thw, strides=additional_images_strides)
            additional_images_features = spatial_downsampling(additional_images_features, additional_images_thw, strides=additional_images_strides)
            additional_images_features_raw = self.get_model().get_vision_encoder()(additional_images_list_raw, grid_thws=additional_images_thw_raw, strides=additional_images_strides_raw)
            local_images_features_raw = self.get_model().get_vision_encoder()(local_images_list_raw, grid_thws=local_images_thw_raw, strides=local_images_strides_raw)
            
            reshaped_features = reshape_images_to_raw_grid(additional_images_features, additional_images_thw)
            reshaped_features_raw = reshape_images_to_raw_grid(additional_images_features_raw, additional_images_thw_raw)
            reshaped_features_raw_local = reshape_images_to_raw_grid(local_images_features_raw, local_images_thw_raw)

            mask_feats = self.model.region_encoder(reshaped_features, masks, reshaped_features_raw, reshaped_features_raw_local, masks_raw, box_params, mask_num=self.config.mask_num)
            input_embeds[mask_selected] = input_embeds[mask_selected]*0.0 + mask_feats
        # except Exception as e:
        #     print(e)

        new_input_embeds = input_embeds.reshape(1, -1, C)


        return None, attention_mask, past_key_values, new_input_embeds, labels, position_ids
