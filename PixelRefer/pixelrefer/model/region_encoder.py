import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def generate_position_tensor(h, w, box_xy, raw_h, raw_w):
    box_x = box_xy[2] - box_xy[0]
    box_y = box_xy[3] - box_xy[1]
    x_tensor = torch.arange(w, dtype=torch.float32) / w  
    x_tensor = x_tensor.repeat(h, 1) 
    x_tensor = (x_tensor*box_x+box_xy[0])/(raw_w-1)

    y_tensor = torch.arange(h, dtype=torch.float32) / h 
    y_tensor = y_tensor.unsqueeze(1).repeat(1, w) 
    y_tensor = (y_tensor*box_y+box_xy[1])/(raw_h-1)

    tensor_3d = torch.stack([x_tensor, y_tensor], dim=2)
    return tensor_3d

class MaskExtractor(nn.Module):
    def __init__(self, config, mm_hidden_size, depth=2):
        super(MaskExtractor, self).__init__()
        self.mask_pooling = MaskPooling()
        modules = [nn.Linear(mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.feat_linear =  nn.Sequential(*modules)
        self.pos_linear = nn.Linear(2, mm_hidden_size)

    def forward(self, feats, masks, box_params, mask_num):
        query_feats = []
        
        if masks is None: #infer
            return None
            # masks = torch.zeros((1, 1, 336, 336)).to(feats.device).float()

        num_imgs = len(masks)
        image_idx = 0
        for idx in range(num_imgs):
            if masks[idx]==None:
                continue
            for mask_idx in range(len(masks[idx])):
                mask = masks[idx][mask_idx].unsqueeze(0).unsqueeze(0).float()
                box_param = box_params[idx][mask_idx]
                box_xy, raw_h, raw_w = box_param
                if len(mask[0])==0:
                    print('mask error')
                    mask = torch.zeros((1, 1, 336, 336)).to(feats.device).float()

                feat = feats[image_idx].unsqueeze(0)
                pos_emb = generate_position_tensor(feat.shape[1], feat.shape[2], box_xy, raw_h, raw_w).unsqueeze(0).to(feat)
                pos_emb = self.pos_linear(pos_emb)
                feat = feat+pos_emb
                
                image_idx+=1
                
                # h, w = feat.shape[1:3]
                feat = feat.permute(0,3,1,2)

                feat = feat.to(mask.dtype)
                
                mask_feat_raw = self.mask_pooling(feat, mask, mask_token_num=mask_num) # [n, 1024]

                query_feats.append(mask_feat_raw)
        if len(query_feats)==0:
            return None
        mask_feats = torch.cat(query_feats, dim=0)
        mask_feats = mask_feats.to(feats[0].dtype)
        mask_feats_linear = self.feat_linear(mask_feats)
        return mask_feats_linear

def kmeans_fast(tokens, num_clusters=10, num_iterations=5):
    n, d = tokens.shape
    centroids = tokens[torch.randperm(n)[:num_clusters]]

    for _ in range(num_iterations):
        tokens_expand = tokens.unsqueeze(1)  # [n, 1, d]
        centroids_expand = centroids.unsqueeze(0)  # [1, num_clusters, d]
        
        distances = torch.sum((tokens_expand - centroids_expand) ** 2, dim=2)  # [n, num_clusters]
        
        labels = torch.argmin(distances, dim=1)  # [n]

        new_centroids = torch.stack([tokens[labels == i].mean(dim=0) if tokens[labels == i].size(0) > 0 else centroids[i] for i in range(num_clusters)])
        
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = new_centroids
    
    return centroids

class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask, mask_token_num=10):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            x = F.interpolate(x, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            # mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if not x.device == mask.device:
            mask = mask.to(x.device)
        # b, c, h ,w = x.shape
        # b, q, h, w = mask.shape
        mask = (mask > 0).to(mask.dtype)
        mask = mask.permute(1,0,2,3)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8
       
        mask_emb = x * mask
        mask = torch.any(mask_emb != 0, dim=(0, 1))
        mask_emb = mask_emb[:,:, mask]
        mask_embedding = mask_emb[0].permute(1,0)

        if len(mask_embedding)>mask_token_num: #FIXME
            mask_embedding = kmeans_fast(mask_embedding, mask_token_num)
        return mask_embedding


def build_region_encoder(config, mm_hidden_size):

    return MaskExtractor(config, mm_hidden_size)
   