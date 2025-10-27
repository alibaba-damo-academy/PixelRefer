import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def generate_position_tensor(h, w, box_xy, raw_h, raw_w):
    box_x = box_xy[2] - box_xy[0]
    box_y = box_xy[3] - box_xy[1]
    x_tensor = torch.arange(w, dtype=torch.float32) / w  
    x_tensor = x_tensor.repeat(h, 1)  # 纵向扩展为 h 行
    x_tensor = (x_tensor*box_x+box_xy[0])/(raw_w-1)

    y_tensor = torch.arange(h, dtype=torch.float32) / h 
    y_tensor = y_tensor.unsqueeze(1).repeat(1, w) 
    y_tensor = (y_tensor*box_y+box_xy[1])/(raw_h-1)

    tensor_3d = torch.stack([x_tensor, y_tensor], dim=2)
    return tensor_3d


class MaskExtractor(nn.Module):
    def __init__(self, config, mm_hidden_size, depth=2, num_heads=8):
        super(MaskExtractor, self).__init__()
        self.mask_pooling = MaskPooling()
        modules = [nn.Linear(mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.feat_linear =  nn.Sequential(*modules)
        self.mask_embedding = nn.Conv2d(in_channels=1, out_channels=mm_hidden_size, kernel_size=1)
        self.fusion_attention_local = nn.MultiheadAttention(embed_dim=mm_hidden_size, num_heads=num_heads, batch_first=True)
        self.fusion_attention_global = nn.MultiheadAttention(embed_dim=mm_hidden_size, num_heads=num_heads, batch_first=True)
        self.ln_local = nn.LayerNorm(mm_hidden_size)
        self.ln_global = nn.LayerNorm(mm_hidden_size)
        self.pos_linear = nn.Linear(2, mm_hidden_size)
        self.num_heads = num_heads

    def fusion_image_mask_embedding(self, query, key, type):

        value = key
        if type=='local':
            query_norm = self.ln_local(query)
            attn_output = self.fusion_attention_local(query_norm, key, value)[0]
        elif type=='global':
            query_norm = self.ln_global(query)
            attn_output = self.fusion_attention_global(query_norm, key, value)[0]
        else:
            raise notImplementedError 
        fused_feature = query + attn_output
        return fused_feature

    def forward(self, feats, masks, feats_raw, local_feats_raw, masks_raw, box_params, mask_num):
        query_feats = []
        if masks is None: #infer
            return None
            # masks = torch.zeros((1, 1, 336, 336)).to(feats.device).float()

        num_imgs = len(masks)
        region_token_nums = []
        image_idx = 0
        mask_feats_all = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for idx in range(num_imgs):
                if masks[idx]==None:
                    continue
                global_feats = []
                local_feats = []
                mask_feats = []
                for mask_idx in range(len(masks[idx])):
                    mask = masks[idx][mask_idx].unsqueeze(0).unsqueeze(0).bfloat16()#.cuda()
                    box_param = box_params[idx][mask_idx]
                    box_xy, raw_h, raw_w = box_param
                    mask_raw = masks_raw[idx][mask_idx].unsqueeze(0).unsqueeze(0).bfloat16()#.cuda()
                    # if len(mask[0])==0:
                    #     print('mask error')
                    #     mask = torch.zeros((1, 1, 336, 336)).to(feats.device).float()

                    feat = feats[image_idx].unsqueeze(0)
                    pos_emb = generate_position_tensor(feat.shape[1], feat.shape[2], box_xy, raw_h, raw_w).unsqueeze(0).to(feat)
                    pos_emb = self.pos_linear(pos_emb)
                    feat = feat+pos_emb
                    
                    feat = feat.permute(0,3,1,2)
                    
                    mask_feat_raw = self.mask_pooling(feat, mask, mask_token_num=mask_num) # [n, 1024]
        
                    mask_feat_raw = torch.nn.functional.pad(mask_feat_raw, (0, 0, 0, mask_num-len(mask_feat_raw)))
                    if mask_num-len(mask_feat_raw)>0:
                        print('padding: ', mask_num-len(mask_feat_raw))
                    
                    feat_local = local_feats_raw[image_idx].unsqueeze(0)
                    feat_local = feat_local.reshape(feat_local.shape[0], -1, feat_local.shape[-1])
                    feat_raw = feats_raw[image_idx].unsqueeze(0)
                    feat_raw = feat_raw.reshape(feat_raw.shape[0], -1, feat_raw.shape[-1])

                    mask_feats.append(mask_feat_raw.unsqueeze(0))
                    local_feats.append(feat_local)
                    global_feats.append(feat_raw)
                    image_idx+=1
            
                mask_feats_flatten = torch.cat(mask_feats, dim=0)
                global_feats_flatten = torch.cat(global_feats, dim=0)
                local_feats_flatten = torch.cat(local_feats, dim=0)

                # print('local ',feat.shape)
                fused_feature = self.fusion_image_mask_embedding(mask_feats_flatten, local_feats_flatten, 'local')
                fused_feature = self.fusion_image_mask_embedding(fused_feature, global_feats_flatten, 'global')

                mask_feats_ = fused_feature.reshape(-1, fused_feature.shape[-1])
                mask_feats_all.append(mask_feats_)
            # mask_feats = mask_feats.to(feats[0].dtype)
            mask_feats_all = torch.cat(mask_feats_all, dim=0)
            mask_feats_linear = self.feat_linear(mask_feats_all)
        return mask_feats_linear

def kmeans_fast(tokens, num_iterations=3, mask_token_num=16):

    num_clusters = mask_token_num
    n, d = tokens.shape
    centroids = tokens[torch.randperm(n)[:num_clusters]]

    for _ in range(num_iterations):
        # 扩展tokens和centroids维度以计算距离，避免显式循环
        tokens_expand = tokens.unsqueeze(1)  # [n, 1, d]
        centroids_expand = centroids.unsqueeze(0)  # [1, num_clusters, d]
        
        # 计算每个token到各个中心点的距离
        distances = torch.sum((tokens_expand - centroids_expand) ** 2, dim=2)  # [n, num_clusters]
        
        # 找到每个token最近的中心点
        labels = torch.argmin(distances, dim=1)  # [n]

        # 计算新的中心点
        new_centroids = torch.stack([tokens[labels == i].mean(dim=0) if tokens[labels == i].size(0) > 0 else centroids[i] for i in range(num_clusters)])
        
        # 检查是否收敛
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = new_centroids
    
    return centroids

class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask, mask_token_num=16):
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            x = F.interpolate(x, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            # mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if not x.device == mask.device:
            mask = mask.to(x.device, non_blocking=True)
        b, c, h ,w = x.shape
        # b, q, h, w = mask.shape
        mask = mask > 0  # [B, Q, H, W]
        valid = mask.any(dim=(0, 1))

        mask_embedding = x[..., valid] 
        mask_embedding = mask_embedding.permute(2, 0, 1).reshape(-1, c)  # [N, C]
     
        if len(mask_embedding)>mask_token_num: #FIXME
            mask_embedding = kmeans_fast(mask_embedding, mask_token_num=mask_token_num)
            # mask_embedding = mask_embedding[:mask_token_num]
        return mask_embedding


def build_region_encoder(config, mm_hidden_size):

    return MaskExtractor(config, mm_hidden_size)
   