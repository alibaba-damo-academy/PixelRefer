import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class MaskExtractor(nn.Module):
    def __init__(self, config, mm_hidden_size, depth=2):
        super(MaskExtractor, self).__init__()
        self.mask_pooling = MaskPooling()
        modules = [nn.Linear(mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.feat_linear =  nn.Sequential(*modules)

    def forward(self, feats, masks):
        query_feats = []
        
        if masks is None: #infer
            return None
            # masks = torch.zeros((1, 1, 336, 336)).to(feats.device).float()

        num_imgs = len(masks)
        region_token_nums = []
        image_idx = 0
        for idx in range(num_imgs):
            if masks[idx]==None:
                continue
            for mask_idx in range(len(masks[idx])):
                mask = masks[idx][mask_idx].unsqueeze(0).unsqueeze(0).float()
                if len(mask[0])==0:
                    print('mask error')
                    mask = torch.zeros((1, 1, 336, 336)).to(feats.device).float()

                feat = feats[image_idx].unsqueeze(0)
                image_idx+=1
                
                # h, w = feat.shape[1:3]
                feat = feat.permute(0,3,1,2)

                raw_dtype = feat.dtype
                feat = feat.to(mask.dtype)
                
                mask_feat_raw = self.mask_pooling(feat, mask) # [n, 1024]

                query_feats.append(mask_feat_raw)
        if len(query_feats)==0:
            return None
        mask_feats = torch.cat(query_feats, dim=0)
        mask_feats = mask_feats.to(feats[0].dtype)
        mask_feats_linear = self.feat_linear(mask_feats)
        return mask_feats_linear

def kmeans_fast(tokens, num_clusters=10, num_iterations=20):
    # tokens: 输入的token数据，shape为[n, d]
    # num_clusters: 压缩后的组数
    # num_iterations: K-means算法的迭代次数
    
    # 初始化中心点
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

    def forward(self, x, mask):

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

        if len(mask_embedding)>10: #FIXME
            mask_embedding = kmeans_fast(mask_embedding)

        return mask_embedding


def build_region_encoder(config, mm_hidden_size):

    return MaskExtractor(config, mm_hidden_size)
   