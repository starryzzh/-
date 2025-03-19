import torch
import torch.nn as nn
from config import config

class MFModel(nn.Module):
    """矩阵分解模型（兼容原有架构）"""
    def __init__(self):
        super().__init__()
        from config import config
        
        self.user_emb = nn.Embedding(config.num_users, config.latent_dim)
        self.item_emb = nn.Embedding(config.num_items, config.latent_dim)
        self.user_bias = nn.Embedding(config.num_users, 1)
        self.item_bias = nn.Embedding(config.num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        pred = (self.global_bias
                + self.user_bias(user).squeeze()
                + self.item_bias(item).squeeze()
                + (user_emb * item_emb).sum(dim=1))
        return torch.sigmoid(pred)
