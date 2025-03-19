import torch
import torch.nn as nn
from config import config

class NCF(nn.Module):
    """神经协同过滤模型（兼容原有架构）"""
    def __init__(self):
        super().__init__()
        from config import config
        
        self.user_emb = nn.Embedding(config.num_users, config.latent_dim)
        self.item_emb = nn.Embedding(config.num_items, config.latent_dim)
        
        self.mlp = nn.Sequential()
        input_dim = 2 * config.latent_dim
        for i, units in enumerate(config.ncf_layers):
            self.mlp.add_module(f'fc_{i}', nn.Linear(input_dim, units))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
            self.mlp.add_module(f'dropout_{i}', nn.Dropout(config.dropout))
            input_dim = units
        self.output = nn.Linear(input_dim, 1)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        
        # 特征交互
        interaction = torch.cat([user_emb, item_emb], dim=1)
        mlp_out = self.mlp(interaction)
        
        return torch.sigmoid(self.output(mlp_out).squeeze())
