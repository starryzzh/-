import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import kaiming_normal_, xavier_normal_, constant_
from config import config
from typing import Dict, Optional


class NCF(nn.Module):
    """改进的神经协同过滤模型

    主要改进：
    - 共享基础嵌入层
    - 动态深度MLP
    - 冷启动处理
    - 多任务学习支持
    - 自适应温度控制
    - 混合精度优化
    """

    def __init__(self):
        super().__init__()

        # 共享的基础嵌入
        self.user_emb = self._create_embedding(config.num_users, config.latent_dim)
        self.item_emb = self._create_embedding(config.num_items, config.latent_dim)

        # 投影层（GMF/MLP路径）
        self.gmf_user_proj = self._create_projection()
        self.gmf_item_proj = self._create_projection()
        self.mlp_user_proj = self._create_projection()
        self.mlp_item_proj = self._create_projection()

        # 动态深度MLP
        self.mlp = self._build_mlp()
        self.mlp_dropout = nn.Dropout(config.ncf_mlp_dropout)

        # 预测层
        self.fc = nn.Linear(config.latent_dim + config.mlp_units[-1], 1)

        # 多任务输出
        self.task_heads = nn.ModuleDict({
            'ctr': nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()),
            'rating': nn.Linear(1, 1)
        })

        # 动态温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 初始化
        self._init_weights()

    def _create_embedding(self, num: int, dim: int) -> nn.Embedding:
        """创建带冷启动处理的嵌入层"""
        return nn.Embedding(
            num_embeddings=num + 1,  # 0为未知ID
            embedding_dim=dim,
            padding_idx=0
        )

    def _create_projection(self) -> nn.Sequential:
        """创建带层归一化的投影层"""
        return nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.ReLU()
        )

    def _build_mlp(self) -> nn.ModuleList:
        """动态构建MLP层"""
        layers = nn.ModuleList()
        input_dim = 2 * config.latent_dim

        for units in config.ncf_mlp_units:
            layers.extend([
                nn.Linear(input_dim, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(config.ncf_mlp_dropout)
            ])
            input_dim = units

        return layers

    def _init_weights(self):
        """自适应初始化策略"""
        # 嵌入层
        xavier_normal_(self.user_emb.weight.data[1:])  # 跳过padding索引
        xavier_normal_(self.item_emb.weight.data[1:])

        # 投影层
        for proj in [self.gmf_user_proj, self.gmf_item_proj,
                     self.mlp_user_proj, self.mlp_item_proj]:
            kaiming_normal_(proj[0].weight, mode='fan_out')
            constant_(proj[0].bias, 0.0)

        # MLP层
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                kaiming_normal_(layer.weight, mode='fan_out')
                constant_(layer.bias, 0.0)

        # 预测层
        xavier_normal_(self.fc.weight)
        constant_(self.fc.bias, 0.0)

    def forward(self,
                users: torch.LongTensor,
                items: torch.LongTensor,
                task: str = 'ctr') -> torch.Tensor:
        """
        前向传播
        参数：
            users: [batch_size] 用户ID
            items: [batch_size] 物品ID
            task: 预测任务类型 ('ctr'或'rating')
        返回：
            预测值 [batch_size]
        """
        # 冷启动处理
        users = self._mask_unknown(users, is_user=True)
        items = self._mask_unknown(items, is_user=False)

        # 基础嵌入
        u_base = self.user_emb(users)
        i_base = self.item_emb(items)

        # GMF路径
        u_gmf = self.gmf_user_proj(u_base)
        i_gmf = self.gmf_item_proj(i_base)
        gmf_out = u_gmf * i_gmf  # 逐元素积

        # MLP路径
        u_mlp = self.mlp_user_proj(u_base)
        i_mlp = self.mlp_item_proj(i_base)
        mlp_input = torch.cat([u_mlp, i_mlp], dim=1)

        for layer in self.mlp:
            mlp_input = layer(mlp_input)
        mlp_out = self.mlp_dropout(mlp_input)

        # 合并特征
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        logits = self.fc(combined).squeeze()

        # 任务特定输出
        if task == 'rating':
            output = self.task_heads['rating'](logits)
            output = torch.clamp(output, 1.0, 5.0)
        else:
            scaled_logits = logits / self.temperature.clamp(min=0.5, max=2.0)
            output = self.task_heads['ctr'](scaled_logits)

        return output

    def _mask_unknown(self,
                      ids: torch.LongTensor,
                      is_user: bool) -> Tensor:
        """处理未知ID"""
        max_id = self.user_emb.num_embeddings - 1 if is_user else self.item_emb.num_embeddings - 1
        return torch.where((ids < 1) | (ids > max_id),
                           torch.zeros_like(ids), ids)

    def l2_loss(self) -> torch.Tensor:
        """计算正则化损失"""
        loss = 0.0
        # 嵌入正则
        loss += self.user_emb.weight.norm(dim=1).pow(2).mean() * config.ncf_emb_reg
        loss += self.item_emb.weight.norm(dim=1).pow(2).mean() * config.ncf_emb_reg
        # 投影层正则
        for proj in [self.gmf_user_proj, self.gmf_item_proj]:
            loss += proj[0].weight.norm().pow(2) * config.ncf_proj_reg
        return loss

    def batch_predict(self,
                      users: torch.LongTensor,
                      items: torch.LongTensor) -> Dict[str, torch.Tensor]:
        """批量多任务预测"""
        return {
            'ctr': self.forward(users, items, 'ctr'),
            'rating': self.forward(users, items, 'rating')
        }


class NCFLoss(nn.Module):
    """多任务损失函数"""

    def __init__(self):
        super().__init__()
        self.ctr_loss = nn.BCELoss()
        self.rating_loss = nn.MSELoss()

    def forward(self,
                preds: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                model: NCF) -> torch.Tensor:
        # 任务损失
        ctr_loss = self.ctr_loss(preds['ctr'], targets['ctr'])
        rating_loss = self.rating_loss(preds['rating'], targets['rating'])

        # 正则化
        reg_loss = model.l2_loss()

        return ctr_loss + rating_loss + reg_loss


# 兼容原有接口
NeuralCF = NCF