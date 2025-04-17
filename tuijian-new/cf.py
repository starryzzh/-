import torch
import torch.nn as nn
from config import config


class MFModel(nn.Module):
    """
    改进的矩阵分解模型
    主要优化：
    - 评分归一化处理
    - 自适应设备支持
    - 数值稳定性增强
    """

    def __init__(self):
        super().__init__()
        # 嵌入层 + 冷启动处理（padding_idx=0）
        self.user_emb = nn.Embedding(
            num_embeddings=config.num_users + 1,
            embedding_dim=config.latent_dim,
            padding_idx=0
        )
        self.item_emb = nn.Embedding(
            num_embeddings=config.num_items + 1,
            embedding_dim=config.latent_dim,
            padding_idx=0
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """自适应初始化策略"""
        nn.init.xavier_normal_(self.user_emb.weight[1:])  # 跳过padding索引
        nn.init.xavier_normal_(self.item_emb.weight[1:])

    def forward(self, users, items):
        """
        前向传播
        参数：
            users: [batch_size] 用户ID
            items: [batch_size] 物品ID
        返回：
            预测评分 [batch_size] (范围0-1)
        """
        # 嵌入查找
        u = self.user_emb(users)  # [batch_size, latent_dim]
        i = self.item_emb(items)  # [batch_size, latent_dim]

        # 交互计算（元素乘积）
        interaction = u * i  # [batch_size, latent_dim]

        # 预测值计算（带数值稳定处理）
        pred = torch.sum(interaction, dim=1)  # [batch_size]
        pred = torch.clamp(pred, -10, 10)  # 防止梯度爆炸

        # Sigmoid激活确保输出在0-1之间
        return torch.sigmoid(pred)  # 关键修改：添加激活函数

    def l2_loss(self):
        """计算正则化损失"""
        loss = 0.0
        loss += torch.sum(self.user_emb.weight[1:]  **  2) *config.reg
        loss += torch.sum(self.item_emb.weight[1:]  **  2) *config.reg
        return loss


# 兼容原有接口
NeuralCF = MFModel  # 如需兼容NCF框架