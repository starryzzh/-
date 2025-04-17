from dataclasses import dataclass, field
import os
from typing import List

@dataclass
class Config:
    cf_dropout: float = 0.2  # MF模型专用丢弃率
    # 数据配置
    data_dir: str = os.path.normpath(r"D:\tuijian\ml-1m")
    num_users: int = 6040
    num_items: int = 3952
    train_path: str = os.path.join(data_dir, 'train.rating')
    test_path: str = os.path.join(data_dir, 'test.rating')
    test_negative_path: str = os.path.join(data_dir, 'test.negative')

    # 模型参数
    num_negatives: int = 4
    latent_dim: int = 64
    reg: float = 1e-5
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 20

    # 模型特定参数
    mf_hidden: int = 8
    ncf_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.2


    # 评估参数
    top_k: int = 10

config = Config()