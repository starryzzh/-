import os
import numpy as np
import torch
import mmap
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from config import config
import time


class OptimizedDataset(Dataset):
    def __init__(self, file_path, mode='train', test_negatives=None):
        self.file_path = file_path
        self.mode = mode
        self.test_negatives = test_negatives or {}
        self._mmap = None
        self._offsets = []
        self._init_mmap()

    def _init_mmap(self):
        """内存映射初始化，提升大文件读取速度"""
        start_time = time.time()
        with open(self.file_path, 'r+b') as f:
            self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            ptr = 0
            while ptr < self._mmap.size():
                newline_pos = self._mmap.find(b'\n', ptr)
                if newline_pos == -1:
                    break
                self._offsets.append((ptr, newline_pos - ptr))
                ptr = newline_pos + 1
        print(f"MMap init time: {time.time() - start_time:.2f}s")

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        """零拷贝数据解析"""
        ptr, length = self._offsets[idx]
        line = self._mmap[ptr:ptr + length].decode('utf-8')
        parts = line.strip().split('::')

        # 关键修复：修正字段索引位置
        user = int(parts[0])
        item = int(parts[1])
        rating = float(parts[2])   # 第三个字段是rating（预处理后已转为整数）
        timestamp = int(parts[3]) if len(parts) >=4 else 0

        # 测试模式处理负样本
        if self.mode == 'test' and user in self.test_negatives:
            return {
                'user': torch.tensor(user, dtype=torch.long),
                'item': torch.tensor(item, dtype=torch.long),
                'rating': torch.tensor(rating, dtype=torch.float),
                'negatives': torch.tensor(self.test_negatives[user], dtype=torch.long)
            }

        return {
            'user': torch.tensor(user, dtype=torch.long),
            'item': torch.tensor(item, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

    def __del__(self):
        if self._mmap:
            self._mmap.close()


@lru_cache(maxsize=2)
def load_negatives_cached(test_negative_path):
    """带缓存的高效负样本加载"""
    negatives = {}
    with open(test_negative_path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) < 2:
                continue
            user = int(parts[0])
            # 修复负样本解析逻辑
            items = list(map(int, parts[1].split(','))) if ',' in parts[1] else list(map(int, parts[1:]))
            negatives[user] = items[:config.num_negatives]  # 确保数量一致
    return negatives


class BatchCollator:
    """智能批次处理器，支持动态填充"""

    def __init__(self, mode='train'):
        self.mode = mode

    def __call__(self, batch):
        # 基础字段处理
        users = torch.stack([x['user'] for x in batch])
        items = torch.stack([x['item'] for x in batch])
        ratings = torch.stack([x['rating'] for x in batch])

        batch_data = {
            'users': users,
            'items': items,
            'rating': ratings
        }

        # 测试模式负样本处理
        if self.mode == 'test' and 'negatives' in batch[0]:
            max_negs = max(len(x['negatives']) for x in batch)
            padded_negs = torch.full((len(batch), max_negs), -1, dtype=torch.long)
            mask = torch.zeros(len(batch), max_negs, dtype=torch.bool)

            for i, x in enumerate(batch):
                negs = x['negatives']
                actual_len = min(len(negs), max_negs)
                padded_negs[i, :actual_len] = negs[:actual_len]
                mask[i, :actual_len] = True

            batch_data.update({
                'negatives': padded_negs,
                'neg_mask': mask
            })

        return batch_data


def get_optimized_loader(file_type='train', batch_size=None):
    """高性能数据加载入口"""
    file_map = {
        'train': config.train_path,
        'test': config.test_path
    }

    # 动态参数设置
    batch_size = batch_size or config.batch_size
    num_workers = 0  # Windows系统必须设为0

    # 负样本处理
    test_negatives = None
    if file_type == 'test':
        test_negatives = load_negatives_cached(config.test_negative_path)

    dataset = OptimizedDataset(
        file_map[file_type],
        mode=file_type,
        test_negatives=test_negatives
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(file_type == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=BatchCollator(file_type),
        prefetch_factor=None  # 禁用预取
    )


# 兼容原有接口
get_data_loader = get_optimized_loader