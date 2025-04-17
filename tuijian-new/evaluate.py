import torch
import numpy as np
from tqdm import tqdm
from config import config


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def batch_predict(self, users, items):
        """高效批量预测"""
        with torch.no_grad():
            users_exp = users.view(-1, 1).expand(-1, items.shape[1]).contiguous()
            items_flat = items.contiguous().view(-1)
            return self.model(users_exp.view(-1), items_flat).view(items.shape)

    def compute_metrics(self, scores, pos_idx=0):
        """计算多种评估指标"""
        metrics = {}

        # 多K值评估
        for k in [5, 10, 20]:
            _, topk = torch.topk(scores, k, dim=1)
            hits = (topk == pos_idx).any(dim=1).float()
            metrics[f'HR@{k}'] = hits.mean().item()

            # 计算NDCG
            pos_mask = (topk == pos_idx)
            ranks = pos_mask.nonzero()[:, 1] + 1
            if ranks.numel() > 0:
                ndcg = (1 / torch.log2(ranks + 1)).sum() / len(scores)
                metrics[f'NDCG@{k}'] = ndcg.item()
            else:
                metrics[f'NDCG@{k}'] = 0.0

        # 计算AUC
        pos_scores = scores[:, pos_idx]
        neg_scores = scores[:, 1:].mean(dim=1)
        auc = (pos_scores > neg_scores).float().mean().item()
        metrics['AUC'] = auc

        return metrics

    def evaluate(self, test_loader):
        """主评估函数"""
        all_metrics = {}
        progress = tqdm(test_loader, desc='Evaluating', leave=False)

        for batch in progress:
            users = batch['users'].to(self.device)
            items = batch['items'].to(self.device)
            negatives = batch.get('negatives', None)

            if negatives is not None:
                # 处理预定义负样本
                neg_items = negatives.to(self.device)
                all_items = torch.cat([items.unsqueeze(1), neg_items], dim=1)
            else:
                # 处理隐式反馈
                all_items = items.unsqueeze(1)

            # 批量预测
            scores = self.batch_predict(users, all_items)

            # 计算指标
            batch_metrics = self.compute_metrics(scores)

            # 聚合结果
            for k, v in batch_metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

        # 计算平均指标
        return {k: np.mean(v) for k, v in all_metrics.items()}


def evaluate(model, test_loader, device):
    """兼容原有接口"""
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    return metrics.get(f'HR@{config.top_k}', 0), metrics.get(f'NDCG@{config.top_k}', 0)