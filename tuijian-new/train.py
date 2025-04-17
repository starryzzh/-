import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_data_loader
from evaluate import evaluate
from cf import MFModel
from ncf import NCF
from config import config
import argparse
import os


def train(model_name='MF'):
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    os.makedirs('checkpoints', exist_ok=True)

    # 数据加载
    train_loader = get_data_loader('train')
    test_loader = get_data_loader('test')

    # 模型初始化
    if model_name == 'MF':
        model = MFModel().to(device)
    else:
        model = NCF().to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.BCELoss()
    scaler = torch.cuda.amp.GradScaler()

    # 训练状态
    best_hr = 0
    early_stop_counter = 0

    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            users = batch['users'].to(device, non_blocking=True)
            items = batch['items'].to(device, non_blocking=True)
            labels = batch['rating'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast():
                predictions = model(users, items)
                loss = criterion(predictions, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # 评估
        hr, ndcg = evaluate(model, test_loader, device)
        scheduler.step(hr)

        # 记录日志
        writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
        writer.add_scalar('HR/test', hr, epoch)
        writer.add_scalar('NDCG/test', ndcg, epoch)

        # 早停和模型保存
        if hr > best_hr:
            best_hr = hr
            early_stop_counter = 0
            torch.save(model.state_dict(), f'checkpoints/best_{model_name}.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f'Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, '
              f'HR@10={hr:.4f}, NDCG@10={ndcg:.4f}, Best HR={best_hr:.4f}')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['MF', 'NCF'], default='MF')
    parser.add_argument('--embedding_dim', type=int, default=config.latent_dim)
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--lr', type=float, default=config.learning_rate)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)

    args = parser.parse_args()

    # 更新配置
    config.latent_dim = args.embedding_dim
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.batch_size = args.batch_size

    train(model_name=args.model)