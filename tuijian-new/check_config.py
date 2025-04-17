import os
from config import config

def check_config():
    # 检查路径是否存在
    print("=== 路径检查 ===")
    print(f"训练集路径: {config.train_path} \t 存在: {os.path.exists(config.train_path)}")
    print(f"测试集路径: {config.test_path} \t 存在: {os.path.exists(config.test_path)}")
    print(f"测试负样本路径: {config.test_negative_path} \t 存在: {os.path.exists(config.test_negative_path)}")

    # 检查数据格式和ID范围
    print("\n=== 数据检查 ===")
    try:
        import pandas as pd
        # 尝试读取训练集（假设是::分隔）
        df_train = pd.read_csv(config.train_path, sep='::', header=None,
                             names=['user', 'item', 'rating'], engine='python')
        print(f"训练集记录数: {len(df_train)}")
        print(f"用户ID范围: {df_train['user'].min()} ~ {df_train['user'].max()} (配置: {config.num_users})")
        print(f"物品ID范围: {df_train['item'].min()} ~ {df_train['item'].max()} (配置: {config.num_items})")

        # 检查测试负样本格式
        if os.path.exists(config.test_negative_path):
            with open(config.test_negative_path, 'r') as f:
                first_line = f.readline().strip()
                print(f"\n测试负样本首行示例: {first_line[:100]}...")  # 避免过长输出
    except Exception as e:
        print(f"数据读取失败: {str(e)}")

if __name__ == '__main__':
    check_config()