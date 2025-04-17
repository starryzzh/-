import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from config import config
import shutil
from tqdm import tqdm

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DELIMITER = '::'
REQUIRED_FILES = ['ratings.dat', 'users.dat', 'movies.dat']


def validate_input():
    """验证输入文件完整性"""
    logger.info("验证原始数据文件...")
    missing = [f for f in REQUIRED_FILES if not Path(config.data_dir).joinpath(f).exists()]
    if missing:
        raise FileNotFoundError(f"缺失必要文件: {', '.join(missing)}")

    # 验证ratings.dat格式样例
    with open(Path(config.data_dir) / 'ratings.dat', 'r') as f:
        sample_line = f.readline().strip()
        if len(sample_line.split(DELIMITER)) != 4:
            raise ValueError(f"ratings.dat格式错误，示例行: {sample_line}")

        # 新增：验证评分是否为整数
        rating = sample_line.split(DELIMITER)[2]
        if not rating.isdigit():
            raise ValueError(f"评分值必须为整数，但发现: {rating}")


def clear_output():
    """清理旧输出文件"""
    output_files = [
        config.train_path,
        config.test_path,
        config.test_negative_path
    ]
    for f in output_files:
        path = Path(f)
        if path.exists():
            logger.info(f"删除旧文件: {path}")
            path.unlink(missing_ok=True)


def process_user_data(user_id, user_ratings):
    """处理单个用户的数据"""
    try:
        # 按时间排序
        sorted_ratings = sorted(user_ratings, key=lambda x: x['timestamp'])
        split_idx = int(len(sorted_ratings) * 0.8)

        # 训练数据
        train_records = []
        for r in sorted_ratings[:split_idx]:
            # 关键修改：将评分强制转换为整数
            normalized_rating = (int(r['rating']) - 1) / 4.0  # 归一化到0-1
            train_records.append(
                f"{user_id}{DELIMITER}{r['item_id']}{DELIMITER}{normalized_rating}{DELIMITER}{int(r['timestamp'])}"
            )
        # 测试数据
        test_record = None
        if len(sorted_ratings) > split_idx:
            test_rating = sorted_ratings[split_idx]
            test_record = (
                f"{user_id}{DELIMITER}{test_rating['item_id']}{DELIMITER}"
                f"{int(test_rating['rating'])}{DELIMITER}{test_rating['timestamp']}"  # 强制转换
            )

        # 生成负样本
        positive_items = {r['item_id'] for r in sorted_ratings}
        negatives = []
        attempts = 0
        while len(negatives) < config.num_negatives and attempts < 1000:
            item = np.random.randint(1, config.num_items + 1)
            if item not in positive_items:
                negatives.append(str(item))
            attempts += 1

        return {
            'train': train_records,
            'test': test_record,
            'negatives': negatives
        }
    except Exception as e:
        logger.error(f"处理用户 {user_id} 失败: {str(e)}", exc_info=True)
        return None


def process_ratings():
    """主处理流程"""
    try:
        logger.info("启动数据处理流程")
        validate_input()
        clear_output()

        # 初始化输出文件
        with open(config.train_path, 'w') as train_f, \
                open(config.test_path, 'w') as test_f, \
                open(config.test_negative_path, 'w') as neg_f:

            # 读取评分数据
            ratings = pd.read_csv(
                Path(config.data_dir) / 'ratings.dat',
                sep=DELIMITER,
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                dtype={
                    'user_id': 'int32',
                    'item_id': 'int32',
                    'rating': 'int32',  # 关键修改：直接读取为整数
                    'timestamp': 'int32'
                },
                engine='python'
            )

            # 按用户分组处理
            user_groups = ratings.groupby('user_id')
            for user_id, group in tqdm(user_groups, desc="处理用户数据"):
                user_data = process_user_data(
                    user_id,
                    group.to_dict('records')
                )
                if not user_data:
                    continue

                # 写入训练数据
                if user_data['train']:
                    train_f.write('\n'.join(user_data['train']) + '\n')

                # 写入测试数据
                if user_data['test']:
                    test_f.write(user_data['test'] + '\n')
                    neg_line = DELIMITER.join([str(user_id)] + user_data['negatives'])
                    neg_f.write(neg_line + '\n')

                # 定期刷新缓冲区
                if user_id % 1000 == 0:
                    train_f.flush()
                    test_f.flush()
                    neg_f.flush()

        logger.info("数据处理完成")

    except Exception as e:
        logger.error("数据处理流程异常终止", exc_info=True)
        raise


def validate_output():
    """验证输出文件"""
    logger.info("验证输出文件...")

    # 检查文件存在性
    required_files = {
        config.train_path: (4, 100_000),
        config.test_path: (4, 10_000),
        config.test_negative_path: (config.num_negatives + 1, 10_000)
    }

    for file_path, (field_count, min_lines) in required_files.items():
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"输出文件不存在: {path}")

        # 检查文件内容
        with open(file_path) as f:
            lines = f.readlines()
            if len(lines) < min_lines:
                raise ValueError(
                    f"文件 {path.name} 行数不足，期望至少 {min_lines} 行，实际 {len(lines)} 行"
                )

            sample = lines[0].strip().split(DELIMITER)
            if len(sample) != field_count:
                raise ValueError(
                    f"文件 {path.name} 字段数错误，期望 {field_count} 字段，实际 {len(sample)} 字段"
                )

    logger.info("所有输出文件验证通过")


if __name__ == '__main__':
    process_ratings()
    validate_output()