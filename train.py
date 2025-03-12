import argparse
import mlflow
import wandb
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

# 实验配置
EXPERIMENT_SETTINGS = {
    'dataset': 'ml-1m',
    'benchmark': ['CF', 'MF', 'DSSM'],
    'train_batch_size': 512,
    'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}},
}

# 初始化实验跟踪
mlflow.set_tracking_uri("http://localhost:5000")
wandb.init(project="recsys-reproducibility")


def prepare_data(config_dict):
    """数据预处理流水线"""
    config = Config(config_dict=config_dict)
    init_seed(2023)
    init_logger(config)
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return train_data, valid_data, test_data


def train_model(model_name, train_data, valid_data, config):
    """模型训练流程"""
    with mlflow.start_run():
        # 记录超参数
        mlflow.log_params({
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'embedding_size': config['embedding_size']
        })
        
        # 初始化模型
        model_class = get_model_class(model_name)
        model = model_class(config, train_data.dataset).to(config['device'])
        
        # 训练过程
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        
        # 评估并记录指标
        test_result = trainer.evaluate(test_data)
        mlflow.log_metrics(test_result)
        wandb.log(test_result)
        
        return model


def get_model_class(model_name):
    """模型工厂方法"""
    model_map = {
        'CF': 'RecBoleCFModel',
        'MF': 'RecBoleMFModel',
        'DSSM': 'CornacDSSM'
    }
    return __import__(f'models.{model_map[model_name]}', fromlist=[''])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', nargs='+', default=EXPERIMENT_SETTINGS['benchmark'])
    parser.add_argument('--dataset', default=EXPERIMENT_SETTINGS['dataset'])
    args = parser.parse_args()
    
    # 准备标准化数据集
    train_data, valid_data, test_data = prepare_data(EXPERIMENT_SETTINGS)
    
    # 多算法对比实验
    for algo in args.algorithms:
        print(f"Training {algo}...")
        model = train_model(algo, train_data, valid_data, EXPERIMENT_SETTINGS)