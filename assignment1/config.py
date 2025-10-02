"""
配置模块
处理训练配置和命令行参数
"""

import argparse
import json
import os
import numpy as np


class Config:
    """训练配置类"""
    
    def __init__(self):
        """初始化默认配置"""
        # 网络结构配置
        self.layer_sizes = [784, 256, 128, 10]
        self.activation = 'relu'
        self.weight_init = 'random'
        
        # 训练配置
        self.epochs = 100 
        self.batch_size = 64
        self.learning_rate = 0.01
        
        # 损失函数和优化器
        self.loss_function = 'mse'  # 默认：mse
        self.optimizer = 'sgd'      # 默认：sgd
        self.regularization = 'l2'  # 默认：l2正则化
        
        # 优化器参数
        self.momentum = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.weight_decay = 0.01
        
        # 正则化参数
        self.lambda_reg = 0.01      # L1/L2正则化系数
        self.dropout_rate = 0.1     # Dropout比例
        
        # 数据增强参数
        self.rotation_range = 5
        self.translation_range = 2
        self.noise_std = 0.01
        self.scale_range = 0.05
        
        # 数据配置
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.shuffle_data = True
        
        # 输出配置
        self.save_model = True
        self.model_dir = 'models'  # 'models'目录
        self.log_dir = 'logs'
        
        # 随机种子
        self.random_seed = 42
        
        # 验证配置
        self.verbose = True
        self.print_interval = 10  # 每10个epoch打印一次


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST神经网络训练')
    
    # 网络结构参数
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh', 'sigmoid'],
                       help='激活函数类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,  
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01,
                       help='学习率')
    
    # 损失函数和优化器
    parser.add_argument('--loss', '--loss_function', type=str, default='cross_entropy',
                       choices=['cross_entropy', 'mse', 'l1_loss'],
                       help='损失函数类型')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'momentum', 'adam', 'adamw', 'adagrad'],
                       help='优化器类型')
    parser.add_argument('--regularization', type=str, default='l2',
                       choices=['none', 'l1', 'l2', 'dropout', 'data_augmentation'],
                       help='正则化方法')
    
    # 优化器参数
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='动量系数（用于momentum优化器）')
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='Adam优化器的beta1参数')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Adam优化器的beta2参数')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减系数（用于AdamW）')
    
    # 正则化参数
    parser.add_argument('--lambda_reg', type=float, default=0.01,
                       help='正则化系数')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout比例')
    
    # 数据增强参数
    parser.add_argument('--rotation_range', type=float, default=5,
                       help='旋转角度范围（度）')
    parser.add_argument('--translation_range', type=int, default=2,
                       help='平移像素范围')
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='高斯噪声标准差')
    parser.add_argument('--scale_range', type=float, default=0.05,
                       help='缩放范围')
    
    # 数据配置
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--no_shuffle', dest='shuffle_data', action='store_false',
                       help='不随机打乱数据')
    
    # 输出配置
    parser.add_argument('--no_save', dest='save_model', action='store_false',
                       help='不保存模型')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志保存目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='详细输出')
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                       help='静默模式')
    parser.add_argument('--print_interval', type=int, default=10,
                       help='打印间隔（epoch）')
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    return parser.parse_args()


def load_config_from_file(config_path):
    """从配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = Config()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"警告: 配置文件中的未知参数: {key}")
        
        print(f"从 {config_path} 加载配置")
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在")
        return None
    except json.JSONDecodeError:
        print(f"配置文件 {config_path} 格式错误")
        return None


def save_config_to_file(config, config_path):
    """保存配置到文件"""
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            config_dict[attr] = getattr(config, attr)
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {config_path}")


def args_to_config(args):
    """将命令行参数转换为配置对象"""
    config = Config()
    
    # 网络结构
    # 固定网络层数，不从命令行读取
    config.activation = args.activation
    
    # 训练参数
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # 损失函数和优化器
    config.loss_function = args.loss
    config.optimizer = args.optimizer
    config.regularization = args.regularization
    
    # 优化器参数
    config.momentum = args.momentum
    config.beta1 = args.beta1
    config.beta2 = args.beta2
    config.weight_decay = args.weight_decay
    
    # 正则化参数
    config.lambda_reg = args.lambda_reg
    config.dropout_rate = args.dropout_rate
    
    # 数据增强参数
    config.rotation_range = args.rotation_range
    config.translation_range = args.translation_range
    config.noise_std = args.noise_std
    config.scale_range = args.scale_range
    
    # 数据配置
    config.train_ratio = args.train_ratio
    config.val_ratio = args.val_ratio
    config.shuffle_data = args.shuffle_data
    
    # 输出配置
    config.save_model = args.save_model
    config.model_dir = args.model_dir
    config.log_dir = args.log_dir
    
    # 其他参数
    config.random_seed = args.seed
    config.verbose = args.verbose
    config.print_interval = args.print_interval
    
    return config


def get_config():
    """获取配置（从命令行参数或配置文件）"""
    args = parse_args()
    
    # 如果指定了配置文件，优先使用配置文件
    if args.config:
        config = load_config_from_file(args.config)
        if config:
            return config
    
    # 否则使用命令行参数
    config = args_to_config(args)
    
    # 设置随机种子
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        print(f"设置随机种子: {config.random_seed}")
    
    return config


def print_config(config):
    """打印配置信息"""
    print("=" * 60)
    print("训练配置")
    print("=" * 60)
    
    # 网络结构
    print(f"网络结构: {config.layer_sizes}")
    print(f"激活函数: {config.activation}")
    print(f"权重初始化: {config.weight_init}")
    
    # 训练参数
    print(f"训练轮数: {config.epochs}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    
    # 损失函数和优化器
    print(f"损失函数: {config.loss_function}")
    print(f"优化器: {config.optimizer}")
    print(f"正则化: {config.regularization}")
    
    # 正则化参数
    if config.regularization in ['l1', 'l2']:
        print(f"正则化系数: {config.lambda_reg}")
    elif config.regularization == 'dropout':
        print(f"Dropout比例: {config.dropout_rate}")
    elif config.regularization == 'data_augmentation':
        print(f"数据增强参数: 旋转{config.rotation_range}°, 平移{config.translation_range}px, 噪声{config.noise_std}")
    
    
    # 数据配置
    print(f"训练集比例: {config.train_ratio}")
    print(f"验证集比例: {config.val_ratio}")
    
    print("=" * 60)