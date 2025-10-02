"""
数据预处理模块
负责MNIST数据的预处理和数据集划分
"""

import numpy as np
from download import load_mnist_data


def normalize_data(images):
    """
    数据归一化：将像素值从[0,255]归一化到[0,1]
    """                              
    return images.astype(np.float32) / 255.0

 
def reshape_data(images):                                                        
    """
    数据重塑：将28x28的图像重塑为784维向量
    """
    return images.reshape(images.shape[0], -1)                                                                                                                                                                                 


def one_hot_encode(labels, num_classes=10):
    """
    One-hot编码标签
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def split_dataset(images, labels, train_ratio=0.8, val_ratio=0.2):
    """
    划分训练集和验证集
    """
    total_samples = len(images)
    
    # 计算划分点
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    # 随机打乱数据
    indices = np.random.permutation(total_samples)
    
    # 划分数据
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]
    
    return train_images, train_labels, val_images, val_labels


def create_batches(data, labels, batch_size, shuffle=True):
    """
    创建mini-batch数据
    """
    num_samples = len(data)
    
    if shuffle:
        indices = np.random.permutation(num_samples)
        data = data[indices]
        labels = labels[indices]
    
    batches = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_data = data[i:end_idx]
        batch_labels = labels[i:end_idx]
        batches.append((batch_data, batch_labels))
    
    return batches


def preprocess_mnist(train_ratio=0.8, val_ratio=0.2, normalize=True, one_hot=True):
    """
    完整的MNIST数据预处理流程
    
    Args:
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        normalize: 是否归一化
        one_hot: 是否使用one-hot编码
    
    Returns:
        dict: 包含所有预处理后数据的字典
    """
    # 构建缓存文件名，包含参数信息以确保一致性
    cache_filename = f"data/processed_mnist_t{train_ratio}_v{val_ratio}_n{normalize}_o{one_hot}.pkl"
    
    # 尝试从缓存加载预处理数据
    cached_data = load_processed_data(cache_filename)
    if cached_data is not None:
        print("使用缓存的预处理数据")
        return cached_data
    
    print("开始预处理MNIST数据...")
    
    # 加载原始数据
    raw_data = load_mnist_data()
    if raw_data is None:
        raise ValueError("无法加载MNIST数据")
    
    train_images = raw_data['train_images']
    train_labels = raw_data['train_labels']
    test_images = raw_data['test_images']
    test_labels = raw_data['test_labels']
    
    print(f"原始数据形状:")
    print(f"  训练图像: {train_images.shape}")
    print(f"  训练标签: {train_labels.shape}")
    print(f"  测试图像: {test_images.shape}")
    print(f"  测试标签: {test_labels.shape}")
    
    # 数据归一化
    if normalize:
        print("正在归一化数据...")
        train_images = normalize_data(train_images)
        test_images = normalize_data(test_images)
    
    # 数据重塑
    print("正在重塑数据...")
    train_images = reshape_data(train_images)
    test_images = reshape_data(test_images)
    
    # 划分训练集和验证集
    print("正在划分训练集和验证集...")
    train_images, train_labels, val_images, val_labels = split_dataset(
        train_images, train_labels, train_ratio, val_ratio
    )
    
    # One-hot编码
    if one_hot:
        print("正在进行One-hot编码...")
        train_labels = one_hot_encode(train_labels)
        val_labels = one_hot_encode(val_labels)
        test_labels = one_hot_encode(test_labels)
    
    # 保存预处理后的数据
    processed_data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'input_size': 784,  # 28*28
        'num_classes': 10,
        'class_names': [str(i) for i in range(10)]
    }
    
    print("数据预处理完成!")
    print(f"预处理后数据形状:")
    print(f"  训练集: {train_images.shape}")
    print(f"  验证集: {val_images.shape}")
    print(f"  测试集: {test_images.shape}")
    print(f"  输入维度: {processed_data['input_size']}")
    print(f"  类别数量: {processed_data['num_classes']}")
    
    # 保存预处理数据到缓存
    save_processed_data(processed_data, cache_filename)
    
    return processed_data


def get_data_info(data):
    """
    获取数据集信息
    """
    info = {
        'train_samples': len(data['train_images']),
        'val_samples': len(data['val_images']),
        'test_samples': len(data['test_images']),
        'input_size': data['input_size'],
        'num_classes': data['num_classes'],
        'train_image_shape': data['train_images'].shape,
        'train_label_shape': data['train_labels'].shape
    }
    return info


def save_processed_data(data, filename='data/processed_mnist.pkl'):
    """
    保存预处理后的数据到缓存文件
    
    Args:
        data: 预处理后的数据
        filename: 缓存文件路径
    """
    import pickle
    import os
    
    # 创建目录
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"预处理数据已保存到: {filename}")


def load_processed_data(filename='data/processed_mnist.pkl'):
    """
    加载预处理后的数据
    
    Args:
        filename: 缓存文件路径
    
    Returns:
        dict: 预处理后的数据，如果文件不存在则返回None
    """
    import pickle
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"从 {filename} 加载预处理数据")
        return data
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，将重新预处理数据")
        return None