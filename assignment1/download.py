"""
MNIST数据集下载
"""

import os
import urllib.request
import gzip
import numpy as np
import pickle


def download_mnist():
    """
    下载MNIST数据集
    """
    # 创建数据目录
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # MNIST数据集URL - 提供多个备用源
    url_sources = {
        'train_images': [
            'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        ],
        'train_labels': [
            'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        ],
        'test_images': [
            'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        ],
        'test_labels': [
            'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ]
    }
    
    files = {}
    
    print("正在下载MNIST数据集...")
    
    for key, urls in url_sources.items():
        filename = os.path.join(data_dir, key + '.gz')
        
        # 如果文件已存在，跳过下载
        if os.path.exists(filename):
            print(f"{key} 已存在，跳过下载")
            files[key] = filename
            continue
            
        success = False
        for i, url in enumerate(urls):
            try:
                print(f"尝试下载 {key} (源 {i+1}/{len(urls)})...")
                urllib.request.urlretrieve(url, filename)
                files[key] = filename
                print(f"{key} 下载完成")
                success = True
                break
            except Exception as e:
                print(f"源 {i+1} 下载失败: {e}")
                if i == len(urls) - 1:  # 最后一个源也失败了
                    print(f"所有源都无法下载 {key}")
                    return None
    
    return files


def extract_mnist_data(files):
    """
    解压并提取MNIST数据
    """
    if not files:
        return None
    
    print("正在提取数据...")
    
    def read_images(filename):
        """读取图像数据"""
        with gzip.open(filename, 'rb') as f:
            # 跳过前16字节的头部信息
            f.read(16)
            data = f.read()
        # 确保数据是bytes类型
        if isinstance(data, str):
            data = data.encode('latin-1')
        return np.frombuffer(data, dtype=np.uint8).reshape(-1, 28, 28)
    
    def read_labels(filename):
        """读取标签数据"""
        with gzip.open(filename, 'rb') as f:
            # 跳过前8字节的头部信息
            f.read(8)
            data = f.read()
        # 确保数据是bytes类型
        if isinstance(data, str):
            data = data.encode('latin-1')
        return np.frombuffer(data, dtype=np.uint8)
    
    try:
        # 读取训练数据
        train_images = read_images(files['train_images'])
        train_labels = read_labels(files['train_labels'])
        
        # 读取测试数据
        test_images = read_images(files['test_images'])
        test_labels = read_labels(files['test_labels'])
        
        print(f"训练集: {train_images.shape[0]} 张图像")
        print(f"测试集: {test_images.shape[0]} 张图像")
        
        # 保存为pickle文件便于后续使用
        data = {
            'train_images': train_images,
            'train_labels': train_labels,
            'test_images': test_images,
            'test_labels': test_labels
        }
        
        with open('data/mnist_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print("数据提取完成，已保存为 data/mnist_data.pkl")
        return data
        
    except Exception as e:
        print(f"数据提取失败: {e}")
        return None


def load_mnist_data():
    """
    加载MNIST数据（如果已存在pickle文件则直接加载）
    """
    pickle_path = 'data/mnist_data.pkl'
    
    if os.path.exists(pickle_path):
        print("从pickle文件加载数据...")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # 如果pickle文件不存在，下载并提取数据
    print("pickle文件不存在，开始下载数据...")
    files = download_mnist()
    if files:
        return extract_mnist_data(files)
    else:
        print("数据下载失败")
        return None


def verify_data_integrity(data):
    """
    验证数据完整性
    """
    if not data:
        return False
    
    print("验证数据完整性...")
    
    # 检查数据形状
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    
    assert train_images.shape == (60000, 28, 28), f"训练图像形状错误: {train_images.shape}"
    assert train_labels.shape == (60000,), f"训练标签形状错误: {train_labels.shape}"
    assert test_images.shape == (10000, 28, 28), f"测试图像形状错误: {test_images.shape}"
    assert test_labels.shape == (10000,), f"测试标签形状错误: {test_labels.shape}"
    
    # 检查标签范围
    assert np.all((train_labels >= 0) & (train_labels <= 9)), "训练标签范围错误"
    assert np.all((test_labels >= 0) & (test_labels <= 9)), "测试标签范围错误"
    
    # 检查像素值范围
    assert np.all((train_images >= 0) & (train_images <= 255)), "训练图像像素值范围错误"
    assert np.all((test_images >= 0) & (test_images <= 255)), "测试图像像素值范围错误"
    
    print("数据完整性验证通过")
    return True


if __name__ == "__main__":
    # 测试下载和加载功能
    data = load_mnist_data()
    if data:
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        
        if verify_data_integrity(data):
            print("MNIST数据集加载成功！")
            print(f"训练集: {train_images.shape}")
            print(f"测试集: {test_images.shape}")
            print(f"标签范围: {np.unique(train_labels)}")
        else:
            print("数据验证失败")
    else:
        print("数据加载失败")
