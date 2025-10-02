"""
损失函数模块
实现多种损失函数
"""

import numpy as np


class LossFunction:
    """损失函数基类"""
    
    def forward(self, predictions, targets):
        """
        前向传播计算损失
        
        Args:
            predictions: 预测值 (batch_size, num_classes)
            targets: 真实标签 (batch_size, num_classes) 或 (batch_size,)
        
        Returns:
            loss: 标量损失值
        """
        raise NotImplementedError
    
    def backward(self, predictions, targets):
        """
        反向传播计算梯度
        
        Args:
            predictions: 预测值 (batch_size, num_classes)
            targets: 真实标签 (batch_size, num_classes) 或 (batch_size,)
        
        Returns:
            gradients: 对预测值的梯度 (batch_size, num_classes)
        """
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):
    """交叉熵损失函数"""
    
    def __init__(self, epsilon=1e-15):
        """
        Args:
            epsilon: 防止log(0)的小常数
        """
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        """
        计算交叉熵损失
        注意：predictions应该是经过softmax的输出
        
        Args:
            predictions: 预测值 (batch_size, num_classes) - 应该是softmax后的结果
            targets: 真实标签，可以是one-hot编码或类别索引
        
        Returns:
            loss: 平均交叉熵损失
        """
        # 防止log(0)
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # 如果targets是类别索引，转换为one-hot
        if targets.ndim == 1:
            batch_size = len(targets)
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((batch_size, num_classes))
            one_hot_targets[np.arange(batch_size), targets] = 1
            targets = one_hot_targets
        
        # 计算交叉熵损失
        loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
        
        return loss
    
    def backward(self, predictions, targets):
        """
        计算交叉熵损失的梯度
        注意：predictions应该是经过softmax的输出
        
        Returns:
            gradients: 对预测值的梯度
        """
        # 如果targets是类别索引，转换为one-hot
        if targets.ndim == 1:
            batch_size = len(targets)
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((batch_size, num_classes))
            one_hot_targets[np.arange(batch_size), targets] = 1
            targets = one_hot_targets
        
        # 交叉熵损失对softmax输入的梯度
        gradients = (predictions - targets) / len(targets)
        
        return gradients


class MSELoss(LossFunction):
    """均方误差损失函数"""
    
    def forward(self, predictions, targets):
        """
        计算均方误差损失
        
        Args:
            predictions: 预测值 (batch_size, num_classes)
            targets: 真实标签，可以是one-hot编码或类别索引
        
        Returns:
            loss: 平均均方误差损失
        """
        # 如果targets是类别索引，转换为one-hot
        if targets.ndim == 1:
            batch_size = len(targets)
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((batch_size, num_classes))
            one_hot_targets[np.arange(batch_size), targets] = 1
            targets = one_hot_targets
        
        # 计算均方误差
        loss = np.mean((predictions - targets) ** 2)
        
        return loss
    
    def backward(self, predictions, targets):
        """
        计算均方误差损失的梯度
        
        Returns:
            gradients: 对预测值的梯度
        """
        # 如果targets是类别索引，转换为one-hot
        if targets.ndim == 1:
            batch_size = len(targets)
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((batch_size, num_classes))
            one_hot_targets[np.arange(batch_size), targets] = 1
            targets = one_hot_targets
        
        # 均方误差的梯度
        gradients = 2 * (predictions - targets) / len(targets)
        
        return gradients


class L1Loss(LossFunction):
    """绝对值损失函数（L1损失）"""
    
    def forward(self, predictions, targets):
        """
        计算绝对值损失
        
        Args:
            predictions: 预测值 (batch_size, num_classes)
            targets: 真实标签，可以是one-hot编码或类别索引
        
        Returns:
            loss: 平均绝对值损失
        """
        # 如果targets是类别索引，转换为one-hot
        if targets.ndim == 1:
            batch_size = len(targets)
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((batch_size, num_classes))
            one_hot_targets[np.arange(batch_size), targets] = 1
            targets = one_hot_targets
        
        # 计算绝对值损失
        loss = np.mean(np.abs(predictions - targets))
        
        return loss
    
    def backward(self, predictions, targets):
        """
        计算绝对值损失的梯度
        
        Returns:
            gradients: 对预测值的梯度
        """
        # 如果targets是类别索引，转换为one-hot
        if targets.ndim == 1:
            batch_size = len(targets)
            num_classes = predictions.shape[1]
            one_hot_targets = np.zeros((batch_size, num_classes))
            one_hot_targets[np.arange(batch_size), targets] = 1
            targets = one_hot_targets
        
        # 绝对值损失的梯度（符号函数）
        gradients = np.sign(predictions - targets) / len(targets)
        
        return gradients


def get_loss_function(loss_name):
    """
    根据名称获取损失函数
    
    Args:
        loss_name: 损失函数名称 ('cross_entropy', 'mse', 'l1_loss')
    
    Returns:
        loss_function: 损失函数实例
    """
    loss_functions = {
        'cross_entropy': CrossEntropyLoss(),
        'mse': MSELoss(),
        'l1_loss': L1Loss()
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"不支持的损失函数: {loss_name}. 支持的损失函数: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name]