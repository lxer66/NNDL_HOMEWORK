"""
正则化模块
实现多种正则化方法
"""

import numpy as np


class Regularization:
    """正则化基类"""
    
    def apply(self, weights):
        """
        应用正则化到权重
        
        Args:
            weights: 权重矩阵
        
        Returns:
            regularized_weights: 正则化后的权重
        """
        raise NotImplementedError
    
    def gradient(self, weights):
        """
        计算正则化项的梯度
        
        Args:
            weights: 权重矩阵
        
        Returns:
            regularization_gradient: 正则化梯度
        """
        raise NotImplementedError


class L1Regularization(Regularization):
    """L1正则化（Lasso）"""
    
    def __init__(self, lambda_reg=0.01):
        """
        Args:
            lambda_reg: L1正则化系数
        """
        self.lambda_reg = lambda_reg
    
    def apply(self, weights):
        """
        L1正则化不直接修改权重，而是通过梯度影响更新
        """
        return weights
    
    def gradient(self, weights):
        """
        计算L1正则化的梯度
        gradient = lambda * sign(weights)
        """
        return self.lambda_reg * np.sign(weights)
    
    def penalty(self, weights):
        """
        计算L1正则化惩罚项
        """
        return self.lambda_reg * np.sum(np.abs(weights))


class L2Regularization(Regularization):
    """L2正则化（Ridge）"""
    
    def __init__(self, lambda_reg=0.01):
        """
        Args:
            lambda_reg: L2正则化系数
        """
        self.lambda_reg = lambda_reg
    
    def apply(self, weights):
        """
        L2正则化不直接修改权重，而是通过梯度影响更新
        """
        return weights
    
    def gradient(self, weights):
        """
        计算L2正则化的梯度
        gradient = lambda * weights
        """
        return self.lambda_reg * weights
    
    def penalty(self, weights):
        """
        计算L2正则化惩罚项
        """
        return self.lambda_reg * 0.5 * np.sum(weights ** 2)


class Dropout(Regularization):
    """Dropout正则化"""
    
    def __init__(self, dropout_rate=0.5):
        """
        Args:
            dropout_rate: Dropout比例
        """
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def apply(self, activations, training=True):
        """
        应用Dropout到激活值
        
        Args:
            activations: 激活值
            training: 是否在训练模式
        
        Returns:
            dropped_activations: Dropout后的激活值
        """
        if not training:
            return activations
        
        # 生成随机掩码
        self.mask = np.random.random(activations.shape) > self.dropout_rate
        
        # 应用掩码并缩放
        dropped_activations = activations * self.mask / (1 - self.dropout_rate)
        
        return dropped_activations
    
    def gradient(self, gradients):
        """
        计算Dropout的梯度（反向传播时使用）
        
        Args:
            gradients: 反向传播的梯度
        
        Returns:
            masked_gradients: 掩码后的梯度
        """
        if self.mask is None:
            return gradients
        
        return gradients * self.mask / (1 - self.dropout_rate)


class DataAugmentation(Regularization):
    """数据增强"""
    
    def __init__(self, rotation_range=5, translation_range=2, noise_std=0.01, scale_range=0.05):
        """
        Args:
            rotation_range: 旋转角度范围（度）
            translation_range: 平移像素范围
            noise_std: 高斯噪声标准差
            scale_range: 缩放范围
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def apply(self, images, training=True):
        """
        应用数据增强到图像
        优化版本：移除耗时的几何变换，只保留轻微平移和适量噪声
        
        Args:
            images: 输入图像 (batch_size, 784)
            training: 是否在训练模式
        
        Returns:
            augmented_images: 增强后的图像
        """
        if not training:
            return images
        
        augmented_images = images.copy()
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # 重塑为28x28图像
            img = images[i].reshape(28, 28)
            
            # 轻微平移（仅在-1到1像素之间平移）
            if self.translation_range > 0:
                dx = np.random.randint(-1, 2)
                dy = np.random.randint(-1, 2)
                img = self._translate_image(img, dx, dy)
            
            # 添加少量高斯噪声
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std/2, img.shape)  # 减少噪声强度
                img = img + noise
            
            # 确保像素值在[0,1]范围内
            img = np.clip(img, 0, 1)
            
            # 重新展平
            augmented_images[i] = img.reshape(-1)
        
        return augmented_images
    
    def _rotate_image(self, img, angle):
        """旋转图像"""
        return img  # 移除旋转以提高速度
    
    def _translate_image(self, img, dx, dy):
        """平移图像"""
        # 简化的平移实现
        h, w = img.shape
        result = np.zeros_like(img)
        
        # 计算源区域和目标区域
        src_y1 = max(0, -dy)
        src_y2 = min(h, h - dy)
        src_x1 = max(0, -dx)
        src_x2 = min(w, w - dx)
        
        dst_y1 = max(0, dy)
        dst_y2 = min(h, h + dy)
        dst_x1 = max(0, dx)
        dst_x2 = min(w, w + dx)
        
        if src_y2 > src_y1 and src_x2 > src_x1 and dst_y2 > dst_y1 and dst_x2 > dst_x1:
            result[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        
        return result


def get_regularization(reg_name, **kwargs):
    """
    根据名称获取正则化方法
    
    Args:
        reg_name: 正则化名称
        **kwargs: 正则化参数
    
    Returns:
        regularization: 正则化实例
    """
    # 过滤掉None值的参数
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if reg_name == 'none' or reg_name is None:
        return None
    elif reg_name == 'l1':
        return L1Regularization(lambda_reg=filtered_kwargs.get('lambda_reg', 0.01))
    elif reg_name == 'l2':
        return L2Regularization(lambda_reg=filtered_kwargs.get('lambda_reg', 0.01))
    elif reg_name == 'dropout':
        return Dropout(dropout_rate=filtered_kwargs.get('dropout_rate', 0.5))
    elif reg_name == 'data_augmentation':
        return DataAugmentation(
            rotation_range=filtered_kwargs.get('rotation_range', 5),
            translation_range=filtered_kwargs.get('translation_range', 2),
            noise_std=filtered_kwargs.get('noise_std', 0.01),
            scale_range=filtered_kwargs.get('scale_range', 0.05)
        )
    else:
        raise ValueError(f"不支持的正则化方法: {reg_name}. 支持的方法: ['none', 'l1', 'l2', 'dropout', 'data_augmentation']")