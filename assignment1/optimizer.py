"""
优化器模块
实现多种优化算法
"""

import numpy as np


class Optimizer:
    """优化器基类"""
    
    def __init__(self, learning_rate):
        """
        Args:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
    
    def update(self, weights, gradients):
        """
        更新权重
        
        Args:
            weights: 当前权重
            gradients: 梯度
        
        Returns:
            updated_weights: 更新后的权重
        """
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降优化器"""
    
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, weights, gradients):
        """
        标准SGD更新规则
        w = w - lr * gradient
        """
        return weights - self.learning_rate * gradients


class MomentumSGD(Optimizer):
    """带动量的随机梯度下降"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Args:
            learning_rate: 学习率
            momentum: 动量系数
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}  # 为每层权重单独维护velocity
    
    def update(self, weights, gradients, layer_key=None):
        """
        动量SGD更新规则
        v = momentum * v - lr * gradient
        w = w + v
        
        Args:
            weights: 权重矩阵
            gradients: 梯度矩阵
            layer_key: 层的标识符（索引或名称），用于区分不同层的状态
        """
        # 使用层标识符作为键，如果未提供则使用权重形状
        if layer_key is None:
            weight_key = weights.shape
        else:
            weight_key = layer_key
        
        # 检查是否已有对应的状态变量，如果没有则创建
        if weight_key not in self.velocities:
            try:
                # 使用与权重相同形状和数据类型的数组初始化velocity，避免内存分配问题
                self.velocities[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
            except np.core._exceptions._ArrayMemoryError as e:
                # 处理内存分配失败的情况
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储velocity状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    # 如果float64分配失败，尝试使用float32
                    self.velocities[weight_key] = np.zeros_like(weights, dtype=np.float32)
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e  # 重新抛出原始异常
            except Exception as e:
                # 处理其他可能的异常
                print(f"创建velocity状态时发生未知错误: {e}")
                raise e
        
        # 检查现有velocity的形状是否与当前权重匹配
        elif self.velocities[weight_key].shape != weights.shape:
            # 如果形状不匹配，重新创建velocity
            try:
                self.velocities[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储velocity状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    self.velocities[weight_key] = np.zeros_like(weights, dtype=np.float32)
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e
        
        # 更新速度: v = momentum * v - lr * gradient
        self.velocities[weight_key] = self.momentum * self.velocities[weight_key] - self.learning_rate * gradients
        
        # 更新权重: w = w + v
        return weights + self.velocities[weight_key]


class AdaGrad(Optimizer):
    """AdaGrad自适应学习率优化器"""
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        Args:
            learning_rate: 学习率
            epsilon: 防止除零的小常数
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.caches = {}  # 为每层权重单独维护缓存
    
    def update(self, weights, gradients, layer_key=None):
        """
        AdaGrad更新规则
        cache = cache + gradient^2
        w = w - lr * gradient / sqrt(cache + epsilon)
        
        Args:
            weights: 权重矩阵
            gradients: 梯度矩阵
            layer_key: 层的标识符（索引或名称），用于区分不同层的状态
        """
        # 使用层标识符作为键，如果未提供则使用权重形状
        if layer_key is None:
            weight_key = weights.shape
        else:
            weight_key = layer_key
        
        if weight_key not in self.caches:
            # 使用与权重相同形状和数据类型的数组初始化cache，避免内存分配问题
            try:
                self.caches[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储cache状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    self.caches[weight_key] = np.zeros_like(weights, dtype=np.float32)
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e
            except Exception as e:
                print(f"创建cache状态时发生未知错误: {e}")
                raise e
        
        # 检查现有cache的形状是否与当前权重匹配
        elif self.caches[weight_key].shape != weights.shape:
            # 如果形状不匹配，重新创建cache
            try:
                self.caches[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储cache状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    self.caches[weight_key] = np.zeros_like(weights, dtype=np.float32)
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e
        
        # 更新缓存: cache = cache + gradient^2
        self.caches[weight_key] += gradients ** 2
        
        # 更新权重: w = w - lr * gradient / sqrt(cache + epsilon)
        return weights - self.learning_rate * gradients / (np.sqrt(self.caches[weight_key]) + self.epsilon)


class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 防止除零的小常数
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = {}  # 一阶矩估计
        self.second_moments = {}  # 二阶矩估计
        self.timesteps = {}  # 时间步
    
    def update(self, weights, gradients, layer_key=None):
        """
        Adam更新规则
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        w = w - lr * m_hat / sqrt(v_hat + epsilon)
        
        Args:
            weights: 权重矩阵
            gradients: 梯度矩阵
            layer_key: 层的标识符（索引或名称），用于区分不同层的状态
        """
        # 使用层标识符作为键，如果未提供则使用权重形状
        if layer_key is None:
            weight_key = weights.shape
        else:
            weight_key = layer_key
        
        # 初始化该层的状态
        if weight_key not in self.moments:
            # 使用与权重相同形状和数据类型的数组初始化状态变量，避免内存分配问题
            try:
                self.moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.second_moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.timesteps[weight_key] = 0
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    self.moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.second_moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.timesteps[weight_key] = 0
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e
            except Exception as e:
                print(f"创建状态变量时发生未知错误: {e}")
                raise e
        
        # 检查现有状态变量的形状是否与当前权重匹配
        elif (self.moments[weight_key].shape != weights.shape or 
              self.second_moments[weight_key].shape != weights.shape):
            # 如果形状不匹配，重新创建状态变量
            try:
                self.moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.second_moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.timesteps[weight_key] = 0
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    self.moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.second_moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.timesteps[weight_key] = 0
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e
        
        # 更新时间步
        self.timesteps[weight_key] += 1
        
        # 更新一阶和二阶矩估计
        self.moments[weight_key] = self.beta1 * self.moments[weight_key] + (1 - self.beta1) * gradients
        self.second_moments[weight_key] = self.beta2 * self.second_moments[weight_key] + (1 - self.beta2) * (gradients ** 2)
        
        # 偏差修正
        m_hat = self.moments[weight_key] / (1 - self.beta1 ** self.timesteps[weight_key])
        v_hat = self.second_moments[weight_key] / (1 - self.beta2 ** self.timesteps[weight_key])
        
        # 更新权重
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdamW(Optimizer):
    """AdamW优化器（带权重衰减的Adam）"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        """
        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 防止除零的小常数
            weight_decay: 权重衰减系数
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.moments = {}  # 一阶矩估计
        self.second_moments = {}  # 二阶矩估计
        self.timesteps = {}  # 时间步
    
    def update(self, weights, gradients, layer_key=None):
        """
        AdamW更新规则
        w = w - lr * weight_decay * w  # 权重衰减
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        w = w - lr * m_hat / sqrt(v_hat + epsilon)
        
        Args:
            weights: 权重矩阵
            gradients: 梯度矩阵
            layer_key: 层的标识符（索引或名称），用于区分不同层的状态
        """
        # 使用层标识符作为键，如果未提供则使用权重形状
        if layer_key is None:
            weight_key = weights.shape
        else:
            weight_key = layer_key
        
        # 初始化该层的状态
        if weight_key not in self.moments:
            try:
                # 使用与权重相同形状和数据类型的数组初始化状态变量，避免内存分配问题
                self.moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.second_moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.timesteps[weight_key] = 0
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    self.moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.second_moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.timesteps[weight_key] = 0
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e
            except Exception as e:
                print(f"创建状态变量时发生未知错误: {e}")
                raise e
        
        # 检查现有状态变量的形状是否与当前权重匹配
        elif (self.moments[weight_key].shape != weights.shape or 
              self.second_moments[weight_key].shape != weights.shape):
            # 如果形状不匹配，重新创建状态变量
            try:
                self.moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.second_moments[weight_key] = np.zeros_like(weights, dtype=weights.dtype)
                self.timesteps[weight_key] = 0
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"警告: 无法为形状为{weights.shape}的权重分配内存来存储状态: {e}")
                print("尝试使用更小的数据类型...")
                try:
                    self.moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.second_moments[weight_key] = np.zeros_like(weights, dtype=np.float32)
                    self.timesteps[weight_key] = 0
                except np.core._exceptions._ArrayMemoryError as e2:
                    print(f"错误: 即使使用float32也无法分配所需内存: {e2}")
                    raise e
        
        # 权重衰减
        weights = weights - self.learning_rate * self.weight_decay * weights
        
        # 更新时间步
        self.timesteps[weight_key] += 1
        
        # 更新一阶和二阶矩估计
        self.moments[weight_key] = self.beta1 * self.moments[weight_key] + (1 - self.beta1) * gradients
        self.second_moments[weight_key] = self.beta2 * self.second_moments[weight_key] + (1 - self.beta2) * (gradients ** 2)
        
        # 偏差修正
        m_hat = self.moments[weight_key] / (1 - self.beta1 ** self.timesteps[weight_key])
        v_hat = self.second_moments[weight_key] / (1 - self.beta2 ** self.timesteps[weight_key])
        
        # 更新权重
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def get_optimizer(optimizer_name, learning_rate=0.01, **kwargs):
    """
    根据名称获取优化器
    
    Args:
        optimizer_name: 优化器名称
        learning_rate: 学习率
        **kwargs: 其他优化器参数
    
    Returns:
        optimizer: 优化器实例
    """
    # 根据优化器类型过滤参数
    if optimizer_name == 'sgd':
        # SGD只接受learning_rate参数
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == 'momentum':
        # MomentumSGD接受learning_rate和momentum参数
        momentum = kwargs.get('momentum', 0.9)
        optimizer = MomentumSGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'adam':
        # Adam接受learning_rate, beta1, beta2, epsilon参数
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)
        optimizer = Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    elif optimizer_name == 'adamw':
        # AdamW接受learning_rate, beta1, beta2, epsilon, weight_decay参数
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)
        weight_decay = kwargs.get('weight_decay', 0.01)
        optimizer = AdamW(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        # AdaGrad接受learning_rate和epsilon参数
        epsilon = kwargs.get('epsilon', 1e-8)
        optimizer = AdaGrad(learning_rate=learning_rate, epsilon=epsilon)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}. 支持的优化器: ['sgd', 'momentum', 'adam', 'adamw', 'adagrad']")
    
    return optimizer


def test_optimizers():
    """测试优化器"""
    print("测试优化器...")
    
    # 创建测试数据
    weights = np.random.randn(10, 5)
    gradients = np.random.randn(10, 5)
    
    # 测试所有优化器
    optimizer_names = ['sgd', 'momentum', 'adam', 'adamw', 'adagrad']
    
    for opt_name in optimizer_names:
        print(f"\n测试 {opt_name}:")
        
        try:
            if opt_name == 'momentum':
                optimizer = get_optimizer(opt_name, learning_rate=0.01, momentum=0.9)
            elif opt_name in ['adam', 'adamw']:
                optimizer = get_optimizer(opt_name, learning_rate=0.001, weight_decay=0.01)
            else:
                optimizer = get_optimizer(opt_name, learning_rate=0.01)
            
            # 测试更新
            updated_weights = optimizer.update(weights.copy(), gradients)
            
            print(f"  原始权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"  更新后权重范围: [{updated_weights.min():.4f}, {updated_weights.max():.4f}]")
            print(f"  权重变化量: {np.mean(np.abs(updated_weights - weights)):.4f}")
            
        except Exception as e:
            print(f"  测试失败: {e}")


if __name__ == "__main__":
    test_optimizers()
