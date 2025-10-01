"""
神经网络模块
"""

import numpy as np


class NeuralNetwork:
    """全连接神经网络"""
    
    def __init__(self, layer_sizes, activation='relu', weight_init='random'):
        """
        初始化神经网络
        
        Args:
            layer_sizes: 各层神经元数量，例如 [784, 512, 256, 10]
            activation: 激活函数类型 ('relu', 'tanh', 'sigmoid')
            weight_init: 权重初始化方法 ('xavier', 'he', 'random')
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        self.weight_init = weight_init
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        self._initialize_weights()
        
        # 存储前向传播的中间结果（用于反向传播）
        self.activations = []
        self.z_values = []
        
        # 存储梯度
        self.weight_gradients = []
        self.bias_gradients = []
    
    def _initialize_weights(self):
        """初始化权重和偏置"""
        for i in range(self.num_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            if self.weight_init == 'xavier':
                # Xavier初始化
                limit = np.sqrt(6.0 / (input_size + output_size))
                weight = np.random.uniform(-limit, limit, (input_size, output_size))
            elif self.weight_init == 'he':
                # He初始化（适用于ReLU）
                std = np.sqrt(2.0 / input_size)
                weight = np.random.normal(0, std, (input_size, output_size))
            else:  # random -> normal(Gaussian)
                # 正态分布初始化（更简单稳定的默认）
                weight = np.random.normal(0, 0.01, (input_size, output_size))
            
            bias = np.zeros((1, output_size))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _activation_function(self, z):
        """激活函数"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # 防止溢出
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def _activation_derivative(self, z):
        """激活函数的导数"""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'sigmoid':
            s = self._activation_function(z)
            return s * (1 - s)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def _softmax(self, z):
        """Softmax激活函数（用于输出层）"""
        # 数值稳定的softmax实现
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        """
        前向传播
        
        Args:
            X: 输入数据 (batch_size, input_size)
            training: 是否在训练模式
        
        Returns:
            output: 网络输出 (batch_size, num_classes)
        """
        # 清空之前的中间结果
        self.activations = []
        self.z_values = []
        
        # 输入层
        current_input = X
        self.activations.append(current_input)
        
        # 隐藏层
        for i in range(self.num_layers - 2):  # 不包括输出层
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            a = self._activation_function(z)
            self.activations.append(a)
            current_input = a
        
        # 输出层（使用softmax）
        z = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        
        output = self._softmax(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, loss_gradient):
        """
        反向传播
        
        Args:
            X: 输入数据
            y: 真实标签
            loss_gradient: 损失函数对输出的梯度
        
        Returns:
            gradients: 权重和偏置的梯度
        """
        # 初始化梯度
        self.weight_gradients = [np.zeros_like(w) for w in self.weights]
        self.bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # 输出层的梯度
        delta = loss_gradient
        
        # 从输出层向前计算梯度
        for i in reversed(range(self.num_layers - 1)):
            # 权重梯度
            self.weight_gradients[i] = np.dot(self.activations[i].T, delta)
            
            # 偏置梯度
            self.bias_gradients[i] = np.sum(delta, axis=0, keepdims=True)
            
            # 如果不是第一层，计算前一层的delta
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i-1])
        
        return self.weight_gradients, self.bias_gradients
    
    def predict(self, X):
        """
        预测（不使用训练模式）
        
        Args:
            X: 输入数据
        
        Returns:
            predictions: 预测结果
        """
        return self.forward(X, training=False)
    
    def predict_classes(self, X):
        """
        预测类别
        
        Args:
            X: 输入数据
        
        Returns:
            classes: 预测的类别索引
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def get_weights(self):
        """获取所有权重和偏置"""
        return self.weights, self.biases
    
    def set_weights(self, weights, biases):
        """设置权重和偏置"""
        self.weights = weights
        self.biases = biases
    
    def save_weights(self, filename):
        """保存权重到文件"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)
        print(f"权重已保存到: {filename}")
    
    def load_weights(self, filename):
        """从文件加载权重"""
        import pickle
        with open(filename, 'rb') as f:
            weights, biases = pickle.load(f)
        self.set_weights(weights, biases)
        print(f"权重已从 {filename} 加载")
    
    def get_parameter_count(self):
        """获取参数数量"""
        total_params = 0
        for i in range(len(self.weights)):
            total_params += self.weights[i].size + self.biases[i].size
        return total_params
    
    def summary(self):
        """打印网络结构摘要"""
        print("=" * 50)
        print("神经网络结构摘要")
        print("=" * 50)
        print(f"层数: {self.num_layers}")
        print(f"激活函数: {self.activation}")
        print(f"权重初始化: {self.weight_init}")
        print()
        
        for i in range(self.num_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            weight_params = input_size * output_size
            bias_params = output_size
            
            layer_type = "输出层" if i == self.num_layers - 2 else f"隐藏层{i+1}"
            print(f"{layer_type}: {input_size} → {output_size}")
            print(f"  权重参数: {weight_params:,}")
            print(f"  偏置参数: {bias_params:,}")
            print(f"  总参数: {weight_params + bias_params:,}")
            print()
        
        print(f"总参数数量: {self.get_parameter_count():,}")
        print("=" * 50)


def test_neural_network():
    """测试神经网络"""
    print("测试神经网络...")
    
    # 创建网络
    layer_sizes = [784, 512, 256, 10]
    network = NeuralNetwork(layer_sizes, activation='relu', weight_init='he')
    
    # 打印网络结构
    network.summary()
    
    # 创建测试数据
    batch_size = 32
    X = np.random.randn(batch_size, 784)
    y = np.random.randint(0, 10, batch_size)
    
    print(f"\n测试数据形状:")
    print(f"  输入: {X.shape}")
    print(f"  标签: {y.shape}")
    
    # 前向传播
    print("\n前向传播测试:")
    output = network.forward(X)
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  输出和: {np.sum(output, axis=1)[:5]}")  # 应该接近1（softmax）
    
    # 预测
    print("\n预测测试:")
    predictions = network.predict_classes(X)
    print(f"  预测类别: {predictions[:10]}")
    print(f"  预测形状: {predictions.shape}")
    
    # 反向传播测试
    print("\n反向传播测试:")
    # 模拟损失梯度（交叉熵的梯度）
    batch_size = X.shape[0]
    num_classes = 10
    one_hot_y = np.zeros((batch_size, num_classes))
    one_hot_y[np.arange(batch_size), y] = 1
    
    # 交叉熵损失对输出的梯度
    loss_grad = (output - one_hot_y) / batch_size
    
    weight_grads, bias_grads = network.backward(X, y, loss_grad)
    
    print(f"  权重梯度数量: {len(weight_grads)}")
    print(f"  偏置梯度数量: {len(bias_grads)}")
    
    for i, (w_grad, b_grad) in enumerate(zip(weight_grads, bias_grads)):
        print(f"  第{i+1}层权重梯度范围: [{w_grad.min():.4f}, {w_grad.max():.4f}]")
        print(f"  第{i+1}层偏置梯度范围: [{b_grad.min():.4f}, {b_grad.max():.4f}]")


if __name__ == "__main__":
    test_neural_network()
