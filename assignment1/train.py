"""
训练模块
实现神经网络的训练流程
"""

import os
import time
import numpy as np
from network import NeuralNetwork
from loss_function import get_loss_function
from optimizer import get_optimizer
from regulation import get_regularization
from preprocess import preprocess_mnist, create_batches
from evaluate import evaluate_model, print_evaluation_results
from visualize import plot_training_history
from config import get_config, print_config


class Trainer:
    """训练器"""
    
    def __init__(self, config):
        """
        Args:
            config: 训练配置
        """
        self.config = config
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.regularization = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # 创建必要的目录
        os.makedirs('models', exist_ok=True)
    
    def _initialize_components(self):
        """初始化网络组件"""
        # 初始化网络
        self.network = NeuralNetwork(
            layer_sizes=self.config.layer_sizes,
            activation=self.config.activation,
            weight_init=self.config.weight_init
        )
            
        # 初始化损失函数
        self.loss_function = get_loss_function(self.config.loss_function)
            
        # 初始化优化器
        optimizer_kwargs = {}
        if self.config.optimizer == 'momentum':
            optimizer_kwargs['momentum'] = self.config.momentum
        elif self.config.optimizer in ['adam', 'adamw']:
            optimizer_kwargs['beta1'] = self.config.beta1
            optimizer_kwargs['beta2'] = self.config.beta2
            optimizer_kwargs['epsilon'] = self.config.epsilon
            if self.config.optimizer == 'adamw':
                optimizer_kwargs['weight_decay'] = self.config.weight_decay
            
        self.optimizer = get_optimizer(
            self.config.optimizer,
            learning_rate=self.config.learning_rate,
            **optimizer_kwargs
        )
            
        # 初始化正则化
        self.regularization = get_regularization(
            self.config.regularization,
            lambda_reg=self.config.lambda_reg,
            dropout_rate=self.config.dropout_rate,
            rotation_range=self.config.rotation_range,
            translation_range=self.config.translation_range,
            noise_std=self.config.noise_std,
            scale_range=self.config.scale_range
        )
    
    def _prepare_data(self):
        """准备数据"""
        print("准备数据...")
        
        # 预处理数据
        data = preprocess_mnist(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            normalize=True,
            one_hot=True
        )
        
        return data
    
    def _train_epoch(self, train_data, train_labels, epoch):
        """训练一个epoch"""
        # 创建batches
        batches = create_batches(
            train_data, train_labels, 
            self.config.batch_size, 
            shuffle=self.config.shuffle_data
        )
        
        epoch_loss = 0
        epoch_accuracy = 0
        
        for batch_data, batch_labels in batches:
            # 数据增强（如果使用）
            if self.config.regularization == 'data_augmentation':
                batch_data = self.regularization.apply(batch_data, training=True)
            
            # 前向传播
            predictions = self.network.forward(batch_data, training=True)
            
            # 计算损失（使用原始预测结果）
            loss = self.loss_function.forward(predictions, batch_labels)
            
            # 计算准确率（使用原始预测结果）
            acc = self._calculate_accuracy(predictions, batch_labels)
            
            # 反向传播（仅在需要梯度时应用Dropout）
            loss_gradient = self.loss_function.backward(predictions, batch_labels)
            
            # 应用Dropout梯度（如果使用）
            if self.config.regularization == 'dropout':
                loss_gradient = self.regularization.gradient(loss_gradient)
            
            weight_gradients, bias_gradients = self.network.backward(
                batch_data, batch_labels, loss_gradient
            )
            
            # 应用正则化梯度
            if self.config.regularization in ['l1', 'l2']:
                for i in range(len(weight_gradients)):
                    weight_gradients[i] += self.regularization.gradient(
                        self.network.weights[i]
                    )
            
            # 更新权重
            for i in range(len(self.network.weights)):
                self.network.weights[i] = self.optimizer.update(
                    self.network.weights[i], weight_gradients[i]
                )
                self.network.biases[i] = self.optimizer.update(
                    self.network.biases[i], bias_gradients[i]
                )
            
            epoch_loss += loss
            epoch_accuracy += acc
        
        # 计算平均指标
        avg_loss = epoch_loss / len(batches)
        avg_accuracy = epoch_accuracy / len(batches)
        
        return avg_loss, avg_accuracy
    
    def _calculate_accuracy(self, predictions, targets):
        """计算准确率"""
        if predictions.ndim > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        if targets.ndim > 1:
            true_classes = np.argmax(targets, axis=1)
        else:
            true_classes = targets
        
        correct = np.sum(pred_classes == true_classes)
        return correct / len(pred_classes)
    
    def _validate(self, val_data, val_labels):
        """验证"""
        # 评估模型
        results = evaluate_model(
            self.network, val_data, val_labels,
            self.config.loss_function, self.config.batch_size
        )
        
        return results['loss'], results['accuracy']
    
    def train(self):
        """训练模型"""
        print("开始训练...")
        print_config(self.config)
        
        # 初始化组件
        self._initialize_components()
        
        # 准备数据
        data = self._prepare_data()
        
        train_data = data['train_images']
        train_labels = data['train_labels']
        val_data = data['val_images']
        val_labels = data['val_labels']
        
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        print(f"网络参数数量: {self.network.get_parameter_count():,}")
        
        # 组合保存命名前缀：优化器_损失_正则化
        name_prefix = f"{self.config.optimizer}_{self.config.loss_function}_{self.config.regularization}"
        
        # 创建pictures目录
        os.makedirs('train_results', exist_ok=True)
        
        # 学习率调度已移除
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(train_data, train_labels, epoch)
            
            # 验证
            val_loss, val_acc = self._validate(val_data, val_labels)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            
            # 打印进度
            if self.config.verbose and (epoch + 1) % self.config.print_interval == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1:3d}/{self.config.epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                      f"time={epoch_time:.2f}s")
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成！总时间: {total_time:.2f}s")
        
        # 最终验证
        final_results = evaluate_model(
            self.network, val_data, val_labels,
            self.config.loss_function, self.config.batch_size
        )
        print_evaluation_results(final_results, "验证集")
        
        # 添加训练集评估结果
        train_results = evaluate_model(
            self.network, train_data, train_labels,
            self.config.loss_function, self.config.batch_size
        )
        
        # 合并训练和验证结果，以便在可视化中显示
        combined_results = {
            'train_loss': train_results['loss'],
            'val_loss': final_results['loss'],
            'train_accuracy': train_results['accuracy'],
            'val_accuracy': final_results['accuracy'],
            'precision': final_results['precision']  # 使用验证集的precision
        }
        
        # 训练历史可视化并保存
        history_path = os.path.join('train_results', f"{name_prefix}_training_history.png")
        plot_training_history(
            self.train_losses,
            self.val_losses,
            self.train_accuracies,
            self.val_accuracies,
            final_results=combined_results,
            save_path=history_path
        )
        
        # 保存最终模型
        if self.config.save_model:
            final_model_path = os.path.join(
                'models',
                f"{name_prefix}_final.npy"
            )
            self.network.save_weights(final_model_path)
        
        return {
            'network': self.network,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'final_val_results': final_results
        }


def main():
    """主函数"""
    # 获取配置
    config = get_config()
    
    # 设置随机种子
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    main()
