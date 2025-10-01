# MNIST神经网络分类器

这是一个使用纯NumPy实现的全连接神经网络，用于MNIST手写数字分类。

## 功能特点

- **纯NumPy实现**：完全不使用深度学习框架，所有算法从零实现
- **模块化设计**：清晰的代码结构，便于理解和扩展
- **多种损失函数**：交叉熵、均方误差、绝对值损失
- **多种优化器**：SGD、Momentum、Adam、AdamW、AdaGrad
- **多种正则化**：L1、L2、Dropout、数据增强
- **完整评估**：准确率、精确率、混淆矩阵
- **可视化支持**：混淆矩阵、各类别准确率、错误样本展示、网络权重分布

## 文件结构

```
assignment1/
├── config.py                # 训练配置和命令行参数
├── download.py              # MNIST数据集下载
├── preprocess.py            # 数据预处理
├── network.py               # 神经网络实现
├── loss_function.py         # 损失函数实现
├── optimizer.py             # 优化器实现
├── regulation.py            # 正则化方法实现
├── train.py                 # 训练流程（主程序）
├── evaluate.py              # 评估验证
├── test.py                  # 独立测试模块
├── visualize.py             # 可视化功能
├── train.sh                 # 训练脚本
├── test.sh                  # 测试脚本
└── assignment_instructions.md # 说明文档
```

## 使用方法

### 1. 训练模型

使用默认配置训练（SGD + 交叉熵 + L2正则化）：
```bash
python train.py
```

使用自定义配置训练：
```bash
python train.py --loss cross_entropy --optimizer adam --regularization dropout --epochs 100 --lr 0.001
```

### 2. 测试模型

测试训练好的模型：
```bash
python test.py --model_path models/sgd_cross_entropy_l2_final.npy
```

测试所有模型：
```bash
bash test.sh
```

### 3. 命令行参数

#### 训练参数
- `--loss`: 损失函数 (cross_entropy, mse, l1_loss)
- `--optimizer`: 优化器 (sgd, momentum, adam, adamw, adagrad)
- `--regularization`: 正则化 (none, l1, l2, dropout, data_augmentation)
- `--epochs`: 训练轮数 (默认: 100)
- `--batch_size`: 批次大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.01)
- `--activation`: 激活函数 (relu, tanh, sigmoid)

#### 测试参数
- `--model_path`: 模型文件路径
- `--loss_function`: 损失函数类型
- `--batch_size`: 测试批次大小 (默认: 64)

## 示例用法

### 基础训练
```bash
# 使用默认配置（SGD + 交叉熵 + L2正则化）
python train.py

# 使用Adam优化器训练
python train.py --optimizer adam

# 使用Momentum SGD和Dropout训练
python train.py --optimizer momentum --regularization dropout
```

### 高级配置
```bash
# 使用AdamW优化器和数据增强
python train.py --loss cross_entropy --optimizer adamw --regularization data_augmentation --lr 0.001 --epochs 100

# 使用L1正则化
python train.py --loss mse --optimizer momentum --regularization l1 --lambda_reg 0.0001

# 使用Dropout正则化
python train.py --loss cross_entropy --optimizer adam --regularization dropout --dropout_rate 0.1
```

### 测试和可视化
```bash
# 测试特定模型
python test.py --model_path models/sgd_cross_entropy_l2_final.npy --loss_function cross_entropy

# 测试所有模型
bash test.sh
```

## 默认配置

- **网络结构**: [784, 2048, 512, 256, 10]（输入层784，三个隐藏层2048、512、256，输出层10）
- **激活函数**: ReLU
- **损失函数**: 交叉熵
- **优化器**: SGD
- **正则化**: L2
- **学习率**: 0.01
- **批次大小**: 64
- **训练轮数**: 100

## 输出文件

训练完成后会生成：
- `models/`: 保存的模型权重（命名格式: 优化器_损失函数_正则化方法_final.npy）
- `test_results/`: 测试结果和可视化图表