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
├── analysis.md              # 实验结果分析报告
├── assignment_instructions.md # 说明文档
├── config.py                # 训练配置和命令行参数
├── download.py              # MNIST数据集下载
├── evaluate.py              # 模型评估
├── homework_content.md      # 作业内容说明
├── loss_function.py         # 损失函数实现
├── network.py               # 神经网络实现
├── optimizer.py             # 优化器实现
├── preprocess.py            # 数据预处理
├── regulation.py            # 正则化方法实现
├── requirements.txt         # 项目依赖
├── test.py                  # 独立测试模块
├── train.py                 # 训练流程
├── train.sh                 # 训练脚本
├── test.sh                  # 测试脚本
├── visualize.py             # 可视化功能
├── data/                    # 数据目录（此文件夹为上传至github）
│   ├── train_images.gz      # MNIST训练图像数据
│   ├── train_labels.gz      # MNIST训练标签数据
│   ├── test_images.gz       # MNIST测试图像数据
│   ├── test_labels.gz       # MNIST测试标签数据
│   ├── mnist_data.pkl       # 下载的原始数据缓存
│   └── processed_mnist*.pkl # 预处理后的数据缓存
├── models/                  # 训练好的模型权重
├── test_results/            # 测试结果和可视化图表
└── train_results/           # 训练过程可视化图表
```

## 网络结构

默认网络结构为 [784, 256, 128, 10]：
- 输入层：784个神经元（对应28x28像素的MNIST图像）
- 隐藏层1：256个神经元
- 隐藏层2：128个神经元
- 输出层：10个神经元（对应0-9十个数字类别）

## 损失函数

支持三种损失函数：
1. **交叉熵损失（Cross Entropy）**：适用于分类任务的经典损失函数
2. **均方误差损失（MSE）**：常用于回归任务，但也可用于分类
3. **绝对值损失（L1 Loss）**：对异常值相对不敏感的损失函数

## 优化器

支持五种优化算法：
1. **SGD（随机梯度下降）**：基础优化算法
2. **Momentum SGD（带动量的SGD）**：通过动量加速收敛
3. **Adam**：结合动量和自适应学习率的优化算法
4. **AdamW**：带权重衰减的Adam优化器
5. **AdaGrad**：自适应学习率优化算法

## 正则化方法

支持五种正则化技术：
1. **L1正则化**：通过权重绝对值和进行约束
2. **L2正则化**：通过权重平方和进行约束
3. **Dropout**：训练过程中随机丢弃部分神经元
4. **数据增强**：通过对训练数据进行轻微变换增加数据多样性
5. **无正则化**：不使用任何正则化技术

## 使用方法（即作业完成的步骤）

### 1. 训练模型

将原始题目和两个附加题一起在同一个训练脚本中完成所有实验，直接使用以下命令：
```bash
bash train.sh
```

如果想要使用自定义配置训练（与作业无关），可以使用以下命令：
```bash
python train.py --loss 选择损失函数 --optimizer 选择优化算法 --regularization 选择正则化方法 --epochs 选择训练轮数 --lr 选择学习率 --activation 选择激活函数
```
具体其他参数还可以通过命令行参数进行配置，详见配置文件`config.py`。

### 2. 测试模型

测试步骤1训练好的所有模型，直接使用以下命令：
```bash
bash test.sh
```

自定义测试模型（前提是该权重路径存在），可以使用以下命令：
```bash
python test.py --model_path 选择权重路径
```
### 3. 常用命令行参数介绍

#### 训练参数
- `--loss`: 损失函数 (cross_entropy, mse, l1_loss)，默认为cross_entropy
- `--optimizer`: 优化器 (sgd, momentum, adam, adamw, adagrad)，默认为sgd
- `--regularization`: 正则化 (none, l1, l2, dropout, data_augmentation)，默认为l2
- `--epochs`: 训练轮数 (默认: 100)
- `--batch_size`: 批次大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.01)
- `--activation`: 激活函数 (relu, tanh, sigmoid)，默认为relu

#### 测试参数
- `--model_path`: 模型文件路径，默认为models/sgd_cross_entropy_l2_final.npy
- `--loss_function`: 损失函数类型，默认为cross_entropy
- `--batch_size`: 测试批次大小 (默认: 64)

## 默认配置

- **网络结构**: [784, 256, 128, 10]（输入层784，两个隐藏层256、128，输出层10）
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
- `train_results/`: 训练过程可视化图表

## 作业完成情况说明

* **作业内容说明：直接使用`train.sh`和`test.sh`即可完成作业。`train.sh`直接完成原始题目和两个附加题的训练，`test.sh`直接完成原始题目和两个附加题的测试，然后生成结果和可视化图表保存在train_results和test_results文件夹中，根据所有结果分析，得到analysis.md文件，即完成全部的assignment1作业，作业内容同时上传到[github](https://github.com/lxer66/NNDL_HOMEWORK)。**

* **关于大模型的使用说明：主要使用cursor以及通义灵码辅助生成。总体思路和文件的框架以及debug基本由本人完成，包括19个实验组合的安排，各个文件里应该写什么样的代码，完成什么样的工作，训练、测试脚本的使用和以及使用命令行参数的想法，训练和测试结果的分析，解决代码生成问题等等。大模型主要完成的工作是按照本人的指令完成代码具体的细节实现和优化。**