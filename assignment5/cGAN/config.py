import torch

# --- 路径配置 ---
DATA_PATH = './data'                # 数据集存储路径
OUT_DIR = './output_cgan'           # 输出图片保存路径
WEIGHTS_DIR = './weights_cgan'      # 模型权重保存路径
INFERENCE_DIR = './inference_results_cgan'  # 推理结果保存路径

# --- 硬件配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU 或 CPU

# --- 训练参数 (SNGAN / BigGAN 配置) ---
BATCH_SIZE = 128                    # 每批次训练样本数
EPOCHS = 200                        # 总训练轮数
N_CRITIC = 5                        # 判别器每训练 5 次，生成器训练 1 次

# --- 学习率 (TTUR) ---
LR_G = 2e-4                         # 生成器学习率
LR_D = 2e-4                         # 判别器学习率
BETA1 = 0.0                         # Adam 优化器的 beta1 参数
BETA2 = 0.9                         # Adam 优化器的 beta2 参数

# --- 模型架构 ---
IMAGE_SIZE = 32                     # 输入图片尺寸 (宽/高)
NZ = 128                            # 潜在向量维度
NGF = 256                           # 生成器特征图通道数
NDF = 256                           # 判别器特征图通道数
NUM_CLASSES = 10                    # CIFAR-10 数据集类别数

# 打印配置信息
print(f"Config: {DEVICE} | Batch: {BATCH_SIZE} | Conditional Mode: ON")