import torch

# --- 路径配置 ---
DATA_PATH = './data'
OUT_DIR = './output'
WEIGHTS_DIR = './weights'
INFERENCE_DIR = './inference_results'

# --- 硬件配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 训练参数 (SNGAN / BigGAN Best Practices) ---
BATCH_SIZE = 128       # 128 是 CIFAR-10 生成任务的甜点
EPOCHS = 200           # 训练更久以获得最佳 FID
N_CRITIC = 5           # [关键] 判别器跑 5 步，生成器跑 1 步

# --- 学习率 (TTUR) ---
LR_G = 2e-4
LR_D = 2e-4
BETA1 = 0.0            # 必须为 0.0
BETA2 = 0.9

# --- 模型架构 ---
IMAGE_SIZE = 32
NZ = 128               # 潜在向量
NGF = 256              # [加强] 生成器通道数 (你有32G显存，直接拉满)
NDF = 256              # [加强] 判别器通道数 (直接拉满)

print(f"Config: {DEVICE} | Batch: {BATCH_SIZE} | Channels: {NGF}/{NDF} | n_critic: {N_CRITIC}")