import torch

# --- 实验超参数设置 (High-End GPU Version) ---

DATA_PATH = './data'
BATCH_SIZE = 128  # 大显存可以尝试 256，但 128 往往对泛化更好
IMAGE_SIZE = 32
NC = 3

# --- 升级点 1: 扩大潜在空间 ---
NZ = 128  # 从 100 增加到 128，携带更多信息

# --- 升级点 2: 巨大的网络容量 ---
NGF = 128 # 生成器特征基数翻倍 (64 -> 128)
NDF = 128 # 判别器特征基数翻倍 (64 -> 128)

EPOCHS = 100 # 训练更久，反正你跑得快

# --- 升级点 3: TTUR 策略 ---
# 判别器学快点，生成器学慢点
LR_D = 0.0004
LR_G = 0.0001
BETA1 = 0.0 # 现代 GAN (如 BigGAN) 常把 Momentum 设为 0
BETA2 = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config: Using High-Performance device: {DEVICE}")