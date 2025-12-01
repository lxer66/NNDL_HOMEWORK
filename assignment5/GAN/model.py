import torch
import torch.nn as nn
import torch.nn.utils as utils # 引入工具包
import config

# 初始化保持不变，虽然 SN 对初始化不敏感，但保留是个好习惯
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 生成器通常不需要 Spectral Norm，用 BN 足够
        self.main = nn.Sequential(
            # Input: NZ x 1 x 1
            nn.ConvTranspose2d(config.NZ, config.NGF * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.NGF * 4),
            nn.ReLU(True),
            # 4x4

            nn.ConvTranspose2d(config.NGF * 4, config.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NGF * 2),
            nn.ReLU(True),
            # 8x8

            nn.ConvTranspose2d(config.NGF * 2, config.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NGF),
            nn.ReLU(True),
            # 16x16

            nn.ConvTranspose2d(config.NGF, config.NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # 32x32
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # --- 关键升级: 使用 Spectral Normalization 包裹卷积层 ---
        # 这能极大稳定训练，防止梯度爆炸
        self.main = nn.Sequential(
            # 32x32 -> 16x16
            utils.spectral_norm(nn.Conv2d(config.NC, config.NDF, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            utils.spectral_norm(nn.Conv2d(config.NDF, config.NDF * 2, 4, 2, 1, bias=False)),
            # 移除 BatchNorm (有些高级 GAN 在 D 中移除 BN 以避免伪影，SN 已经起到了约束作用)
            # 但在 DCGAN 架构下保留 BN 也可以，这里我们尝试移除 BN 纯靠 SN，或者保留 BN。
            # 稳妥起见，配合 SN 使用 BN 也是可以的，但最纯粹的 SN-GAN 往往去掉 D 的 BN。
            # 这里为了复现难度适中，保留 BN，只加 SN。
            nn.BatchNorm2d(config.NDF * 2), 
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            utils.spectral_norm(nn.Conv2d(config.NDF * 2, config.NDF * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(config.NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1 (Scalar)
            # 最后一层不加 BN
            utils.spectral_norm(nn.Conv2d(config.NDF * 4, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)