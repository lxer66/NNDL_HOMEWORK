import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import config

# --- 1. 初始化工具：正交初始化 (Orthogonal Initialization) ---
# 这是解决深层 GAN 梯度消失的现代标准方法
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

# --- 2. 基础组件 ---
def sn_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

# --- 3. Self-Attention 模块 (BigGAN 同款) ---
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = sn_conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = sn_conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = sn_conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return self.gamma * out + x

# --- 4. ResBlock Generator (BN + ReLU) ---
class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=False):
        super(ResBlockG, self).__init__()
        self.upsample = upsample
        self.conv1 = sn_conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = sn_conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or upsample:
            self.shortcut = sn_conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        h = self.bn1(x)
        h = F.relu(h)
        h = self.conv1(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv2(h)
        
        s = x
        if self.upsample:
            s = F.interpolate(s, scale_factor=2)
        s = self.shortcut(s)
        return h + s

# --- 5. ResBlock Discriminator (SN + ReLU) ---
class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, first_block=False):
        super(ResBlockD, self).__init__()
        self.downsample = downsample
        self.first_block = first_block
        self.conv1 = sn_conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = sn_conv2d(out_ch, out_ch, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or downsample:
            self.shortcut = sn_conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        # 第一层 Block 通常不加 ReLU，保留原始信息
        h = x if self.first_block else F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            
        s = x 
        s = self.shortcut(s)
        if self.downsample:
            s = F.avg_pool2d(s, 2)
        return h + s

# --- 6. Generator 主体 ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Linear 也可以用 SN 稍微稳一点，但不是必须
        self.linear = utils.spectral_norm(nn.Linear(config.NZ, 4 * 4 * config.NGF))
        
        # 4x4 -> 8x8
        self.block1 = ResBlockG(config.NGF, config.NGF, upsample=True)
        # 8x8 -> 16x16
        self.block2 = ResBlockG(config.NGF, config.NGF, upsample=True)
        # Self-Attention
        self.attn = SelfAttention(config.NGF)
        # 16x16 -> 32x32
        self.block3 = ResBlockG(config.NGF, config.NGF, upsample=True)
        
        self.bn_out = nn.BatchNorm2d(config.NGF)
        self.conv_out = sn_conv2d(config.NGF, 3, 3, 1, 1)

    def forward(self, z):
        h = self.linear(z)
        h = h.view(-1, config.NGF, 4, 4)
        h = self.block1(h)
        h = self.block2(h)
        h = self.attn(h)
        h = self.block3(h)
        h = F.relu(self.bn_out(h))
        h = torch.tanh(self.conv_out(h))
        return h

# --- 7. Discriminator 主体 ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 32x32 -> 16x16
        self.block1 = ResBlockD(3, config.NDF, downsample=True, first_block=True)
        # Self-Attention
        self.attn = SelfAttention(config.NDF)
        # 16x16 -> 8x8
        self.block2 = ResBlockD(config.NDF, config.NDF, downsample=True)
        # 8x8 -> 4x4
        self.block3 = ResBlockD(config.NDF, config.NDF, downsample=True)
        # 4x4 -> 4x4
        self.block4 = ResBlockD(config.NDF, config.NDF, downsample=False)
        
        # [核弹级优化]：最后一层绝对不加 Spectral Norm
        # 也不加 Sigmoid，只输出 Logits
        self.linear = nn.Linear(config.NDF, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.attn(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.relu(h)
        h = torch.sum(h, dim=(2, 3)) # Global Sum Pooling
        out = self.linear(h)
        return out