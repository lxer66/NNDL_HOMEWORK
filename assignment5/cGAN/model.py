import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import config

# --- 1. 初始化工具 ---
def init_weights(m):
    # 对卷积、全连接、嵌入层进行正交初始化
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
        nn.init.orthogonal_(m.weight)
    
    # 单独处理 bias，只有当层存在 bias 属性且不为 None 时才初始化
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.fill_(0.0)

# --- 2. 基础组件 ---
def sn_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

def sn_embedding(num_embeddings, embedding_dim):
    return utils.spectral_norm(nn.Embedding(num_embeddings, embedding_dim))

# --- 3. Self-Attention 模块 ---
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

# --- 4. Conditional Batch Normalization (生成器核心) ---
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False) # 关闭自带参数
        
        # 将类别映射为 BN 的 gamma 和 beta
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # 初始化 gamma=1, beta=0
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        out = gamma * out + beta
        return out

# --- 5. ResBlock Generator (带条件输入) ---
class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=False):
        super(ResBlockG, self).__init__()
        self.upsample = upsample
        self.conv1 = sn_conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = sn_conv2d(out_ch, out_ch, 3, 1, 1)
        
        # 使用 CBN 替代普通 BN
        self.cbn1 = ConditionalBatchNorm2d(in_ch, config.NUM_CLASSES)
        self.cbn2 = ConditionalBatchNorm2d(out_ch, config.NUM_CLASSES)
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or upsample:
            self.shortcut = sn_conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x, y):
        h = self.cbn1(x, y)
        h = F.relu(h)
        h = self.conv1(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2)
        h = self.cbn2(h, y)
        h = F.relu(h)
        h = self.conv2(h)
        
        s = x
        if self.upsample:
            s = F.interpolate(s, scale_factor=2)
        s = self.shortcut(s)
        return h + s

# --- 6. ResBlock Discriminator  ---
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

# --- 7. Generator 主体 ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = utils.spectral_norm(nn.Linear(config.NZ, 4 * 4 * config.NGF))
        
        self.block1 = ResBlockG(config.NGF, config.NGF, upsample=True)
        self.block2 = ResBlockG(config.NGF, config.NGF, upsample=True)
        self.attn = SelfAttention(config.NGF)
        self.block3 = ResBlockG(config.NGF, config.NGF, upsample=True)
        
        self.bn_out = nn.BatchNorm2d(config.NGF)
        self.conv_out = sn_conv2d(config.NGF, 3, 3, 1, 1)

    def forward(self, z, y):
        h = self.linear(z)
        h = h.view(-1, config.NGF, 4, 4)
        h = self.block1(h, y)
        h = self.block2(h, y)
        h = self.attn(h)
        h = self.block3(h, y)
        h = F.relu(self.bn_out(h))
        h = torch.tanh(self.conv_out(h))
        return h

# --- 8. Projection Discriminator (判别器核心) ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = ResBlockD(3, config.NDF, downsample=True, first_block=True)
        self.attn = SelfAttention(config.NDF)
        self.block2 = ResBlockD(config.NDF, config.NDF, downsample=True)
        self.block3 = ResBlockD(config.NDF, config.NDF, downsample=True)
        self.block4 = ResBlockD(config.NDF, config.NDF, downsample=False)
        
        # 1. 线性判别
        self.linear = nn.Linear(config.NDF, 1)
        # 2. 投影判别 (类别嵌入)
        self.embed = sn_embedding(config.NUM_CLASSES, config.NDF)

    def forward(self, x, y):
        h = self.block1(x)
        h = self.attn(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.relu(h)
        h = torch.sum(h, dim=(2, 3)) # Global Pooling
        
        out_linear = self.linear(h)
        
        # Projection: h^T * embed(y)
        embed_y = self.embed(y)
        out_projection = torch.sum(embed_y * h, dim=1, keepdim=True)
        
        return out_linear + out_projection