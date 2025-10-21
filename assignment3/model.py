from torch import nn
import torch.nn.functional as F
import torch

def get_norm_layer(norm_type, num_features):
    if norm_type == 'bn':
        return nn.BatchNorm2d(num_features)
    elif norm_type == 'ln':
        return nn.LayerNorm(num_features)
    elif norm_type == 'gn':
        return nn.GroupNorm(num_groups=min(32, num_features//4), num_channels=num_features)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(num_features, affine=True)
    elif norm_type == 'lrn':
        return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

class resnet_block(nn.Module):
    def __init__(self, input_channels, output_channels, downsample=False, normalization='bn'):
        super().__init__()
        self.downsample = downsample
        self.normalization = normalization

        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = get_norm_layer(normalization, output_channels)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = get_norm_layer(normalization, output_channels)
        
        if downsample:
            self.sc = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2)
            self.norm_sc = get_norm_layer(normalization, output_channels)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample:
            identity = self.norm_sc(self.sc(identity))
        
        out += identity
        out = F.relu(out)
        
        return out

class resnet20(nn.Module):
    def __init__(self, dropout=0, normalization='bn'):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.normalization = normalization
        
        self.blocks = nn.Sequential(
            resnet_block(16, 16, normalization=normalization),
            resnet_block(16, 16, normalization=normalization),
            resnet_block(16, 16, normalization=normalization),
            
            resnet_block(16, 32, downsample=True, normalization=normalization),
            resnet_block(32, 32, normalization=normalization),
            resnet_block(32, 32, normalization=normalization),
            
            resnet_block(32, 64, downsample=True, normalization=normalization),
            resnet_block(64, 64, normalization=normalization),
            resnet_block(64, 64, normalization=normalization)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        out = self.conv(x)
        out = self.blocks(out)
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    test_model = resnet20()
    print(test_model)
    test_tensor = torch.randn(64, 3, 32, 32)
    print('input size:', test_tensor.shape)
    out = test_model(test_tensor)
    print('output size:', out.shape)