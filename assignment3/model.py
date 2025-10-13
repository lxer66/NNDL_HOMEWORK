from torch import nn
import torch.nn.functional as F
import torch

class resnet_block(nn.Module):
    def __init__(self, input_channels, output_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
            self.sc = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2)
            self.bn = nn.BatchNorm2d(output_channels)
        else:
            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = self.bn(self.sc(x))
        out += x
        out = F.relu(out)
        return out
        

class resnet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            resnet_block(16, 16),
            resnet_block(16, 32, downsample=True),
            resnet_block(32, 64, downsample=True)
        )
        self.classifer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.blocks(out)
        out = self.classifer(out)
        return out
    
if __name__ == "__main__":
    test_model = resnet20()
    print(test_model)
    test_tensor = torch.randn(64, 3, 32, 32)
    print('input size:', test_tensor.shape)
    out = test_model(test_tensor)
    print('output size:', out.shape)