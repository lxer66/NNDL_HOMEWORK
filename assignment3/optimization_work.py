from model import get_norm_layer
from train_compare import plot_training_curves
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt

# 数据增强和预处理：使用ImageNet的均值和标准差进行标准化，并使用随机翻转和裁剪
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
])

train_set = torchvision.datasets.CIFAR10(
    root='./train_set',
    train=True,
    transform=transform,
    download=True
)        

test_set = torchvision.datasets.CIFAR10(
    root='./test_set',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]),
    download=True
)        
    
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

# 优化版ResNet残差块
class optimized_resnet_block(nn.Module):
    def __init__(self, input_channels, output_channels, downsample=False, normalization='bn'):
        super().__init__()
        self.downsample = downsample
        self.normalization = normalization
        stride = 2 if downsample else 1
        # 使用bias=False，因为后面有归一化层
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_norm_layer(normalization, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm_layer(normalization, output_channels)
        if downsample:
            self.sc = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, bias=False)
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

# 优化版ResNet-20模型
class optimized_resnet20(nn.Module):
    def __init__(self, dropout=0, normalization='bn'):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.normalization = normalization
        self.bn = get_norm_layer(normalization, 16)        
        self.blocks = nn.Sequential(
            optimized_resnet_block(16, 16, normalization=normalization),
            optimized_resnet_block(16, 16, normalization=normalization),
            optimized_resnet_block(16, 16, normalization=normalization),
    
            optimized_resnet_block(16, 32, downsample=True, normalization=normalization),
            optimized_resnet_block(32, 32, normalization=normalization),
            optimized_resnet_block(32, 32, normalization=normalization),
            
            optimized_resnet_block(32, 64, downsample=True, normalization=normalization),
            optimized_resnet_block(64, 64, normalization=normalization),
            optimized_resnet_block(64, 64, normalization=normalization)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = self.classifier(out)
        return out


# 优化训练过程
def optimized_train(net, train_iter, num_epochs, lr, device, weight_decay=0, dropout=0, normalization='bn'):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )
    # 学习率调度器：在第100和150轮时降低学习率
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150], 
        last_epoch=-1
    )
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        train_num = 0
        print(f'epoch {epoch + 1}, current lr {optimizer.param_groups[0]['lr']}', end=' ')
        for i, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_loss += l.item() * x.size(0)
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
            train_num += x.size(0) 
        train_loss = train_loss / train_num
        train_acc = train_acc / train_num
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f'train loss {train_loss:.4f}, train acc {train_acc:.4f}')
        lr_scheduler.step()

    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_name = f"optimized_resnet20.pth"
    model_path = os.path.join('models', model_name)
    torch.save(net.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")
    plot_training_curves(train_losses, train_accuracies, lr, weight_decay, dropout, normalization=normalization)
    return train_losses, train_accuracies


# 测试优化模型
def optimized_test(net, test_iter, model_path='models/optimized_resnet20.pth'):
    net.load_state_dict(torch.load(model_path))
    net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    test_acc = 0
    test_num = 0
    with torch.no_grad():
        for x, y in test_iter:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            test_acc += (y_hat.argmax(dim=1) == y).sum().item()
            test_num += x.size(0)
    
    accuracy = test_acc / test_num
    return accuracy

# 主程序：训练和测试优化模型
if __name__ == "__main__":
    optimized_ResNet20 = optimized_resnet20()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimized_train(optimized_ResNet20, train_loader, 200, 0.1, device, weight_decay=1e-4)
    optimized_accuracy = optimized_test(optimized_ResNet20, test_loader)
    print(f"最终模型测试准确率: {optimized_accuracy:.4f}")
    with open('test.txt', 'a') as f:
        f.write('models/optimized_resnet20.pth' + f' test accuracy {optimized_accuracy:.4f}\n')