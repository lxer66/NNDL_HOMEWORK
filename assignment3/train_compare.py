from model import resnet20
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt


def train(net, train_iter, num_epochs, lr, device, weight_decay=0, dropout=0, normalization='bn'):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=weight_decay
    )
    loss = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        train_num = 0
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
        
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}')
    
    # 创建models文件夹（如果不存在）
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存模型，使用优化器参数命名
    model_name = f"resnet20_lr{lr}_momentum0.9_wd{weight_decay}_dropout{dropout}_{normalization}normalization.pth"
    model_path = os.path.join('models', model_name)
    torch.save(net.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")
    
    plot_training_curves(train_losses, train_accuracies, lr, weight_decay, dropout, normalization=normalization)
    
    return train_losses, train_accuracies

def plot_training_curves(train_losses, train_accuracies, lr, weight_decay, dropout, normalization='bn'):
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制训练损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建共享x轴的第二个y轴
    ax2 = ax1.twinx()
    
    # 绘制训练准确率
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_accuracies, color=color, label='Train Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 设置标题
    plt.title(f'Training Curves - LR:{lr}, WD:{weight_decay}, DO:{dropout}, NO:{normalization}')
    
    # 保存图形到pictures文件夹
    if not os.path.exists('pictures'):
        os.makedirs('pictures')
        
    filename = f"resnet20_lr{lr}_momentum0.9_wd{weight_decay}_dropout{dropout}_{normalization}normalization.png"
    filepath = os.path.join('pictures', filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()  # 关闭图形以释放内存
    print(f"训练曲线已保存为 {filepath}")


if __name__ == "__main__":
    from datasets import train_loader
    ResNet20 = resnet20()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0, normalization='bn')
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0.1, normalization='bn')
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0.5, normalization='bn')
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0, normalization='ln')
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0, normalization='gn')
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0, normalization='in')
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0, normalization='lrn')
    train(ResNet20, train_loader, 100, lr=0.1, device=device, weight_decay=1e-5, dropout=0, normalization='none')