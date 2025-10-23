from model import resnet20
from datasets import train_set, test_set
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 评估模型在验证集上的性能
def evaluate(net, data_iter, device=None):
    net.eval()
    val_loss = 0
    acc_sum = 0
    num_samples = 0
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in data_iter:
            if device:
                x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            val_loss += l.item() * x.size(0)
            acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            num_samples += x.size(0)
    return val_loss / num_samples, acc_sum / num_samples

# 训练并评估模型
def train_and_evaluate(net, train_iter, val_iter, num_epochs, lr, device,  weight_decay=0, dropout=0, is_max=False):
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
    val_loss, val_acc = 0, 0
    max_val_acc = 0
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
        val_loss, val_acc = evaluate(net, val_iter, device)
        if is_max and val_acc > max_val_acc:
            max_val_acc = val_acc
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}, val loss {val_loss:.4f}, val acc {val_acc:.4f}')
            with open('k_fold_cross_valiation.txt', 'a') as f:
                f.write(f'epoch {epoch + 1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}, val loss {val_loss:.4f}, val acc {val_acc:.4f}\n')
    return max_val_acc if is_max else val_acc

# K折交叉验证实现
def k_fold_cross_validation(dataset, k_folds, num_epochs, lr, weight_decay=0, dropout=0, batch_size=64):
    dataset_size = len(dataset)
    fold_size = dataset_size // k_folds
    fold_results = []
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    indices = np.random.permutation(dataset_size)
    
    for fold in range(k_folds):
        print(f"正在进行第 {fold + 1} 折交叉验证...")
        with open('k_fold_cross_valiation.txt', 'a') as f:
            f.write(f'正在进行第 {fold + 1} 折交叉验证...\n')
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else dataset_size
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        net = resnet20()
        val_acc = train_and_evaluate(net, train_loader, val_loader, num_epochs, lr, device, weight_decay=weight_decay, is_max=True)
        fold_results.append(val_acc)
        print(f"第 {fold + 1} 折验证准确率: {val_acc:.4f}")
        with open('k_fold_cross_valiation.txt', 'a') as f:
            f.write(f'第 {fold + 1} 折验证准确率: {val_acc:.4f}\n')
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print(f"{k_folds}折交叉验证结果:")
    print(f"各折验证准确率: {[f'{acc:.4f}' for acc in fold_results]}")
    print(f"平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    with open('k_fold_cross_valiation.txt', 'a') as f:
        f.write(f'{k_folds}折交叉验证结果:\n')
        f.write(f'各折验证准确率: {[f'{acc:.4f}' for acc in fold_results]}\n')
        f.write(f'平均准确率: {mean_acc:.4f}\n')
    return fold_results, mean_acc, std_acc

# 主程序：对不同超参数组合进行K折交叉验证
if __name__ == "__main__":
    for lr in [0.1, 0.01, 0.001]:
        for weight_decay in [0.001, 0.0001, 0.00001]:
            k_fold_results, mean_accuracy, std_accuracy = k_fold_cross_validation(
                dataset=train_set,
                k_folds=3,
                num_epochs=100,  
                lr=lr,
                weight_decay=weight_decay,
                batch_size=128
            )
            print(f'学习率为{lr}, 权重衰退为{weight_decay}时的平均准确率为{mean_accuracy}')
            with open('k_fold_cross_valiation.txt', 'a') as f:
                f.write(f'学习率为{lr}, 权重衰退为{weight_decay}时的平均准确率为{mean_accuracy}\n \n')