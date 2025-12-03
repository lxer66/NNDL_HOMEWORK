import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import config
from model import Generator, Discriminator, init_weights

# 创建输出目录
os.makedirs(config.OUT_DIR, exist_ok=True)
os.makedirs(config.WEIGHTS_DIR, exist_ok=True)

def smooth_curve(points, factor=0.9):
    # 平滑曲线，用于绘制 Loss 图
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def main():
    # 加载 CIFAR-10 数据集
    print(f"Loading CIFAR-10 with Labels on {config.DEVICE}...")
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),  # 调整图片大小
        transforms.ToTensor(),                # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])
    dataset = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化生成器和判别器
    print("Initializing Conditional Projection SNGAN...")
    netG = Generator().to(config.DEVICE)
    netD = Discriminator().to(config.DEVICE)
    
    # 初始化权重
    netG.apply(init_weights)
    netD.apply(init_weights)

    # 定义优化器
    optimizerD = optim.Adam(netD.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))

    # --- 可视化固定噪声 ---
    # 生成 100 张图 (10行 x 10列)，每行对应一个类别
    fixed_noise = torch.randn(100, config.NZ, device=config.DEVICE)
    fixed_labels = torch.arange(10, device=config.DEVICE).repeat_interleave(10)
    
    G_losses = []  # 记录生成器的损失
    D_losses = []  # 记录判别器的损失

    print(f"Start Training (n_critic={config.N_CRITIC})...")
    step = 0  # 全局步数计数器
    
    for epoch in range(config.EPOCHS):
        data_iter = iter(dataloader)
        i = 0
        
        while i < len(dataloader):
            # (1) 更新判别器
            netD.zero_grad()
            try:
                real_img, labels = next(data_iter)  # 获取真实图片和标签
            except StopIteration:
                break
                
            real_img = real_img.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            b_size = real_img.size(0)

            # 计算真实图片的损失
            output_real = netD(real_img, labels)
            errD_real = torch.mean(torch.nn.ReLU()(1.0 - output_real))
            errD_real.backward()

            # 计算生成图片的损失
            noise = torch.randn(b_size, config.NZ, device=config.DEVICE)
            fake = netG(noise, labels).detach()  # 生成图片
            output_fake = netD(fake, labels)
            errD_fake = torch.mean(torch.nn.ReLU()(1.0 + output_fake))
            errD_fake.backward()

            errD = errD_real + errD_fake  # 判别器总损失
            optimizerD.step()
            
            i += 1
            step += 1

            # (2) 更新生成器
            if step % config.N_CRITIC == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, config.NZ, device=config.DEVICE)
                fake = netG(noise, labels)  # 生成图片
                output_fake = netD(fake, labels)
                
                errG = -torch.mean(output_fake)  # 生成器损失
                errG.backward()
                optimizerG.step()

                # 记录损失
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if step % 100 == 0:
                     print(f'[{epoch}/{config.EPOCHS}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # 保存生成结果和模型权重
        if ((epoch + 1) % 5 == 0) or (epoch == config.EPOCHS - 1):
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
            vutils.save_image(fake, f'{config.OUT_DIR}/epoch_{epoch + 1}.png', normalize=True, nrow=10)
            torch.save(netG.state_dict(), f'{config.WEIGHTS_DIR}/netG_epoch_{epoch + 1}.pth')

    # 保存最终模型
    torch.save(netG.state_dict(), f'{config.WEIGHTS_DIR}/netG_final.pth')
    print("Training Finished.")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.title("cGAN Hinge Loss")
    plt.plot(smooth_curve(G_losses), label="Generator")
    plt.plot(smooth_curve(D_losses), label="Discriminator")
    plt.legend()
    plt.savefig(f'{config.OUT_DIR}/loss_curve.png')
    plt.show()

if __name__ == "__main__":
    main()