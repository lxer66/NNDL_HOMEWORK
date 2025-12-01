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

os.makedirs(config.OUT_DIR, exist_ok=True)
os.makedirs(config.WEIGHTS_DIR, exist_ok=True)

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def main():
    print(f"Loading Data on {config.DEVICE}...")
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print("Initializing High-Capacity SNGAN...")
    netG = Generator().to(config.DEVICE)
    netD = Discriminator().to(config.DEVICE)
    
    # [关键] 应用正交初始化
    netG.apply(init_weights)
    netD.apply(init_weights)

    # 优化器
    optimizerD = optim.Adam(netD.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))

    fixed_noise = torch.randn(64, config.NZ, device=config.DEVICE)
    
    G_losses = []
    D_losses = []

    print(f"Starting Training (n_critic={config.N_CRITIC})...")
    
    # 全局 Step 计数器
    step = 0
    
    for epoch in range(config.EPOCHS):
        # 使用 iter 手动控制循环，以便处理 n_critic
        data_iter = iter(dataloader)
        i = 0
        
        while i < len(dataloader):
            ############################
            # (1) Update Discriminator
            ###########################
            # 判别器需要更多的数据，所以我们在内层循环更新它
            netD.zero_grad()
            
            # 获取真实图片
            try:
                data = next(data_iter)
            except StopIteration:
                break # Epoch 结束
                
            real_cpu = data[0].to(config.DEVICE)
            b_size = real_cpu.size(0)

            # --- Real ---
            output_real = netD(real_cpu)
            # Hinge Loss Real: mean(ReLU(1 - D(x)))
            errD_real = torch.mean(torch.nn.ReLU()(1.0 - output_real))
            errD_real.backward()

            # --- Fake ---
            noise = torch.randn(b_size, config.NZ, device=config.DEVICE)
            fake = netG(noise).detach() # 必须 detach，不传梯度给 G
            output_fake = netD(fake)
            # Hinge Loss Fake: mean(ReLU(1 + D(G(z))))
            errD_fake = torch.mean(torch.nn.ReLU()(1.0 + output_fake))
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()
            
            i += 1
            step += 1

            ############################
            # (2) Update Generator (Every n_critic steps)
            ###########################
            if step % config.N_CRITIC == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, config.NZ, device=config.DEVICE)
                fake = netG(noise) # 这里不需要 detach
                output_fake = netD(fake)
                
                # Hinge Loss Generator: -mean(D(G(z)))
                errG = -torch.mean(output_fake)
                errG.backward()
                optimizerG.step()

                # 记录 Loss (只在 G 更新时记录，避免长度不一致)
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if step % 100 == 0:
                     print(f'[{epoch}/{config.EPOCHS}][{i}/{len(dataloader)}] '
                           f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # 保存图片和模型
        if ((epoch + 1) % 5 == 0) or (epoch == config.EPOCHS - 1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            vutils.save_image(fake, f'{config.OUT_DIR}/epoch_{epoch + 1}.png', normalize=True, nrow=8)
            torch.save(netG.state_dict(), f'{config.WEIGHTS_DIR}/netG_epoch_{epoch + 1}.pth')

    torch.save(netG.state_dict(), f'{config.WEIGHTS_DIR}/netG_final.pth')
    print("Training Finished.")

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.title("SNGAN Hinge Loss (n_critic=5)")
    plt.plot(smooth_curve(G_losses), label="Generator", color="#1f77b4")
    plt.plot(smooth_curve(D_losses), label="Discriminator", color="#ff7f0e")
    plt.xlabel("G Iterations") # 注意这里是 G 更新的次数
    plt.ylabel("Hinge Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'{config.OUT_DIR}/loss_curve_final.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()