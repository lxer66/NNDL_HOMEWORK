import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import config
from model import Generator, Discriminator, weights_init

# --- 辅助函数：平滑曲线 ---
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
    # pin_memory=True 加速数据传输
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print("Initializing SN-GAN Models...")
    netG = Generator().to(config.DEVICE)
    netG.apply(weights_init)
    netD = Discriminator().to(config.DEVICE)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    
    # --- 优化可视化：只生成 32 张图，排成 4行8列 ---
    fixed_noise = torch.randn(32, config.NZ, 1, 1, device=config.DEVICE)

    # 采用 TTUR 不同的学习率
    optimizerD = optim.Adam(netD.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))

    print("Starting Training Loop (High Performance Mode)...")
    img_list = []
    G_losses = []
    D_losses = []
    
    iters = 0

    for epoch in range(config.EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D
            ###########################
            netD.zero_grad()
            real_cpu = data[0].to(config.DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=config.DEVICE)
            
            # Label Smoothing: 真实标签 1.0 -> 0.9 (让D不要太自信)
            label.fill_(0.9) 
            
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            noise = torch.randn(b_size, config.NZ, 1, 1, device=config.DEVICE)
            fake = netG(noise)
            label.fill_(0.)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G
            ###########################
            netG.zero_grad()
            label.fill_(1.) 
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(f'[{epoch}/{config.EPOCHS}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            iters += 1

        # 保存图片
        if (epoch % 10 == 0) or (epoch == config.EPOCHS - 1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            # nrow=8 表示一行放8张
            img_list.append(vutils.make_grid(fake, nrow=8, padding=2, normalize=True))

    print("Training Finished.")

    # --- 结果可视化优化 ---
    
    # 1. 绘制 Loss (原始 + 平滑)
    plt.figure(figsize=(12, 6))
    plt.title("Generator and Discriminator Loss (Smoothed)")
    
    # 绘制半透明的原始数据
    plt.plot(G_losses, label="G (Raw)", alpha=0.3, color='tab:blue')
    plt.plot(D_losses, label="D (Raw)", alpha=0.3, color='tab:orange')
    
    # 绘制深色的平滑数据
    plt.plot(smooth_curve(G_losses), label="G (Smooth)", linewidth=2, color='tab:blue')
    plt.plot(smooth_curve(D_losses), label="D (Smooth)", linewidth=2, color='tab:orange')
    
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3) # 加个网格好看点
    plt.savefig('training_loss_enhanced.png')
    plt.show()

    # 2. 展示最终生成的图像 (带解释)
    plt.figure(figsize=(10, 5)) # 调整画布比例适配 4x8
    plt.axis("off")
    plt.title(f"Generated Samples at Epoch {config.EPOCHS} \n(Random Latent Vectors z ~ N(0, I))")
    plt.imshow(np.transpose(img_list[-1], (1,2,0)))
    plt.tight_layout()
    plt.savefig('generated_images_enhanced.png')
    plt.show()

if __name__ == "__main__":
    main()