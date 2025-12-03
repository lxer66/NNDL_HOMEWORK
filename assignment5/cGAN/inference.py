import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import argparse
import os
import matplotlib.pyplot as plt
import config
from model import Generator

# CIFAR-10 类别名称
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def inference(class_idx, num_images, upscale_size):
    device = config.DEVICE
    netG = Generator().to(device)
    
    # 加载生成器权重
    model_path = f'{config.WEIGHTS_DIR}/netG_final.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    print(f"Generating {num_images} images of class: {CLASSES[class_idx]}...")
    
    with torch.no_grad():
        # 随机噪声和类别标签
        z = torch.randn(num_images, config.NZ, device=device)
        labels = torch.full((num_images,), class_idx, dtype=torch.long, device=device)
        
        # 生成图片
        fake = netG(z, labels)
        fake = (fake + 1) / 2.0  # 转换到 [0, 1]

        # 如果需要，调整图片大小
        if upscale_size != 32:
            fake = F.interpolate(fake, size=(upscale_size, upscale_size), mode='bicubic', align_corners=False)
            fake = torch.clamp(fake, 0, 1)

        # 保存生成结果
        os.makedirs(config.INFERENCE_DIR, exist_ok=True)
        save_path = f'{config.INFERENCE_DIR}/cgan_{CLASSES[class_idx]}_{upscale_size}px.png'
        
        nrow = int(num_images**0.5)  # 图片网格行数
        vutils.save_image(fake, save_path, nrow=nrow, padding=2)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx', type=int, default=7, help='0=plane, 1=car, 7=horse')
    parser.add_argument('--num', type=int, default=16)  # 生成图片数量
    parser.add_argument('--size', type=int, default=32)  # 输出图片尺寸
    args = parser.parse_args()
    
    inference(args.class_idx, args.num, args.size)