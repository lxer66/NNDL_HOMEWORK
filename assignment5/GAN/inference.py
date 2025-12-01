import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import argparse
import config
from model import Generator

# --- 用户设定 (默认值) ---
MODEL_PATH = f'{config.WEIGHTS_DIR}/netG_final.pth' # 读取最终模型
NUM_IMAGES = 16   # 一次生成多少张
GRID_ROW = 4      # 网格行数

def inference(target_size):
    # 1. 准备环境
    os.makedirs(config.INFERENCE_DIR, exist_ok=True)
    device = config.DEVICE
    
    print(f"Loading Model from {MODEL_PATH}...")
    netG = Generator().to(device)
    
    # 加载权重
    try:
        netG.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("Error: Model file not found! Please run train.py first.")
        return
    
    netG.eval()

    # 2. 随机生成
    print(f"Generating {NUM_IMAGES} new images with random seeds...")
    print(f"Target Output Size: {target_size}x{target_size}")

    with torch.no_grad():
        # 每次运行 inference，这里的 z 都是随机新生成的
        z = torch.randn(NUM_IMAGES, config.NZ, device=device)
        
        # 生成原始 32x32 图像 (Range: [-1, 1])
        fake_imgs = netG(z)
        
        # 转换范围从 [-1, 1] 到 [0, 1] 以便后续处理和保存
        fake_imgs = (fake_imgs + 1) / 2.0
        
        # 3. 调整大小 (如果有需要)
        if target_size != 32:
            # 使用 Bicubic 插值放大，效果比较平滑
            output_imgs = F.interpolate(fake_imgs, size=(target_size, target_size), mode='bicubic', align_corners=False)
            # 插值可能会让数值略微溢出 [0,1]，重新截断一下保证图片正常
            output_imgs = torch.clamp(output_imgs, 0, 1)
        else:
            output_imgs = fake_imgs
        
        # 4. 保存图片
        save_path = f'{config.INFERENCE_DIR}/generated_{target_size}px.png'
        vutils.save_image(output_imgs, save_path, nrow=GRID_ROW, padding=2)
        print(f"Result saved to {save_path}")
        
        # 5. 展示图片
        # 读取刚刚保存的图片文件来展示 (这样能保证展示的和保存的一模一样)
        try:
            img_grid = plt.imread(save_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(img_grid)
            plt.axis('off')
            plt.title(f"Generated Result ({target_size}x{target_size})")
            plt.show()
        except Exception as e:
            print(f"Could not display image (headless environment?): {e}")

if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="Inference script for GAN generation.")
    parser.add_argument('--size', type=int, default=32, help='Target image size (e.g., 32, 64, 128, 256). Default is 32.')
    
    args = parser.parse_args()
    
    # 运行推理
    inference(target_size=args.size)