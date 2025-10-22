import torch
from model import resnet20
import argparse
import os
import glob

def test(net, test_iter, model_path):
    net.load_state_dict(torch.load(model_path))
    net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    
    test_acc, test_num = 0, 0
    with torch.no_grad():
        for x, y in test_iter:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            test_acc += (y_hat.argmax(dim=1) == y).sum().item()
            test_num += x.size(0)
    return test_acc / test_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ResNet20 on CIFAR-10')
    parser.add_argument('--model_path', type=str, default='', 
                        help='Path to the trained model weights')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing model weights to test')
    args = parser.parse_args()
    
    net = resnet20()
    from datasets import test_loader
    
    model_files = []
    if os.path.isfile(args.model_path):
        model_files = [args.model_path]
    else:
        model_files = glob.glob(os.path.join(args.models_dir, "*.pth"))
    
    for model_file in model_files:
        if model_file == 'models/optimized_resnet20.pth':
            continue
        accuracy = test(net, test_loader, model_file)
        print(f"模型 {os.path.basename(model_file)} 的测试准确率: {accuracy:.4f}")
        with open('test.txt', 'a') as f:
            f.write(model_file + f' test accuracy {accuracy:.4f}\n')