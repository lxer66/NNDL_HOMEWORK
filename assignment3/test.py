import torch
from model import resnet20
import argparse

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
    parser.add_argument('--model_path', type=str, default='models/resnet20_sgd_lr0.1_momentum0.9_wd1e-05.pth', 
                        help='Path to the trained model weights')
    args = parser.parse_args()
    
    net = resnet20()
    
    from datasets import test_loader
    
    accuracy = test(net, test_loader, args.model_path)
    print(f"测试准确率: {accuracy:.4f}")
    with open('test.txt', 'a') as f:
        f.write(args.model_path + f' test accuracy {accuracy:.4f}\n')