import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
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
    transform=transform,
    download=True
)        

    
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)


if __name__ == "__main__":
    for x, y in train_loader:
        print('train data')
        print('len:', len(train_set))
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        break

    for x, y in test_loader:
        print('test data')
        print('len:', len(test_set))
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        break


    
    



