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

    
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


if __name__ == "__main__":
    for x, y in train_loader:
        print('train data')
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        break

    for x, y in test_loader:
        print('test data')
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        break


    
    



