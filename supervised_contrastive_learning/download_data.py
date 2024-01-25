import torchvision.transforms as transforms
from torchvision.datasets import MNIST

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)
