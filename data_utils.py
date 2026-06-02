import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def generate_data(batch_size=64):
    """
    Loads MNIST dataset, flattens images to 784-dim vectors, 
    and returns DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"MNIST Data Loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples.")
    return train_loader, test_loader
