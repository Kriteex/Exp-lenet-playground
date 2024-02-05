import torch
from torchvision import datasets, transforms

def load_data():
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Načítání trénovacích a testovacích dat
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)

    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

    return train_loader, test_loader
