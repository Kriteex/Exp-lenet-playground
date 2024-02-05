import torch
from model import LeNet
from dataset import load_data
import torch.nn.functional as F
import torch.nn as nn
from utils import hinge_loss

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if (batch_idx % 4 == 0):
            loss = F.cross_entropy(output, target)
        elif(batch_idx % 4 == 1):
            loss = F.mse_loss(output, F.one_hot(target, num_classes=10).float())
        elif(batch_idx % 4 == 2):
            output_log_softmax = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output_log_softmax, target)
        elif(batch_idx % 4 == 3):
            loss = hinge_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)')


def main():
    # Nastavení zařízení (GPU, pokud je dostupné)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Načítání dat
    train_loader, test_loader = load_data()

    # Inicializace modelu LeNet
    model = LeNet().to(device)

    # Definice optimizátoru
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trénování modelu
    for epoch in range(1, 2):  # Trénování pro 10 epoch
        train(model, device, train_loader, optimizer, epoch)

    # Testování modelu
    test(model, device, test_loader)
   
if __name__ == "__main__":
    main()