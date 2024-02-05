import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # C1 Convolutional Layer
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # input channels, output channels, kernel size
        # S2 Pooling Layer
        self.pool = nn.AvgPool2d(2, 2) # kernel size, stride
        # C3 Convolutional Layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # C5 Fully Connected Layer
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # in_features, out_features
        # F6 Fully Connected Layer
        self.fc2 = nn.Linear(120, 84)
        # Output Layer
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Applying layers and activations
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6) # Flatten the tensor
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the LeNet model
lenet = LeNet()
print(lenet)
