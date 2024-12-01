import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QSpinBox
from PyQt5.QtCore import Qt

# Definice LeNet architektury
class LeNet(nn.Module):
    def __init__(self, num_filters1=6, num_filters2=16):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=5)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=5)
        self.fc1 = nn.Linear(num_filters2 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# GUI s PyQt
class LeNetPlayground(QMainWindow):
    def __init__(self):
        super().__init__()

        # Inicializace GUI prvků
        self.initUI()

    def initUI(self):
        self.setWindowTitle('LeNet Playground')
        central_widget = QWidget()
        layout = QVBoxLayout()  # This is a local variable, not an instance variable

        # Tlačítko pro spuštění tréninku
        self.train_button = QPushButton('Start Training', self)
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # Slider for the number of filters in Conv1
        self.slider_filters1 = QSlider(self)
        self.slider_filters1.setMinimum(1)
        self.slider_filters1.setMaximum(64)
        self.slider_filters1.setValue(6)
        self.slider_filters1.setOrientation(Qt.Horizontal)
        layout.addWidget(QLabel("Filters in Conv1"))
        layout.addWidget(self.slider_filters1)

        # Slider for the number of filters in Conv2
        self.slider_filters2 = QSlider(self)
        self.slider_filters2.setMinimum(1)
        self.slider_filters2.setMaximum(64)
        self.slider_filters2.setValue(16)
        self.slider_filters2.setOrientation(Qt.Horizontal)
        layout.addWidget(QLabel("Filters in Conv2"))
        layout.addWidget(self.slider_filters2)

        # Slider for kernel size
        self.slider_kernel_size = QSlider(self)
        self.slider_kernel_size.setMinimum(3)
        self.slider_kernel_size.setMaximum(7)
        self.slider_kernel_size.setValue(5)
        self.slider_kernel_size.setOrientation(Qt.Horizontal)
        layout.addWidget(QLabel("Kernel size"))
        layout.addWidget(self.slider_kernel_size)

        # Dropdown for pooling type
        self.pooling_type_combo = QComboBox(self)
        self.pooling_type_combo.addItem("Max Pooling")
        self.pooling_type_combo.addItem("Average Pooling")
        layout.addWidget(QLabel("Pooling Type"))
        layout.addWidget(self.pooling_type_combo)

        # SpinBox for number of epochs
        self.epochs_spinbox = QSpinBox(self)
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(100)
        self.epochs_spinbox.setValue(2)
        layout.addWidget(QLabel("Number of Epochs"))
        layout.addWidget(self.epochs_spinbox)

        # SpinBox for batch size
        self.batch_size_spinbox = QSpinBox(self)
        self.batch_size_spinbox.setMinimum(1)
        self.batch_size_spinbox.setMaximum(512)
        self.batch_size_spinbox.setValue(64)
        layout.addWidget(QLabel("Batch Size"))
        layout.addWidget(self.batch_size_spinbox)

        # Zobrazit výsledky
        self.results_label = QLabel("Results will appear here", self)
        layout.addWidget(self.results_label)

        # Set the layout to the central widget
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def train_model(self):
        filters1 = self.slider_filters1.value()
        filters2 = self.slider_filters2.value()
        kernel_size = self.slider_kernel_size.value()
        pooling_type = self.pooling_type_combo.currentText()
        num_epochs = self.epochs_spinbox.value()
        batch_size = self.batch_size_spinbox.value()

        # Adjust the model based on the selected parameters
        model = LeNet(num_filters1=filters1, num_filters2=filters2)

        if pooling_type == "Max Pooling":
            pool_layer = torch.nn.functional.max_pool2d
        else:
            pool_layer = torch.nn.functional.avg_pool2d

        # Override the forward method to use the selected pooling type
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = pool_layer(x, 2)
            x = torch.relu(self.conv2(x))
            x = pool_layer(x, 2)
            conv_output_size = x.size(1) * x.size(2) * x.size(3)
            x = x.view(-1, conv_output_size)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        # Replace the model's forward method
        model.forward = forward.__get__(model)

        # Dataset and DataLoader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Training loop with the selected number of epochs
        start_time = time.time()
        total_loss = 0.0
        total_batches = len(train_loader)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0  # To accumulate loss over the epoch
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Log progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{total_batches}], Loss: {avg_loss:.4f}')

            total_loss += epoch_loss  # Accumulate loss for all epochs

        end_time = time.time()
        training_time = end_time - start_time

        # Calculate final average loss over all batches and epochs
        final_avg_loss = total_loss / (num_epochs * total_batches)

        # Detailed result message
        self.results_label.setText(
            f'Training completed with {filters1} and {filters2} filters.\n'
            f'Kernel size: {kernel_size}, Pooling: {pooling_type}\n'
            f'Final average loss: {final_avg_loss:.4f}\n'
            f'Total training time: {training_time:.2f} seconds')

# Spuštění aplikace
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LeNetPlayground()
    ex.show()
    sys.exit(app.exec_())
