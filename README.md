# LeNet Playground

## Description
This project provides a graphical user interface (GUI) for experimenting with the LeNet convolutional neural network (CNN) architecture. Users can dynamically adjust hyperparameters such as the number of filters, kernel size, pooling type, number of epochs, and batch size. The model is trained on the MNIST dataset using PyTorch, and training results are displayed in the GUI.

## Features
- **LeNet Architecture**: A customizable implementation of the LeNet CNN model.
- **Interactive GUI**: Built with PyQt5, allowing real-time adjustments of model parameters.
- **Dynamic Hyperparameters**:
  - Number of filters in the first and second convolutional layers.
  - Kernel size for convolutional layers.
  - Pooling type: Max Pooling or Average Pooling.
  - Number of epochs and batch size.
- **Training on MNIST**: Model trains on the MNIST dataset with configurable parameters.
- **Training Results**: Displays final average loss, training time, and selected parameters.

## Requirements
- Python 3.7 or higher
- Libraries: Install via `pip install -r requirements.txt`
  - `torch`
  - `torchvision`
  - `PyQt5`

## How to Run
1. Clone the repository:
  - git clone <repository_url>
  - cd <repository_name>

2. Install dependencies:
  - pip install -r requirements.txt

3. Run the application:

  - python lenet_playground.py

## Usage

1. Adjust Hyperparameters:
    - Use sliders and spinboxes in the GUI to set the number of filters, kernel size, pooling type, number of epochs, and batch size.
2. Start Training:
    - Click the "Start Training" button to begin training the LeNet model with the selected parameters.
3. View Results:
    - Training results, including average loss and training time, will appear in the GUI.

## Parameters

Number of Filters:
    - Adjustable for both convolutional layers (Conv1 and Conv2).
Kernel Size:
    - Configurable from 3 to 7.
Pooling Type:
    - Max Pooling or Average Pooling.
Epochs:
    - Configurable from 1 to 100.
Batch Size:
    - Configurable from 1 to 512.

## Example

- Filters in Conv1: 6
- Filters in Conv2: 16
- Kernel Size: 5
- Pooling: Max Pooling
- Epochs: 5
- Batch Size: 64

## Results:

- Training completed with detailed metrics including final average loss and training time.

## Notes

- Ensure the MNIST dataset is downloaded and available during training.
- Training time and loss may vary depending on the selected hyperparameters and system performance.

## Author

Developed by carlg420.