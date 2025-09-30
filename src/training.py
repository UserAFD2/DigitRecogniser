import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from rich.progress import Progress, BarColumn, TextColumn

'''
print(torch.version.cuda)   # Shows CUDA version PyTorch is using
print(torch.cuda.is_available())  # True means GPU is usable
print(torch.cuda.device_count())  # Number of GPUs detected

def main():
    import matplotlib.pyplot as plt
    import numpy as np
    # GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Define a transform to convert the data to a tensor and normalize it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST training dataset
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Create a DataLoader for the MNIST dataset
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

    # Get a batch of images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)  # Use next() instead of data_iter.next()

    # Function to show an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Show images
    #imshow(torchvision.utils.make_grid(images))

    # Print labels
    print(' '.join('%5s' % labels[j].item() for j in range(4)))
'''

# ---------------------------
# Model
# ---------------------------
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------
# Training loop
# ---------------------------
def train_model(model: ConvolutionalNetwork,
                train_loader: DataLoader,
                optimizer: Optimizer,
                criterion: _Loss,
                device: torch.device,
                epochs=20, patience=10) -> None:
    """
    Train a PyTorch model on the given dataset.
    
    Parameters
    ----------
    model : ConvolutionalNetwork
        The neural network to train.
    train_loader : DataLoader
        DataLoader providing training data.
    optimizer : Optimizer
        PyTorch optimizer.
    criterion : _Loss
        Loss function.
    device : torch.device
        Device to perform computations on.
    epochs : int, optional
        Maximum number of epochs (default is 20).
    patience : int, optional
        Early stopping patience (default is 10).
    
    Returns
    -------
    None
    """
    best_acc = 0
    stop_early = 0

    with Progress(
            TextColumn("[bold blue]{task.fields[name]} {task.completed}/{task.total} | Loss: {task.fields[loss]:.4f} | Acc: {task.fields[acc]:.2f}%"),
            BarColumn(),
            refresh_per_second=10
    ) as progress:

        epoch_task = progress.add_task("Epoch", total=epochs, name="Epoch", loss=0, acc=0)

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            batch_task = progress.add_task(
                f"Batch (Epoch {epoch+1})",
                total=len(train_loader),
                name="    Batch",
                loss=0,
                acc=0
            )

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total

                progress.update(batch_task, advance=1, loss=running_loss / (i + 1), acc=acc)
                progress.update(epoch_task, loss=running_loss / (i + 1), acc=acc)

            # Early stopping check
            if acc <= best_acc:
                stop_early += 1
            else:
                best_acc = acc
                stop_early = 0

            if stop_early > patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} with accuracy {acc:.2f}%")
                break

            progress.update(epoch_task, advance=1)

    print("✅ Training finished")


# ---------------------------
# Evaluation
# ---------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"📊 Test Accuracy: {acc:.2f}%")
    return acc
    # 99.32% train 99.09% test  before dropout with 128 batch size 15 epochs
    # 99.33% train 99.10% test  before dropout with 64 batch size 15 epochs
    # 99.14% train 99.12% test  after dropout with 64 batch size 15 epochs
    # 99.26% train 99.22% test  after dropout with 64 batch size 30 epochs
    # 99.22% 99.33%

# ---------------------------
# Main
# ---------------------------
def main():
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data: torch.utils.data.Dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data: torch.utils.data.Dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader: DataLoader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader: DataLoader = DataLoader(test_data, batch_size=64, shuffle=False)

    model: nn.Module = ConvolutionalNetwork().to(device)
    criterion: _Loss = nn.CrossEntropyLoss()
    optimizer: Optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    train_model(model, train_loader, optimizer, criterion, device, epochs=50, patience=10)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":

    # This line can be omitted if the program is not going to be frozen to produce an executable.
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
    main()


