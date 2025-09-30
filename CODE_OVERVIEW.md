# Digit Recognition Project - Code Overview

This document explains the structure and key sections of the code in this project.

---

## 1. Imports

- `torch`, `torchvision` - core PyTorch libraries.
- `nn`, `optim` - for building and training neural networks.
- `rich.progress` - for progress bars during training.

---

## 2. Model Definition (`ConvolutionalNetwork`)

- Two convolutional layers with ReLU activations.
- Max pooling to reduce spatial dimensions.
- Two fully connected layers with dropout for regularization.
- `forward()` method defines the data flow.

---

## 3. Data Loading

- Uses `torchvision.datasets.MNIST`.
- Data is transformed to tensors and normalized.
- Wrapped in `DataLoader` for batching and shuffling.

```python
import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
```

---

## 4. Training Loop (`train_model`)

- Iterates over epochs and batches.
- Calculates loss and gradients, updates model parameters.
- Tracks accuracy and loss with Rich progress bars.
- Implements early stopping when accuracy plateaus.

---

## 5. Evaluation (`evaluate_model`)

- Puts the model in evaluation mode.
- Calculates overall test accuracy without updating weights.

---

## 6. Main Script

- Sets device (CPU or GPU).
- Loads datasets.
- Instantiates model, loss, optimizer.
- Calls `train_model()` and `evaluate_model()`.