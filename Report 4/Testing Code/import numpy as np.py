import numpy as np
import torch
from torch import nn
import tqdm
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Load and normalize the dataset
train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         (0.1307,), (0.3081,))
                                                 ]))

test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))

# Create validation set (10% of training data)
train_indices, val_indices, _, _ = train_test_split(
    range(len(train_dataset)),
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=0.1,
)

train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, val_indices)

# DataLoader setup
train_batch_size = 512
test_batch_size = 256

train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=False)
test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

num_train_batches = len(train_batches)
num_val_batches = len(val_batches)
num_test_batches = len(test_batches)

# Define the FCN model
class ACAIGFCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ACAIGFCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = ACAIGFCN(input_dim=784, output_dim=10)
learning_rate = 0.001
epochs = 10
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
train_loss_list = np.zeros(epochs)
validation_accuracy_list = np.zeros(epochs)

for epoch in tqdm.trange(epochs):
    model.train()
    train_loss = 0.0
    for train_features, train_labels in train_batches:
        train_features = train_features.reshape(-1, 28*28)
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = loss_func(outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss_list[epoch] = train_loss / num_train_batches
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_features, val_labels in val_batches:
            val_features = val_features.reshape(-1, 28*28)
            outputs = model(val_features)
            _, predicted = torch.max(outputs.data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
    validation_accuracy = correct / total
    validation_accuracy_list[epoch] = validation_accuracy
    print(f"Epoch: {epoch}; Validation Accuracy: {validation_accuracy * 100:.2f}%")

# Plotting results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(validation_accuracy_list, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.legend()
plt.tight_layout()
plt.show()

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for test_features, test_labels in test_batches:
        test_features = test_features.reshape(-1, 28*28)
        outputs = model(test_features)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")