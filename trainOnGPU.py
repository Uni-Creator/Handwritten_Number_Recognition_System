import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

# Load and preprocess data
data = pd.read_csv('data/train.csv')
data = np.array(data)
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:].T / 255.0

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:].T / 255.0

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(f'Using device: {device}')

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

def train(model, X_train, Y_train, criterion, optimizer, epochs=500):
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
        labels = torch.tensor(Y_train, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            _, predictions = torch.max(outputs, 1)
            accuracy = (predictions == labels).sum().item() / labels.size(0)
            print(f"Iteration: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}")

def test(model, X, Y):
    model.eval()
    inputs = torch.tensor(X, dtype=torch.float32).to(device)
    labels = torch.tensor(Y, dtype=torch.long).to(device)
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    accuracy = (predictions == labels).sum().item() / labels.size(0)
    return accuracy, predictions

def visualize_prediction(index, X, Y, model):
    model.eval()
    inputs = torch.tensor(X[index].reshape(1, -1), dtype=torch.float32).to(device)
    outputs = model(inputs)
    _, prediction = torch.max(outputs, 1)
    label = Y[index]
    
    print(f"Index: {index}")
    print(f"Prediction: {prediction.item()}")
    print(f"Label: {label}")
    
    image = X[index].reshape(28, 28) * 255
    plt.title(f"Prediction: {prediction.item()}")
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()

# Initialize model, criterion and optimizer
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, X_train, Y_train, criterion, optimizer, epochs=500)

# Evaluate on development set
accuracy, dev_predictions = test(model, X_dev, Y_dev)
print(f"Development Set Accuracy: {accuracy}")

# Visualize some predictions
for i in range(5):
    index = random.randint(0, len(X_dev) - 1)
    visualize_prediction(index, X_dev, Y_dev, model)


torch.save(model.state_dict(), "model.pth")

print("Model saved successfully as model.pth!")