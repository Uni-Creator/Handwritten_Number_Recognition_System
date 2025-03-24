import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/train.csv')
data = np.array(data)
m, n = data.shape

# Shuffle data
np.random.shuffle(data)

# Split into development and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.0

# Function to load model
def load_model(filename="model.npz"):
    data = np.load(filename)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
    print(f"Model loaded successfully from {filename}!")
    return W1, b1, W2, b2

# Activation Functions
def softmax(Z):
    Z -= np.max(Z, axis=0)  # Prevent overflow
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def Sin(Z):
    return np.tanh(Z)

# Forward Propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = Sin(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Get Predictions
def get_predictions(A2):
    return np.argmax(A2, 0)

# Get Accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Make Predictions
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

# Display an Image with Prediction
def give_prediction(index, W1, b1, W2, b2):
    current_image = X_dev[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)[0]
    label = Y_dev[index]

    print(f"Index: {index}, Prediction: {prediction}, Actual: {label}")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.title(f"Prediction: {prediction} Actual: {label}")
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Load model
W1, b1, W2, b2 = load_model()

# Test the model on validation data
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(f"Validation Accuracy: {get_accuracy(dev_predictions, Y_dev):.4f}")

# Show predictions for 5 random images
for _ in range(5):
    index = random.randint(0, 999)
    give_prediction(index, W1, b1, W2, b2)
