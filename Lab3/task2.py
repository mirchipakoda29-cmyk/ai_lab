import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt

# ===============================
# MNIST DATA LOADER
# ===============================
class MnistDataloader(object):

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):

        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):

        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images.append(img.reshape(28, 28))

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath)

        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath)

        return (x_train, y_train), (x_test, y_test)


# ===============================
# LOCAL DATASET PATH
# ===============================
training_images_filepath = r"C:\Users\raika\OneDrive\Desktop\ai-practice-assign\mnist_dataset\train-images.idx3-ubyte"
training_labels_filepath = r"C:\Users\raika\OneDrive\Desktop\ai-practice-assign\mnist_dataset\train-labels.idx1-ubyte"
test_images_filepath = r"C:\Users\raika\OneDrive\Desktop\ai-practice-assign\mnist_dataset\t10k-images.idx3-ubyte"
test_labels_filepath = r"C:\Users\raika\OneDrive\Desktop\ai-practice-assign\mnist_dataset\t10k-labels.idx1-ubyte"


mnist = MnistDataloader(training_images_filepath,
                        training_labels_filepath,
                        test_images_filepath,
                        test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ===============================
# PREPROCESSING
# ===============================
x_train = np.array(x_train).reshape(len(x_train), -1) / 255.0
x_test = np.array(x_test).reshape(len(x_test), -1) / 255.0

y_train = np.array(y_train)
y_test = np.array(y_test)

def one_hot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# ===============================
# ACTIVATION FUNCTIONS
# ===============================
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s*(1-s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha*x)

def leaky_relu_deriv(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# ===============================
# WEIGHT INITIALIZERS
# ===============================
def initialize_weights(init_type, input_size, hidden_size, output_size):

    if init_type == "zero":
        W1 = np.zeros((input_size, hidden_size))
        W2 = np.zeros((hidden_size, output_size))

    elif init_type == "random":
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        W2 = np.random.randn(hidden_size, output_size) * 0.01

    elif init_type == "xavier":
        W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
        W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)

    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, output_size))

    return W1, b1, W2, b2

# ===============================
# NEURAL NETWORK
# ===============================
class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size,
                 activation, initializer, lr=0.01):

        self.lr = lr

        self.W1, self.b1, self.W2, self.b2 = initialize_weights(
            initializer, input_size, hidden_size, output_size)

        if activation == "sigmoid":
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv

        elif activation == "tanh":
            self.act = tanh
            self.act_deriv = tanh_deriv

        elif activation == "relu":
            self.act = relu
            self.act_deriv = relu_deriv

        elif activation == "leaky_relu":
            self.act = leaky_relu
            self.act_deriv = leaky_relu_deriv

    def forward(self, X):

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.act(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)

        return self.A2

    def loss(self, Y, Y_hat):
        return -np.mean(np.sum(Y * np.log(Y_hat + 1e-9), axis=1))

    def backward(self, X, Y):

        m = X.shape[0]

        dZ2 = self.A2 - Y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.act_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ===============================
# TRAINING FUNCTION
# ===============================
def train_model(model, X, Y, X_test, Y_test, epochs=20):

    losses, train_acc, test_acc = [], [], []

    for epoch in range(epochs):

        Y_hat = model.forward(X)
        loss = model.loss(Y, Y_hat)
        model.backward(X, Y)

        losses.append(loss)

        train_pred = model.predict(X)
        test_pred = model.predict(X_test)

        train_acc.append(np.mean(train_pred == np.argmax(Y, axis=1)))
        test_acc.append(np.mean(test_pred == np.argmax(Y_test, axis=1)))

        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    return losses, train_acc, test_acc

# ===============================
# EXPERIMENTS
# ===============================
activations = ["sigmoid", "tanh", "relu", "leaky_relu"]
initializers = ["zero", "random", "xavier"]

input_size = 784
hidden_size = 128
output_size = 10

for act in activations:
    for init in initializers:

        print(f"\nTraining: {act} + {init}")

        model = NeuralNetwork(input_size, hidden_size,
                              output_size, act, init, lr=0.05)

        losses, train_acc, test_acc = train_model(
            model, x_train, y_train_oh, x_test, y_test_oh, epochs=20)

        # Plot Loss
        plt.plot(losses)
        plt.title(f"Loss vs Epochs ({act}, {init})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

        # Plot Accuracy
        plt.plot(train_acc, label="Train")
        plt.plot(test_acc, label="Test")
        plt.title(f"Accuracy vs Epochs ({act}, {init})")
        plt.legend()
        plt.show()

        print("Final Test Accuracy:", test_acc[-1])
