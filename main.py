import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
print("Loading dataset...")
train_data = np.loadtxt("datasets/mnist_train.csv", delimiter=",")
test_data = np.loadtxt("datasets/mnist_test.csv", delimiter=",")

# Convert x to array of [0.1; 1]
fac = 0.99 / 255
train_x = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_x = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

# Convert y to one hot
lr = np.arange(10)
train_y = (lr==train_labels).astype(np.float64)
test_y = (lr==test_labels).astype(np.float64)

train_v = list(zip(train_x, train_y))
test_v = list(zip(train_x, train_y))

structure = []
controls = []

def init_layer(nodes):
    """Initialize a layer"""
    if structure:
        controls.append(np.random.normal(
            scale=structure[-1] ** -0.5, # @LeCun et al.
            size=(structure[-1] + 1, nodes)
        ))
    structure.append(nodes)

def relu(x):
    """Compute rectified linear unit"""
    return np.maximum(x, 0)

def relu_deriv(x):
    """Compute rectified linear unit derivative"""
    return x > 0

def softmax(x):
    """Compute softmax"""
    exps = np.exp(x)
    return exps / np.sum(exps)

def softmax_deriv(x):
    """Compute softmax"""
    s = softmax(x)
    return s * (1 - s)

def forward_prop(i):
    """Compute forward propagation and return the node values before ReLU transformation"""
    layers = [i]
    for c in controls:
        v = np.matmul(np.append(i, 1), c)
        layers.append(v)
        i = relu(v)

    output = softmax(layers[-1])

    return layers, output

def backward_prop(layers, output, expected):
    """Compute backward propagation and return the gradients"""

    # Compute derivative of loss function with respect to weights and biases [gradients]
    deriv = softmax_deriv(layers[-1]) * (expected - output) / len(output)
    gradients = []

    for i in range(len(controls)):
        gradient = np.append(relu(layers[-2 - i]), 1)[:, None] * deriv[None,:]

        deriv = relu_deriv(layers[-2 - i]) * np.matmul(
            deriv,
            np.transpose(controls[-1 - i][:-1])
        )

        gradients.append(gradient)

    return gradients[::-1]

LEARNING_RATE = 0.02
SAMPLE_SIZE = 1

def init():
    """Initialize the program"""
    structure.clear()
    controls.clear()

    # 1 input layer
    init_layer(28 * 28)

    # 1 hidden layers
    init_layer(10)
    init_layer(10)

    # 1 output layer
    init_layer(10)

    for c in controls:
        print(c.shape)

def train():
    """Train the model [1 iteration]"""
    gradients = [np.zeros(controls[i].shape) for i in range(len(controls))]

    for (x, y) in random.sample(train_v, SAMPLE_SIZE):
        layers, output = forward_prop(x)

        for (i, gradient) in enumerate(backward_prop(layers, output, y)):
            gradients[i] += gradient

    for (i, gradient) in enumerate(gradients):
        controls[i] += (LEARNING_RATE / SAMPLE_SIZE) * gradient

def test():
    """Test the model"""
    terror = 0
    correct = 0
    for (inp, expected) in test_v:
        _, output = forward_prop(np.array([inp]))
        if np.argmax(output) == np.argmax(expected):
            correct += 1
        terror += np.sum(np.square(output - expected))

    terror /= len(test_v)
    correct /= len(test_v)

    print(f"ACCURACY: {correct:4f} - MSE {terror:4f}")

    return terror

def visualize():
    """Visualize the results"""
    _, ax = plt.subplots(nrows=4, ncols=4)

    for (i, (x, y)) in enumerate(list(random.sample(test_v, 16))):
        layers, output = forward_prop(x)

        img_src = x.reshape((28,28))
       
        if np.argmax(y) == np.argmax(output):
            ax.ravel()[i].imshow(img_src, cmap="Greens")
        else:
            ax.ravel()[i].imshow(img_src, cmap="Reds")
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(f"{np.argmax(output)}")
    plt.tight_layout()
    plt.show()


# Main code

init()

while True:
    for _ in range(10):
        for _ in range(40000):
            train()
        test()
    visualize()
