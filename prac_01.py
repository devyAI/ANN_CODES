import numpy as np
import matplotlib.pyplot as plt

x_values = np.linspace(-5, 5, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

functions = {
    "Sigmoid": sigmoid(x_values),
    "Tanh": tanh(x_values),
    "ReLU": relu(x_values),
    "Linear": linear(x_values),
    "Leaky ReLU": leaky_relu(x_values),
    "ELU": elu(x_values)
}

fig, axes = plt.subplots(2,3, figsize=(12,8))
axes = axes.flatten()

for ax, (name, y ) in zip(axes, functions.items()):
    ax.plot(x_values, y)
    ax.set_title(name)
    ax.grid(True)

plt.tight_layout()
plt.show()