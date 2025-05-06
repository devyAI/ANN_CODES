import numpy as np

class Perceptron:
    def __init__(self, lr=1, epochs=100):
        self.W = None
        self.epochs = epochs
        self.lr = lr

    def predict(self, x):
        x_with_bias = np.insert(x, 0, 1)
        z = self.W.dot(x_with_bias)
        return 1 if z >= 0 else 0

    def train(self, X, y):
        self.W = np.zeros(X.shape[1] + 1)
        for _ in range(self.epochs):
            for i in range(len(y)):
                x = X[i]
                y_pred = self.predict(x)
                error = y[i] - y_pred
                if error != 0:
                    x_with_bias = np.insert(x, 0, 1)
                    self.W = self.W + self.lr * error * x_with_bias
        return self

def ascii_to_binary_vector(ascii_val):
    return np.array(list(np.binary_repr(ascii_val, width=7))).astype(int)

def main():
    digits = list(range(10))
    X = np.array([ascii_to_binary_vector(ord(str(d))) for d in digits])
    y = np.array([0 if d % 2 == 0 else 1 for d in digits])  # even = 0, odd = 1

    perceptron = Perceptron(epochs=100)
    perceptron.train(X, y)
    print("Perceptron trained successfully!\n")

    for i in digits:
        ascii_val = ord(str(i))
        x = ascii_to_binary_vector(ascii_val)
        prediction = perceptron.predict(x)
        print(f"Digit: {i}, ASCII: {ascii_val}, Prediction: {'even' if prediction == 0 else 'odd'}")

if __name__ == "__main__":
    main()