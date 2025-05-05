import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Perceptron:
    def __init__(self, lr=1, epochs=100):
        self.W = None  # Initialize weights later when we know input size
        self.epochs = epochs
        self.lr = lr
    
    def predict(self, x):
        x_with_bias = np.insert(x, 0, 1)  # Add bias term
        z = self.W.dot(x_with_bias)
        return 1 if z >= 0 else 0
    
    def train(self, X, y):
        # Initialize weights (including bias weight)
        self.W = np.zeros(X.shape[1] + 1)
        
        for _ in range(self.epochs):
            for i in range(len(y)):
                x = X[i]
                y_pred = self.predict(x)
                error = y[i] - y_pred
                
                # Update weights if prediction was wrong
                if error != 0:
                    x_with_bias = np.insert(x, 0, 1)
                    self.W = self.W + self.lr * error * x_with_bias
        return self

def main():
    digits = np.array([[i] for i in range(10)])
    
    
    encoder = OneHotEncoder(sparse_output=False)
    X = encoder.fit_transform(digits)
    
   
    y = [0 if i % 2 == 0 else 1 for i in range(10)]
    
    
    perceptron = Perceptron(epochs=100)
    perceptron.train(X, y)
    print("Perceptron trained successfully!")
    
    for i in range(10):
        x = encoder.transform([[i]])
        prediction = perceptron.predict(x[0])
        print(f"Number {i} is {'even' if prediction == 0 else 'odd'}")
    
if __name__ == "__main__":
    main()