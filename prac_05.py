import numpy as np


# Define two pairs of bipolar vectors
X = np.array([[1, -1, 1], [-1, 1, -1]])  # Input patterns (X)
Y = np.array([[1, 1, -1], [-1, -1, 1]])  # Associated output patterns (Y)


# Compute the weight matrix using Hebbian Learning
W = np.zeros((X.shape[1], Y.shape[1]))  # Initialize weight matrix
for i in range(len(X)):
    W += np.outer(X[i], Y[i])  # Hebbian rule: W = sum(X * Y^T)


# Function for bidirectional recall
def recall(input_vec, direction="forward"):
    if direction == "forward":
        return np.sign(np.dot(input_vec, W))   # Forward recall: X → Y
    else:
        return np.sign(np.dot(input_vec, W.T)) # Backward recall: Y → X


# Test recall in both directions
test_X = X[0]  # Select first input pattern
test_Y = Y[0]  # Select first output pattern


recalled_Y = recall(test_X, "forward")   # X to Y
recalled_X = recall(test_Y, "backward")  # Y to X


# Display results
print("Original X:", test_X)
print("Recalled Y:", recalled_Y)
print("Original Y:", test_Y)
print("Recalled X:", recalled_X)