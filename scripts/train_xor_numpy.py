import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Architecture: 2 -> 3 -> 1
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights
np.random.seed(42)
w0 = np.random.uniform(-1, 1, (input_size, hidden_size))
b0 = np.random.uniform(-1, 1, (1, hidden_size))
w1 = np.random.uniform(-1, 1, (hidden_size, output_size))
b1 = np.random.uniform(-1, 1, (1, output_size))

# Training
for i in range(epochs):
    # Forward
    l0 = X
    l1 = sigmoid(np.dot(l0, w0) + b0)
    l2 = sigmoid(np.dot(l1, w1) + b1)
    
    # Backprop
    l2_error = y - l2
    l2_delta = l2_error * sigmoid_derivative(l2)
    
    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid_derivative(l1)
    
    w1 += l1.T.dot(l2_delta) * learning_rate
    b1 += np.sum(l2_delta, axis=0, keepdims=True) * learning_rate
    w0 += l0.T.dot(l1_delta) * learning_rate
    b0 += np.sum(l1_delta, axis=0, keepdims=True) * learning_rate

# Print results
print("Results:")
l1 = sigmoid(np.dot(X, w0) + b0)
l2 = sigmoid(np.dot(l1, w1) + b1)
print(l2)

# Convert to fixed point (16.16)
scale = 65536

def to_fixed(val):
    return int(val * scale)

print("\nWeights for VHDL:")
print("-- Layer 0 (Hidden)")
for i in range(hidden_size):
    print(f"-- Neuron {i}")
    for j in range(input_size):
        print(f"w{j}: {to_fixed(w0[j][i])}")
    print(f"bias: {to_fixed(b0[0][i])}")

print("\n-- Layer 1 (Output)")
for i in range(output_size):
    print(f"-- Neuron {i}")
    for j in range(hidden_size):
        print(f"w{j}: {to_fixed(w1[j][i])}")
    print(f"bias: {to_fixed(b1[0][i])}")
