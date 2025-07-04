import numpy as np
import matplotlib.pyplot as plt

# Load and shuffle data
data = np.loadtxt("students.csv", delimiter=",", skiprows=1)
np.random.shuffle(data)

X = data[:, 0:2].T
Y = data[:, 2].reshape(1, -1)
m = X.shape[1]
split_index = int(m * 0.75)
X_train, Y_train = X[:, :split_index], Y[:, :split_index]
X_test, Y_test = X[:, split_index:], Y[:, split_index:]

# Define functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize(n):
    return np.zeros((n, 1)), 0

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    return dw, db, cost

def optimize(w, b, X, Y, lr, iterations):
    for _ in range(iterations):
        dw, db, _ = propagate(w, b, X, Y)
        w -= lr * dw
        b -= lr * db
    return w, b

def predict(w, b, X):
    A = 1 / (1 + np.exp(- (np.dot(w.T, X) + b)))
    return (A > 0.5).astype(int)

# Train
w, b = initialize(X_train.shape[0])
w, b = optimize(w, b, X_train, Y_train, 0.1, 1000)

# Plot decision boundary
x1_vals = np.linspace(0, 10, 200)
x2_vals = np.linspace(0, 10, 200)
xx1, xx2 = np.meshgrid(x1_vals, x2_vals)
grid = np.c_[xx1.ravel(), xx2.ravel()]
z = np.dot(grid, w) + b
sigmoid_vals = sigmoid(z).reshape(xx1.shape)

plt.figure(figsize=(8, 6))
contour = plt.contourf(xx1, xx2, sigmoid_vals, levels=50, cmap='RdYlGn')
plt.colorbar(contour, label='Sigmoid Output (Probability)')

plt.contour(xx1, xx2, sigmoid_vals, levels=[0.5], colors='blue', linewidths=2)

# Plot training and test data
for i in range(X_train.shape[1]):
    plt.scatter(X_train[0, i], X_train[1, i],
                color='green' if Y_train[0, i] == 1 else 'red',
                edgecolors='black', s=100, label='Train' if i == 0 else "")

for i in range(X_test.shape[1]):
    plt.scatter(X_test[0, i], X_test[1, i],
                color='green' if Y_test[0, i] == 1 else 'red',
                marker='x', s=100, label='Test' if i == 0 else "")

# ========== Step 6A: Save the model ==========
import os

# Create directory if not exists
os.makedirs("saved_model", exist_ok=True)

# Save weights and bias
np.save("saved_model/weights.npy", w)
np.save("saved_model/bias.npy", b)

print("Model saved to 'saved_model/' folder.")

# ========== Step 6B: Load the model ==========
w_loaded = np.load("saved_model/weights.npy")
b_loaded = np.load("saved_model/bias.npy", allow_pickle=True).item()

# Predict using loaded model
Y_test_pred = predict(w_loaded, b_loaded, X_test)
print("Predictions from loaded model:", Y_test_pred)

plt.xlabel("Study Hours")
plt.ylabel("Sleep Hours")
plt.title("Logistic Regression: Sigmoid Heatmap & Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
