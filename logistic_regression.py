import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ===== Step 1: Load and preprocess data =====
df = pd.read_csv("breast_cancer.csv")

# Drop ID column if present
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# Map 'M' → 1, 'B' → 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Choose two features for visualization
features = ['radius_mean', 'texture_mean']
X = df[features].values
Y = df['diagnosis'].values.reshape(1, -1)  # Shape: (1, m)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T  # Shape: (2, m)

# Split into train/test
m = X_scaled.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled.T, Y.T, test_size=0.25, random_state=42)
X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.T, Y_test.T

# ===== Step 2: Model functions =====
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize(n):
    return np.zeros((n, 1)), 0

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/m) * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8))
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    return dw, db, cost

def optimize(w, b, X, Y, lr, iterations):
    for i in range(iterations):
        dw, db, cost = propagate(w, b, X, Y)
        w -= lr * dw
        b -= lr * db
        if i % 200 == 0:
            print(f"Iteration {i} => Cost: {cost:.4f}")
    return w, b

def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

# ===== Step 3: Train model =====
w, b = initialize(X_train.shape[0])
w, b = optimize(w, b, X_train, Y_train, lr=0.1, iterations=2000)

# ===== Step 4: Save model =====
os.makedirs("saved_model", exist_ok=True)
np.save("saved_model/weights.npy", w)
np.save("saved_model/bias.npy", b)

print("✅ Model saved to 'saved_model/' folder.")

# ===== Step 5: Load model =====
w_loaded = np.load("saved_model/weights.npy")
b_loaded = np.load("saved_model/bias.npy", allow_pickle=True).item()

# ===== Step 6: Predict test data =====
Y_test_pred = predict(w_loaded, b_loaded, X_test)
print("Predictions:", Y_test_pred)
print("True Labels:", Y_test)

accuracy = 100 - np.mean(np.abs(Y_test_pred - Y_test)) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# ===== Step 7: Visualize decision boundary =====
x1_vals = np.linspace(X_scaled[0].min()-1, X_scaled[0].max()+1, 200)
x2_vals = np.linspace(X_scaled[1].min()-1, X_scaled[1].max()+1, 200)
xx1, xx2 = np.meshgrid(x1_vals, x2_vals)
grid = np.c_[xx1.ravel(), xx2.ravel()]
z = np.dot(grid, w_loaded) + b_loaded
sigmoid_vals = sigmoid(z).reshape(xx1.shape)

plt.figure(figsize=(8, 6))
contour = plt.contourf(xx1, xx2, sigmoid_vals, levels=50, cmap='RdYlGn')
plt.colorbar(contour, label='Sigmoid Probability')

# Draw decision boundary at 0.5
plt.contour(xx1, xx2, sigmoid_vals, levels=[0.5], colors='blue', linewidths=2)

# Plot training data
plt.scatter(X_train[0], X_train[1], c=Y_train[0], cmap='bwr', edgecolors='k', label='Train', s=60)
# Plot test data
plt.scatter(X_test[0], X_test[1], c=Y_test[0], cmap='coolwarm', marker='x', label='Test', s=80)

plt.xlabel("radius_mean (normalized)")
plt.ylabel("texture_mean (normalized)")
plt.title("Logistic Regression on Breast Cancer Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
