import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("breast_cancer.csv")

# Drop ID column if exists
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# Convert diagnosis to binary: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Select features (choose 2 for visualization)
features = ['radius_mean', 'texture_mean']
X = df[features].values  # shape (m, 2)
Y = df['diagnosis'].values.reshape(1, -1)  # shape (1, m)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T  # shape (2, m)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled.T, Y.T, test_size=0.25, random_state=42)
X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.T, Y_test.T

# Save for model
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

print("✅ Preprocessing complete! Saved X/Y splits as .npy files.")
