# 🧠 Logistic Regression from Scratch - Breast Cancer Classification

This project implements **logistic regression from scratch using only NumPy**, without any machine learning libraries like `scikit-learn`. We apply this model to the **Breast Cancer Wisconsin (Diagnostic) dataset** to classify whether tumors are **malignant** or **benign** based on two key features.

---

## 📊 Dataset

We use the [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), which contains 30 real-valued input features for 569 samples. For visualization and simplicity, we use the following 2 features:

- `radius_mean`
- `texture_mean`

The target variable is:
- `1` → Malignant (cancerous)
- `0` → Benign (non-cancerous)

---

## 📦 Project Structure

logistic-rg-model/
│
├── breast_cancer.csv # Original dataset (downloaded from Kaggle)
├── logistic_cancer_visual.py # Main script with model, training, plotting
├── preprocess_breast_cancer.py # Preprocessing: normalization, splitting
├── saved_model/
│ ├── weights.npy # Learned weights
│ └── bias.npy # Learned bias
├── README.md # This file



---

## 🚀 How It Works

1. **Data Preprocessing**
   - Convert labels `M` to `1`, `B` to `0`
   - Select 2 numerical features
   - Normalize features using `StandardScaler`
   - Split into 75% train and 25% test

2. **Model Training**
   - Initialize weights and bias to zeros
   - Use gradient descent to minimize the logistic loss
   - Update weights for 2000 iterations

3. **Prediction**
   - Compute sigmoid of weighted sum
   - Predict `1` if sigmoid ≥ 0.5, else `0`

4. **Evaluation**
   - Compute accuracy on test set
   - Visualize decision boundary and sigmoid output

---

## 📈 Visualization

The final plot shows:
- A **sigmoid heatmap** indicating predicted probabilities
- A **decision boundary** at probability = 0.5
- **Training points** (red/blue circles)
- **Test points** (red/blue Xs)

![Example Plot](https://github.com/gajapathi22/logistic-rg-model/assets/your-image-link.png)

---

## ✅ Accuracy

With just 2 features:
- Expected test accuracy: **90–96%**

Using all 30 features would likely improve accuracy further.

---

## 🛠️ Run the Project

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
