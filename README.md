# 🧠 Logistic Regression from Scratch using NumPy

A simple machine learning project that implements **binary logistic regression** from scratch using only **NumPy**.  
This model predicts whether a student will **pass** or **fail** based on:

- 📚 Study Hours  
- 💤 Sleep Hours

---

## 📁 Project Structure

```
logistic_regression_project/
├── students.csv                 # Input dataset
├── logistic_regression.py      # Main script (train, test, visualize, save/load)
└── saved_model/                # Trained model weights
    ├── weights.npy
    └── bias.npy
```

---

## 📊 Dataset Format (`students.csv`)

```
study_hours,sleep_hours,passed
1,5,0
2,6,0
3,5,0
4,5,1
5,6,1
6,5,1
7,6,1
8,7,1
```

- `study_hours`: How many hours the student studied  
- `sleep_hours`: How many hours the student slept  
- `passed`: 1 if the student passed, 0 if failed

---

## 🧪 What This Project Does

- ✅ Implements logistic regression **without ML libraries**
- ✅ Uses **sigmoid** as activation
- ✅ Trains on real-world student data
- ✅ Splits data into **train/test**
- ✅ Visualizes the **decision boundary** and **sigmoid output**
- ✅ Calculates accuracy
- ✅ Saves and loads model weights

---

## 🚀 How to Run

1. Make sure you have Python and NumPy:
```
pip install numpy matplotlib
```

2. Place `students.csv` in the project directory.

3. Run the training and evaluation script:
```
python logistic_regression.py
```

4. The trained model will be saved to:
```
saved_model/weights.npy
saved_model/bias.npy
```

---

## 📈 Output Example

- Training accuracy: ~100%
- Test accuracy: ~100% (depends on data split)
- Visualization:
  - Green = Passed
  - Red = Failed
  - Blue Line = Decision Boundary (where prediction flips from 0 to 1)

---

## 💡 Concepts Covered

- Gradient descent
- Loss function (log-loss / binary cross-entropy)
- Feature scaling (not needed here, but discussed)
- Decision boundaries
- Train/test split
- Probability interpretation of classification

---

## 📌 Author

**Gajapathi Kikkara**  
Project guided by OpenAI’s ChatGPT  
🎓 Built from scratch for learning purposes
