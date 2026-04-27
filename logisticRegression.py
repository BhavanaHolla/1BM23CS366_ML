# Logistic Regression - Multiclass Classification (Zoo Dataset)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# =========================
# STEP 1: LOAD DATASET FROM URL
# =========================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"

# Column names (dataset doesn't include headers)
columns = [
    "animal_name","hair","feathers","eggs","milk","airborne","aquatic",
    "predator","toothed","backbone","breathes","venomous","fins",
    "legs","tail","domestic","catsize","class_type"
]

# Load dataset
df = pd.read_csv(url, names=columns)

# Display first few rows
print(df.head())

# =========================
# STEP 2: DATA CLEANING
# =========================
# Remove animal_name because it's just a label (not useful for prediction)
df = df.drop("animal_name", axis=1)

# =========================
# STEP 3: DEFINE FEATURES & TARGET
# =========================
# X = input features
X = df.drop("class_type", axis=1)

# y = output (class label)
y = df["class_type"]

# =========================
# STEP 4: TRAIN-TEST SPLIT
# =========================
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 5: MODEL CREATION
# =========================
# Logistic Regression for multiclass classification
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# =========================
# STEP 6: PREDICTION
# =========================
# Predict class labels for test data
y_pred = model.predict(X_test)

# =========================
# STEP 7: EVALUATION METRICS
# =========================
# Accuracy: overall correctness
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix: shows correct & incorrect predictions
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report: precision, recall, F1-score
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# STEP 8: VISUALIZATION
# =========================
# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

# =========================
# EXTRA (OPTIONAL FOR VIVA)
# =========================
# Shows different classes in dataset
print("\nClasses:", model.classes_)
