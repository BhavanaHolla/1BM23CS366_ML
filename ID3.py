import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# PART 1: DIABETES DATASET
# =========================
print("------ Diabetes Dataset ------")

# Load dataset from URL
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Decision Tree using Entropy (ID3)
model = DecisionTreeClassifier(criterion="entropy")

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("\nID3 Classification Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))


# =========================
# PART 2: HEART DATASET
# =========================
print("\n------ Heart Dataset ------")

# Load dataset from URL
df = pd.read_csv("https://raw.githubusercontent.com/anshupandey/Machine-Learning/master/heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Decision Tree using Entropy (ID3)
model = DecisionTreeClassifier(criterion="entropy")

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("\nID3 Classification Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
