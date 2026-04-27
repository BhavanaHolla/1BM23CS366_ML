# Import required libraries
from sklearn.model_selection import train_test_split   # to split dataset
from sklearn.ensemble import AdaBoostClassifier        # AdaBoost model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# STEP 1: LOAD DATASET FROM URL
# =========================
# Using Iris dataset from online source
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# =========================
# STEP 2: DEFINE FEATURES & TARGET
# =========================
# Features (input variables)
X = df.drop("species", axis=1)

# Target (output class)
y = df["species"]

# Store feature names (used later for importance plot)
feature_names = X.columns

# =========================
# STEP 3: TRAIN-TEST SPLIT
# =========================
# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 4: CREATE MODEL
# =========================
# AdaBoost classifier with:
# n_estimators = number of weak learners
# learning_rate = contribution of each learner
model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# =========================
# STEP 5: TRAIN MODEL
# =========================
# Model learns from training data
model.fit(X_train, y_train)

# =========================
# STEP 6: PREDICTION
# =========================
# Predict class labels for test data
y_pred = model.predict(X_test)

# =========================
# STEP 7: EVALUATION
# =========================
# Accuracy → overall correctness
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Classification report → precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix → actual vs predicted
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =========================
# STEP 8: VISUALIZATION (CONFUSION MATRIX)
# =========================
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

# =========================
# STEP 9: FEATURE IMPORTANCE
# =========================
# Shows which features contributed most to prediction
importance = pd.Series(model.feature_importances_, index=feature_names)

print("\nFeature Importance:")
print(importance)

# Plot feature importance
importance.plot(kind='bar')
plt.title("AdaBoost Feature Importance")
plt.ylabel("Importance Score")
plt.show()
