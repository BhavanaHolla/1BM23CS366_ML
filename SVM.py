# Titanic Survival Prediction using SVM (All-in-One)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# STEP 1: Load dataset
url = "https://raw.githubusercontent.com/rolandmueller/titanic/main/titanic3.csv"
df = pd.read_csv(url)
# STEP 2: Select relevant columns
df = df[['pclass','sex','age','fare','survived']]
# STEP 3: Handle missing values
df = df.dropna()
# STEP 4: Convert categorical to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
# STEP 5: Define features and target
X = df[['pclass','sex','age','fare']]
y = df['survived']
# STEP 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# STEP 7: Feature scaling (VERY IMPORTANT for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
22

X_test = scaler.transform(X_test)
# STEP 8: Train SVM model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
# STEP 9: Predictions
y_pred = model.predict(X_test)
# STEP 10: Evaluation
print("Accuracy:", model.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
