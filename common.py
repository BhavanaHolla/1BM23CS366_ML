#Load ANY dataset safely
import pandas as pd

# Load dataset (change filename)
df = pd.read_csv("dataset.csv")

# Quick inspection (VERY IMPORTANT)
print(df.head())
print(df.info())

#Define Features & Target (YOU MUST DECIDE)
# Replace 'target_column' with actual column name
X = df.drop('target_column', axis=1)
y = df['target_column']

#3. Handle Categorical Data
X = pd.get_dummies(X)

#4. Handle Missing Values (IMPORTANT)
X = X.fillna(X.mean())

#5. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#6. Feature Scaling (VERY IMPORTANT FOR SVM)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#7. Apply SVM
from sklearn.svm import SVC

model = SVC(kernel='rbf')   # change kernel if needed
model.fit(X_train, y_train)

#8. Predictions
y_pred = model.predict(X_test)

#9. Metrics (ALWAYS WRITE THESE)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#10. Visualization (QUICK + IMPRESSIVE)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

'''datasets:
kaggle
uci machine learning repo
google dataset search

-----
data.gov.in
world bank open data
WHO

----

hugging face
openML
-----

papers

csv converters online
'''











