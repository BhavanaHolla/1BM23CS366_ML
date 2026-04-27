import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix



# Load dataset
df = pd.read_csv("adult.data", header=None)

# Add column names
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df.columns = columns

# Split features and target
X = df.drop("income", axis=1)
y = df["income"]


# Step 3: Handle categorical data (if any)
X = pd.get_dummies(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create model
model = RandomForestClassifier(n_estimators=100, random_state=42) 
'''RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=None,     # depth of each tree
    random_state=42
)'''

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)
# ex: model.predict([[30, 40000]])

# Step 8: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#support- no of samples
#Macro Avg-Treats both classes equally
#Weighted Avg-Gives more importance to majority class (≤50K)
