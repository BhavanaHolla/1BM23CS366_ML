import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# =========================
# STEP 1: LOAD DATASET
# =========================
data = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target   # house price

print(df.head())

# =========================
# STEP 2: FEATURES & TARGET
# =========================
X = df.drop('Target', axis=1)
y = df['Target']

# =========================
# STEP 3: TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 4: FEATURE SCALING (IMPORTANT FOR KNN)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# STEP 5: KNN MODEL
# =========================
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# =========================
# STEP 6: PREDICTION
# =========================
predictions = knn.predict(X_test_scaled)

# =========================
# STEP 7: METRICS
# =========================
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)

# =========================
# STEP 8: VISUALIZATION
# =========================
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")
plt.show()

# =========================
# STEP 9: TRY DIFFERENT K
# =========================
print("\nK Comparison:")
for k in [3, 5, 7]:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print(f"K={k}, MSE={mean_squared_error(y_test, preds)}")
