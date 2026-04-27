Linear Regression
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
data = {
'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8],
'Attendance': [50, 55, 60, 65, 70, 75, 80, 85],
'AssignmentMarks': [20, 25, 30, 35, 40, 45, 50, 55],
'ExamScore': [30, 35, 40, 50, 55, 65, 75, 85]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)
# 2. FEATURES AND TARGET
X = df[['StudyHours', 'Attendance', 'AssignmentMarks']]
y = df['ExamScore']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
8

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lr = lin_reg.predict(X_test_scaled)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("RMSE:", rmse_lr)
print("MAE:", mae_lr)
print("R2 Score:", r2_lr)
