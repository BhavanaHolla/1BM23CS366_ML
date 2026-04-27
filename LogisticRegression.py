import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# 1. Dataset
data = {
'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8],
'Attendance': [50, 55, 60, 65, 70, 75, 80, 85],
'AssignmentMarks': [20, 25, 30, 35, 40, 45, 50, 55],
'ExamScore': [30, 35, 40, 50, 55, 65, 75, 85]
}
9

df = pd.DataFrame(data)
# 2. Features and Target
X = df[['StudyHours', 'Attendance', 'AssignmentMarks']]
y = df['ExamScore']
# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=42
)
# 4. Train Multilinear Model
model = LinearRegression()
model.fit(X_train, y_train)
# 5. Predictions on Test Data
predictions = model.predict(X_test)

print("Predicted Scores:", predictions)
print("MSE:", mean_squared_error(y_test, predictions))
# 6. Predict for New Student (FIXED - No Warning)
new_student = pd.DataFrame(
[[6, 78, 48]],
columns=['StudyHours', 'Attendance', 'AssignmentMarks']
)
predicted_score = model.predict(new_student)
print("Predicted Exam Score for new student:", predicted_score)
