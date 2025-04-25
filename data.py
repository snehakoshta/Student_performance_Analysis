import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingClassifier
from sklearn.metrics import mean_squared_error
import pickle
import joblib
import numpy as np

# Load the dataset
data = pd.read_csv('C:/Users/HP/Downloads/performance/student_habits_performance.csv')  # Update with the correct file path

# Preview dataset columns
print("Dataset Columns:", data.columns)

# Encode categorical columns
gender_mapping = {'Male': 0, 'Female': 1}
part_time_job_mapping = {'No': 0, 'Yes': 1}
diet_quality_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2}
parental_education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
internet_quality_mapping = {'Poor': 0, 'Average': 1, 'Good': 2}
extracurricular_mapping = {'No': 0, 'Yes': 1}

data['gender'] = data['gender'].map(gender_mapping)
data['part_time_job'] = data['part_time_job'].map(part_time_job_mapping)
data['diet_quality'] = data['diet_quality'].map(diet_quality_mapping)
data['parental_education_level'] = data['parental_education_level'].map(parental_education_mapping)
data['internet_quality'] = data['internet_quality'].map(internet_quality_mapping)
data['extracurricular_participation'] = data['extracurricular_participation'].map(extracurricular_mapping)

# Define features and target
features = [
    'age', 'gender', 'study_hours_per_day', 'social_media_hours', 
    'netflix_hours', 'part_time_job', 'attendance_percentage', 
    'sleep_hours', 'diet_quality', 'exercise_frequency', 
    'parental_education_level', 'internet_quality', 
    'mental_health_rating', 'extracurricular_participation'
]
target = 'exam_score'

X = data[features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

# Create Accuracy Table
accuracy_data = {
    "Model": ["Decision Tree Regressor", "Random Forest Regressor"],
    "Algorithm": ["Decision Tree", "Random Forest"],
    "Mean Squared Error": [dt_mse, rf_mse]
}

accuracy_df = pd.DataFrame(accuracy_data)

# Display the Accuracy Table
print("\nModel Evaluation Metrics:")
print(accuracy_df.to_string(index=False))

# Save the trained models
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)
print("Decision Tree model saved as 'decision_tree_model.pkl'")

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
print("Random Forest model saved as 'random_forest_model.pkl'")
