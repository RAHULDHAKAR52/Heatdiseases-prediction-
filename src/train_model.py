# ==============================
# 1. Import Libraries
# ==============================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# 2. Load Dataset
# ==============================

df = pd.read_csv("/Users/rahuldhakad/Desktop/Acadamic_Risk_Project/project_minor/Data/student_academic_risk_dataset_5000.csv")

print("Dataset Shape:", df.shape)


# ==============================
# 3. Data Cleaning
# ==============================

# remove duplicates
df = df.drop_duplicates()

# check missing values
print(df.isnull().sum())


# ==============================
# 4. Encode Categorical Variables
# ==============================

le = LabelEncoder()

categorical_cols = [
    "Gender",
    "Parent_Education",
    "Family_Income",
    "Internet_Access"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# ==============================
# 5. Define Features and Target
# ==============================

X = df.drop(["Student_ID", "Academic_Risk"], axis=1)

y = df["Academic_Risk"]


# ==============================
# 6. Feature Scaling
# ==============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==============================
# 7. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==============================
# 8. Train Logistic Regression Model
# ==============================

model = LogisticRegression()

model.fit(X_train, y_train)


# ==============================
# 9. Model Prediction
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 10. Model Evaluation
# ==============================

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))


# ==============================
# 11. Save Model
# ==============================

with open("model/academic_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully")