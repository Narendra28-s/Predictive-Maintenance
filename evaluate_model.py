
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load dataset and trained model
df = pd.read_csv("ai4i2020.csv")
model = joblib.load("predictive_maintenance_model.pkl")

# Prepare input features
X = df.drop(columns=[
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

X = pd.get_dummies(X, columns=["Type"], drop_first=True)
y = df["Machine failure"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Predict using the trained model
X_test = X_test.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

predictions = model.predict(X_test)

# Calculate performance metrics
print("MODEL PERFORMANCE")
print("-----------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, predictions, zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, predictions, zero_division=0) * 100, 2), "%")
print("F1-score:", round(f1_score(y_test, predictions, zero_division=0) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
