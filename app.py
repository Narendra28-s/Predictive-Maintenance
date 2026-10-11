
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ai4i2020.csv")
MODEL_PATH = os.path.join(
    BASE_DIR, "predictive_maintenance_model.pkl"
)

model = joblib.load(MODEL_PATH)

# Threshold for this academic demonstration
FAILURE_THRESHOLD = 0.20


def get_dashboard_data():
    df = pd.read_csv(DATA_PATH)

    total_records = len(df)
    failures = int(df["Machine failure"].sum())
    normal = total_records - failures

    failure_percentage = round(
        failures / total_records * 100, 2
    )

    X = df.drop(columns=[
        "UDI", "Product ID", "Machine failure",
        "TWF", "HDF", "PWF", "OSF", "RNF"
    ])

    X = pd.get_dummies(
        X, columns=["Type"], drop_first=True
    )

    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_test = X_test.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "precision": round(
            precision_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "recall": round(
            recall_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "f1": round(
            f1_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        )
    }

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    feature_importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_
    }).sort_values(
        "importance", ascending=False
    )

    return {
        "total_records": total_records,
        "failures": failures,
        "normal": normal,
        "failure_percentage": failure_percentage,
        "metrics": metrics,
        "matrix": matrix.tolist(),
        "feature_labels": feature_importance[
            "feature"
        ].tolist(),
        "feature_values": feature_importance[
            "importance"
        ].round(4).tolist()
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        **get_dashboard_data()
    )


@app.route("/predict", methods=["POST"])
def predict():
    machine_type = request.form["Type"]

    data = {
        "Air temperature [K]": float(
            request.form["Air temperature [K]"]
        ),
        "Process temperature [K]": float(
            request.form["Process temperature [K]"]
        ),
        "Rotational speed [rpm]": float(
            request.form["Rotational speed [rpm]"]
        ),
        "Torque [Nm]": float(
            request.form["Torque [Nm]"]
        ),
        "Tool wear [min]": float(
            request.form["Tool wear [min]"]
        ),
        "Type_L": 0,
        "Type_M": 0
    }

    if machine_type == "L":
        data["Type_L"] = 1
    elif machine_type == "M":
        data["Type_M"] = 1

    features = pd.DataFrame([data])

    features = features.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Calculate the estimated failure probability
    probability = model.predict_proba(features)[0][1]

    risk_percentage = round(probability * 100, 2)

    # Apply the demonstration threshold
    if probability >= FAILURE_THRESHOLD:
        result = (
            f"Warning: Machine failure risk is "
            f"{risk_percentage:.2f}%"
        )
    else:
        result = (
            f"Machine is predicted to operate normally. "
            f"Failure risk: {risk_percentage:.2f}%"
        )

    return render_template(
        "index.html",
        **get_dashboard_data(),
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)
