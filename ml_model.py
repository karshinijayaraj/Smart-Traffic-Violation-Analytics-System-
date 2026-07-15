"""
ml_model.py
-----------
Trains a machine-learning model to classify each traffic violation's
risk level (Low / Medium / High). This score can drive prioritisation:
e.g. auto-flag High-risk cases for immediate police follow-up, or feed
a repeat-offender watchlist.

Model: RandomForestClassifier (robust to mixed categorical/numeric
features, gives interpretable feature importances - good for a
traffic-authority stakeholder audience).

Run:  python3 src/ml_model.py
Outputs:
  - outputs/models/risk_model.joblib   (trained pipeline)
  - outputs/plots/feature_importance.png
  - outputs/plots/confusion_matrix.png
  - prints classification report to stdout
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = "data/traffic_violations.csv"
MODEL_PATH = "outputs/models/risk_model.joblib"

NUMERIC_FEATURES = ["speed_kmph", "speed_limit_kmph", "vehicle_age_years",
                     "prior_violations", "hour"]
CATEGORICAL_FEATURES = ["vehicle_type", "violation_type", "weather",
                         "signal_status", "day_of_week", "location"]
TARGET = "risk_level"


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
    )
    return Pipeline(steps=[("prep", preprocessor), ("clf", model)])


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Risk Level - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance plot
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    importances = pipe.named_steps["clf"].feature_importances_

    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=imp_df, y="feature", x="importance", color="#3b82f6")
    plt.title("Top 15 Feature Importances - Violation Risk Model")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")
    print("Plots saved -> outputs/plots/confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()
