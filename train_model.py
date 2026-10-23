"""Train the Smart Home IoT room-status classifier."""

from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "iot_sensor_dataset.csv"
MODEL_PATH = PROJECT_DIR / "model" / "smart_home_model.joblib"
FEATURES = ["temperature_c", "humidity_percent"]
EXPECTED_LABELS = {"Comfortable", "Too Hot", "Too Cold"}


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + ["room_status"]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )
    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="raise")
    labels = set(df["room_status"].unique())
    if labels != EXPECTED_LABELS:
        raise ValueError(
            "Dataset labels must be exactly: "
            + ", ".join(sorted(EXPECTED_LABELS))
            + f". Found: {', '.join(sorted(labels))}"
        )

    X = df[FEATURES]
    y = df["room_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
