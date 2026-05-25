from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/wallet_dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--model", default="models/wallet_type_model.joblib", help="Output model path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    features, labels = _load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({"pipeline": pipeline}, model_path)
    print(f"Model saved to: {model_path}")


def _load_dataset(path: Path) -> tuple[list[dict[str, float]], list[str]]:
    features: list[dict[str, float]] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "label" not in (reader.fieldnames or []):
            raise ValueError("Dataset must contain 'label' column")

        for row in reader:
            label = row.pop("label")
            if label is None:
                continue
            labels.append(label)
            features.append({k: _safe_float(v) for k, v in row.items()})

    if not features:
        raise ValueError("Dataset is empty or invalid")

    return features, labels


def _safe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()
