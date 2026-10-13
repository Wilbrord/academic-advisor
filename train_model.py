from pathlib import Path
from typing import cast

# pyright: reportMissingTypeStubs=false

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split  # type: ignore[reportUnknownVariableType]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_student_performance.csv"
MODEL_PATH = ROOT / "models" / "performance_risk_model.pkl"
FEATURES = [
    "gpa",
    "attendance_pct",
    "weekly_study_hours",
    "credits_enrolled",
    "prior_failed_courses",
]


def make_synthetic_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible teaching dataset, not real student records."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "gpa": rng.uniform(0.0, 4.0, rows).round(2),
            "attendance_pct": rng.uniform(45.0, 100.0, rows).round(1),
            "weekly_study_hours": rng.uniform(0.0, 40.0, rows).round(1),
            "credits_enrolled": rng.integers(6, 22, rows),
            "prior_failed_courses": rng.integers(0, 7, rows),
        }
    )

    log_odds = (
        -2.4
        + 1.5 * (2.5 - data["gpa"])
        + 0.04 * (75.0 - data["attendance_pct"])
        - 0.12 * (data["weekly_study_hours"] - 10.0)
        + 0.12 * (data["credits_enrolled"] - 15)
        + 0.3 * data["prior_failed_courses"]
    )
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    data["at_risk"] = rng.binomial(1, probability)
    return data


def train_and_save() -> tuple[Pipeline, float, float]:
    data = make_synthetic_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)

    x_train, x_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            data[FEATURES],
            data["at_risk"],
            test_size=0.2,
            random_state=42,
            stratify=data["at_risk"],
        ),
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)  # type: ignore[reportUnknownMemberType]

    predictions = model.predict(x_test)  # type: ignore[reportUnknownMemberType]
    risk_probabilities = model.predict_proba(  # type: ignore[reportUnknownMemberType]
        x_test
    )[:, 1]
    accuracy = float(accuracy_score(y_test, predictions))
    auc = float(roc_auc_score(y_test, risk_probabilities))
    joblib.dump(  # type: ignore[reportUnknownMemberType]
        {"model": model, "features": FEATURES, "synthetic_data": True},
        MODEL_PATH,
    )
    return model, accuracy, auc


if __name__ == "__main__":
    _, accuracy, auc = train_and_save()
    print(f"Synthetic demo accuracy: {accuracy:.3f}")
    print(f"Synthetic demo ROC AUC: {auc:.3f}")
    print(f"Dataset saved to: {DATA_PATH.relative_to(ROOT)}")
    print(f"Model saved to: {MODEL_PATH.relative_to(ROOT)}")
    print("Do not use this synthetic model for real academic decisions.")