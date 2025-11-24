
"""
model_trainer_core.py
Clean model-training helper (Model Building Agent).

Trains a scikit-learn model on a dataframe + target column and returns metrics.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model_on_dataframe(
    df: pd.DataFrame,
    target_column: str,
    model,
    test_size: float = 0.3,
    random_state: int = 42,
    save_dir: str = "models"
) -> Dict[str, Any]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Basic one-hot encoding for categoricals
    X_enc = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    model_name = type(model).__name__.lower()
    model_path = save_dir_path / f"trained_model_{model_name}.joblib"
    joblib.dump(model, model_path)

    return {
        "status": "success",
        "accuracy": round(float(acc), 4),
        "model_name": type(model).__name__,
        "target_column": target_column,
        "model_path": str(model_path)
    }
