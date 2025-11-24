
"""
synthetic_data_agent.py
Clean synthetic data generator that uses metadata.json produced by column_selector_agent.

It reads stats for each selected column and generates a synthetic CSV with the same schema.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


def _truncated_normal(mean: float, std: float, min_v: float, max_v: float, size: int) -> np.ndarray:
    if std <= 0 or mean is None:
        return np.full(size, mean if mean is not None else 0.0)
    a, b = (min_v - mean) / std, (max_v - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def generate_from_metadata(
    metadata_path: str,
    n_rows: int = 500_000,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a synthetic dataset using the metadata.json created by column_selector_agent.

    Parameters
    ----------
    metadata_path : str
        Path to metadata.json (must contain "__target__" and column stats).
    n_rows : int
        Number of rows to generate.
    output_csv : str, optional
        If given, the synthetic dataframe is also written to this path.

    Returns
    -------
    pd.DataFrame
    """
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)

    target_col = meta.get("__target__")
    if not target_col:
        raise ValueError("metadata.json does not contain '__target__'")

    # Remove special key from feature list
    col_info = {k: v for k, v in meta.items() if k != "__target__"}

    rng = np.random.default_rng(seed=42)
    data: Dict[str, Any] = {}

    for col, info in col_info.items():
        col_type = info.get("type", "numeric")

        if col_type == "numeric":
            min_v = info.get("min", 0) or 0.0
            max_v = info.get("max", 1) or (min_v + 1.0)
            mean = info.get("mean", (min_v + max_v) / 2.0)
            # simple fallback std: 20% of range
            std = info.get("std", max((max_v - min_v) * 0.2, 1e-3))

            vals = _truncated_normal(float(mean), float(std), float(min_v), float(max_v), n_rows)
            data[col] = np.round(vals, 3)

        elif col_type == "categorical":
            top_values = info.get("top_values") or ["A", "B"]
            data[col] = rng.choice(top_values, size=n_rows)

        else:
            # fallback: treat as categorical
            top_values = info.get("top_values") or ["A", "B"]
            data[col] = rng.choice(top_values, size=n_rows)

    df = pd.DataFrame(data)

    # If target column was one of the selected columns, just keep as-is.
    # Otherwise, create a simple binary target correlated with the first numeric feature.
    if target_col in df.columns:
        pass
    else:
        # Simple heuristic target based on first numeric column
        num_cols = [c for c, info in col_info.items() if info.get("type") == "numeric"]
        if num_cols:
            key = num_cols[0]
            threshold = float(df[key].median())
            df[target_col] = (df[key] > threshold).astype(int)
        else:
            df[target_col] = rng.integers(0, 2, size=n_rows)

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df
