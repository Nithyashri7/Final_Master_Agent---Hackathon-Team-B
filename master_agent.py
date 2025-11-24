
"""
master_agent.py
Orchestrator that chains all agents in order:

1) Data Agent (Kaggle)           -> download dataset
2) Column Selector Agent         -> selected_columns.json + metadata.json
3) Synthetic Data Generator      -> big synthetic dataset using metadata
4) Research Agent                -> find best model from recent papers
5) Model Trainer (Model Builder) -> train sklearn model on synthetic data

You can run this file directly:

    python master_agent.py --topic "Heart Disease" --rows 500000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from agents.data_agent import DataAgent
from agents.column_selector_agent import analyze_file
from agents.synthetic_data_agent import generate_from_metadata
from agents.research_runner import run_research_pipeline
from agents.model_trainer_core import train_model_on_dataframe


# -----------------------------
# Helpers
# -----------------------------
def _find_first_csv(directory: Path) -> Path:
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csvs[0]


def _pick_sklearn_model(best_model_text: str):
    """Map Gemini text to a reasonable sklearn model.

    This is intentionally simple + robust. If nothing matches, RandomForestClassifier is used.
    """
    text = (best_model_text or "").lower()

    if "gradient boosting" in text or "gbm" in text:
        return GradientBoostingClassifier()
    if "random forest" in text or "random-forest" in text:
        return RandomForestClassifier()
    if "logistic regression" in text or "logit" in text:
        return LogisticRegression(max_iter=1000)
    if "svm" in text or "support vector" in text:
        return SVC(probability=True)

    return RandomForestClassifier()


# -----------------------------
# Main pipeline
# -----------------------------
def run_master_pipeline(topic: str, n_rows: int = 500_000, workdir: str = "master_run") -> Dict[str, Any]:
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    # 1) DATA AGENT ------------------------------------------------------
    data_out = workdir_path / "downloads"
    data_out.mkdir(exist_ok=True, parents=True)

    data_agent = DataAgent()
    candidates = data_agent.search(topic=topic, limit=20)
    if not candidates:
        raise RuntimeError("DataAgent found no datasets for topic: " + topic)

    chosen_dataset = candidates[0]   # highest score
    raw_dir = Path(data_agent.download_dataset(chosen_dataset, outdir=str(data_out)))
    csv_path = _find_first_csv(raw_dir)

    # 2) COLUMN SELECTOR AGENT ------------------------------------------
    selector_out = workdir_path / "column_selector_output"
    selector_out.mkdir(exist_ok=True, parents=True)

    selected, rejected, metadata, report_text, target_col = analyze_file(str(csv_path), keep_fraction=0.6)

    # Persist artifacts exactly like the original CLI
    (selector_out / "selected_columns.json").write_text(json.dumps(selected, indent=4))
    (selector_out / "metadata.json").write_text(json.dumps(metadata, indent=4))
    (selector_out / "report.txt").write_text(report_text, encoding="utf-8")

    metadata_path = selector_out / "metadata.json"

    # 3) SYNTHETIC DATA GENERATOR ---------------------------------------
    synthetic_out_dir = workdir_path / "synthetic_data"
    synthetic_out_dir.mkdir(exist_ok=True, parents=True)
    synthetic_csv = synthetic_out_dir / "synthetic_dataset.csv"

    synthetic_df = generate_from_metadata(str(metadata_path), n_rows=n_rows, output_csv=str(synthetic_csv))

    # 4) RESEARCH AGENT --------------------------------------------------
    research_result = run_research_pipeline(topic)
    best_model_text = research_result.get("best_model_report") or ""

    # 5) MODEL TRAINER / MODEL BUILDER ----------------------------------
    model = _pick_sklearn_model(best_model_text)
    train_result = train_model_on_dataframe(
        df=synthetic_df,
        target_column=metadata.get("__target__", target_col),
        model=model,
        save_dir=str(workdir_path / "models")
    )

    # SUMMARY JSON ------------------------------------------------------
    summary = {
        "topic": topic,
        "raw_dataset_dir": str(raw_dir),
        "raw_csv_path": str(csv_path),
        "column_selector": {
            "target_column": target_col,
            "selected_columns": selected,
            "output_dir": str(selector_out),
        },
        "synthetic_data": {
            "rows": int(len(synthetic_df)),
            "csv_path": str(synthetic_csv),
        },
        "research": {
            "papers_found": research_result.get("papers_found"),
            "relevant_papers": research_result.get("relevant_papers"),
            "best_model_report": best_model_text,
        },
        "model_training": train_result,
    }

    (workdir_path / "master_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Master Agent Orchestrator (Team B)")
    parser.add_argument("--topic", required=True, help="Problem topic, e.g. 'Heart Disease'")
    parser.add_argument("--rows", type=int, default=500_000, help="Number of synthetic rows to generate")
    parser.add_argument("--workdir", default="master_run", help="Working directory for artifacts")
    args = parser.parse_args()

    summary = run_master_pipeline(topic=args.topic, n_rows=args.rows, workdir=args.workdir)
    print("\n=== MASTER PIPELINE SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
