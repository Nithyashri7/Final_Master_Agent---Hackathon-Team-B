
# Team B Hackathon Project – Master Agent Bundle

This folder bundles all the agents **you already built** plus a new
`master_agent.py` orchestrator and a few clean helper modules.

**Your original files are kept and NOT modified**:
- `agents/data_agent.py`                ← Kaggle Data Agent (as uploaded)
- `agents/column_selector_agent.py`    ← Column Selector Agent (as uploaded)
- `agents/research_agent.py`           ← Research + Best Model Agent (as uploaded)
- `agents/data_generator_agent_original.py` ← Original Colab-style generator
- `agents/model_trainer_agent_original.py`  ← Original Colab-style trainer
- `run_data_agent_cli.py`              ← Your original CLI wrapper
- `dashboard_api_example.py`           ← Your FastAPI demo for dashboard

## New helper modules I added

- `agents/synthetic_data_agent.py`  
  - Clean **Synthetic Data Generator Agent** that uses `metadata.json`
    from the Column Selector Agent and creates a large CSV.

- `agents/model_trainer_core.py`  
  - Clean **Model Building Agent** that trains a scikit-learn model
    on a pandas DataFrame + target column and returns accuracy + path
    to the saved model.

- `agents/research_runner.py`  
  - Wrapper around `research_agent.py` that runs the full research
    pipeline for a topic (search → summarise → compare models) and
    returns a `best_model_report` string + summaries.

- `master_agent.py`  
  - The **Master Orchestrator Agent** that chains everything:

    1. Kaggle Data Agent → downloads best dataset for the topic  
    2. Column Selector Agent → picks important columns + builds `metadata.json`  
    3. Synthetic Data Agent → generates big synthetic dataset from metadata  
    4. Research Agent → finds best model across recent papers  
    5. Model Trainer Agent → trains a chosen sklearn model on synthetic data  

  - Usage (from this folder):

    ```bash
    python master_agent.py --topic "Heart Disease" --rows 500000
    ```

  - Final results are written to:

    - `master_run/<topic_sanitised>/master_summary.json`  
    - plus CSVs, models and reports inside that folder.

## Where I changed code (summary)

- I did **not** edit your original `.py` files.  
- Instead, I added new helper modules that import and reuse your code
  so you can always go back to your exact originals.
