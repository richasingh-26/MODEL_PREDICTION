# Meta-Learning Model Recommender

Production-ready meta-learning system that learns from OpenML datasets, ranks candidate ML models probabilistically, and serves recommendations through both CLI and Streamlit.

## What It Does

- Builds a meta-dataset from multiple tabular datasets to improve model recommendation accuracy.
- Extracts advanced meta-features including kurtosis, entropy, PCA variance, sparsity, and outlier percentage.
- Evaluates candidate base learners with caching, timing, and timeout protection.
- Trains a probabilistic meta-model that returns the top 3 recommended models with confidence scores.
- Saves the trained meta-model and scaler for reuse in batch and web prediction flows.
- Exposes a clean Streamlit UI for dataset upload and interactive recommendations.

## Architecture

```text
                    +----------------------+
                    |    OpenML Datasets   |
                    +----------+-----------+
                               |
                               v
                  +---------------------------+
                  | Validation + Cleaning     |
                  | data_loader.py            |
                  +-------------+-------------+
                                |
                                v
                  +---------------------------+
                  | Meta-Feature Extraction   |
                  | features.py               |
                  +-------------+-------------+
                                |
                                v
                  +---------------------------+
                  | Base Model Benchmarking   |
                  | evaluator.py              |
                  +-------------+-------------+
                                |
                                v
                  +---------------------------+
                  | Meta-Dataset CSV          |
                  | artifacts/meta_dataset.csv|
                  +-------------+-------------+
                                |
                                v
                  +---------------------------+
                  | Meta-Model + Scaler       |
                  | predictor.py              |
                  +-------------+-------------+
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
   +-----------------------+        +--------------------------+
   | CLI / main.py         |        | Streamlit Web App        |
   +-----------------------+        +--------------------------+
```

## Project Structure

```text
src/meta_recommender/
  config.py
  data_loader.py
  evaluator.py
  features.py
  logging_utils.py
  pipeline.py
  predictor.py
main.py
streamlit_app.py
evaluate_meta_model.py
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

```bash
meta-recommender --train --openml-limit 30 --n-jobs 2
```

Artifacts produced:

- `models/meta_model.joblib`
- `models/meta_scaler.joblib`
- `artifacts/meta_dataset.csv`
- `artifacts/meta_model_metrics.json`
- `artifacts/evaluation_cache.joblib`
- `logs.txt`

## CLI Usage

```bash
python main.py --file data.csv --target target_column
```

Output includes:

- best model
- top 3 ranked models
- dataset summary

## Streamlit App

Run locally with:

```bash
streamlit run streamlit_app.py
```

Features:

- CSV upload plus bundled demo dataset mode
- hybrid analysis that compares the trained meta-model with a live holdout benchmark
- richer dataset health checks for missingness, duplicates, readiness score, and target balance
- model recommendation charts, leaderboard tables, and feature-importance diagnostics
- confusion matrix or regression fit views based on task type
- optional meta-feature profile and prediction sample table
- JSON report export and leaderboard CSV download
- loading spinner and user-facing error messages

## Performance Check

```bash
python evaluate_meta_model.py
```

This prints:

- meta-model accuracy
- meta-model top-3 accuracy

## Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Ensure `requirements.txt` is present at repo root.
3. In Streamlit Cloud, set the app entry point to `streamlit_app.py`.
4. Add a deployment link here after publishing: `DEPLOYMENT_LINK_PLACEHOLDER`

## Screenshots

- Add homepage screenshot here
- Add prediction result screenshot here

## Notes

- All failures during dataset loading, feature extraction, or model training are logged to `logs.txt`.
- Per-model training and inference times are stored in the generated meta-dataset.
- Candidate model evaluation is cached to reduce recomputation across training runs.
