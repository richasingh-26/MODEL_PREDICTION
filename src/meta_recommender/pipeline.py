"""End-to-end pipeline to construct, train, evaluate, and serve the meta-learning system."""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .config import (
    DATASET_PROCESS_TIMEOUT_SECONDS,
    DATASET_RETRY_COUNT,
    DEBUG_DATASET_LIMIT,
    DEFAULT_N_JOBS,
    DEFAULT_OPENML_SIZE,
    DEFAULT_TOP_K,
    META_DATASET_PATH,
)
from .evaluator import benchmark_against_automl, evaluate_models
from .features import clean_X, detect_task_type, extract_meta_features
from .logging_utils import log_exception, setup_logging
from .predictor import MetaModelPredictor
from .runtime_utils import HardTimeoutError, run_with_hard_timeout

logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message=r"'meta_recommender\.pipeline' found in sys\.modules after import of package 'meta_recommender'",
    category=RuntimeWarning,
)

if TYPE_CHECKING:
    from .data_loader import DatasetBundle


@dataclass
class MetaRecord:
    """Single meta-learning training record built from one source dataset."""

    dataset_id: int
    dataset_name: str
    best_model: str
    task_type: str
    n_rows: int
    n_cols: int
    model_scores: dict[str, float]
    model_timings: dict[str, dict[str, float | bool | str]]
    meta_features: dict[str, float]


def _summarize_dataset(df: pd.DataFrame) -> dict[str, int | str | float]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 4),
    }


def process_dataset_bundle(bundle: "DatasetBundle", debug: bool = False) -> MetaRecord | None:
    """Process one OpenML dataset into a meta-learning training record."""
    dataset_start = perf_counter()
    try:
        logger.info("Dataset %s (%s) | start", bundle.name, bundle.dataset_id)

        logger.info("Dataset %s | cleaning features", bundle.name)
        X = clean_X(bundle.X)
        y = bundle.y

        logger.info("Dataset %s | extracting meta-features", bundle.name)
        meta = extract_meta_features(X, y)

        logger.info("Dataset %s | benchmarking candidate models", bundle.name)
        eval_result = evaluate_models(
            X,
            y,
            dataset_key=str(bundle.dataset_id),
            debug=debug,
            dataset_name=bundle.name,
        )
        if not eval_result.best_model:
            logger.info("Dataset %s | skipped due to missing valid model scores.", bundle.name)
            return None

        elapsed = perf_counter() - dataset_start
        logger.info("Dataset %s | complete in %.2fs, best=%s", bundle.name, elapsed, eval_result.best_model)
        return MetaRecord(
            dataset_id=bundle.dataset_id,
            dataset_name=bundle.name,
            best_model=eval_result.best_model,
            task_type=eval_result.task_type,
            n_rows=len(X),
            n_cols=X.shape[1],
            model_scores=eval_result.scores,
            model_timings=eval_result.timings,
            meta_features=meta,
        )
    except Exception as exc:  # noqa: BLE001
        log_exception(logger, "dataset processing", bundle.name, exc)
        return None


def _process_dataset_bundle_worker(bundle: "DatasetBundle", debug: bool = False) -> MetaRecord | None:
    setup_logging(debug=debug)
    return process_dataset_bundle(bundle, debug=debug)


def _process_dataset_with_retries(
    bundle: "DatasetBundle",
    *,
    debug: bool,
    timeout_seconds: int,
    retries: int,
) -> MetaRecord | None:
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        logger.info(
            "Dataset %s (%s) | processing attempt %s/%s",
            bundle.name,
            bundle.dataset_id,
            attempt,
            attempts,
        )
        try:
            timed = run_with_hard_timeout(
                _process_dataset_bundle_worker,
                kwargs={"bundle": bundle, "debug": debug},
                timeout_seconds=timeout_seconds,
                stage_name=f"dataset processing {bundle.dataset_id}",
            )
            logger.info(
                "Dataset %s (%s) | subprocess finished in %.2fs",
                bundle.name,
                bundle.dataset_id,
                timed.elapsed_seconds,
            )
            return timed.value
        except HardTimeoutError:
            logger.warning(
                "Dataset %s (%s) | timeout on attempt %s/%s after %ss",
                bundle.name,
                bundle.dataset_id,
                attempt,
                attempts,
                timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt < attempts:
                logger.warning(
                    "Dataset %s (%s) | retrying after failure on attempt %s/%s: %s",
                    bundle.name,
                    bundle.dataset_id,
                    attempt,
                    attempts,
                    exc,
                )
            else:
                log_exception(logger, "dataset processing", bundle.name, exc)
    return None


def _meta_records_to_frame(records: list[MetaRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **asdict(record),
                **{f"meta_{key}": value for key, value in record.meta_features.items()},
            }
            for record in records
        ]
    )


def run_training_pipeline(
    openml_limit: int = DEFAULT_OPENML_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
    *,
    debug: bool = False,
) -> tuple[MetaModelPredictor | None, pd.DataFrame]:
    """Build a meta-dataset from OpenML datasets and train the ranking-first meta-model."""
    setup_logging(debug=debug)
    from .data_loader import load_openml_datasets

    effective_limit = min(openml_limit, DEBUG_DATASET_LIMIT) if debug else openml_limit
    effective_n_jobs = 1
    if n_jobs > 1:
        logger.warning(
            "Dataset-level multiprocessing is forced to 1 worker for stability on Windows/OpenML cache workloads. "
            "Hard subprocess timeouts still isolate each dataset and model."
        )
    if debug:
        logger.info("Debug mode enabled: limiting training to %s datasets and one candidate model.", effective_limit)

    logger.info(
        "Training pipeline start | openml_limit=%s effective_limit=%s requested_n_jobs=%s effective_n_jobs=%s debug=%s",
        openml_limit,
        effective_limit,
        n_jobs,
        effective_n_jobs,
        debug,
    )

    records: list[MetaRecord] = []
    training_start = perf_counter()
    progress = tqdm(total=effective_limit, desc="Training datasets", unit="dataset")

    try:
        for index, bundle in enumerate(load_openml_datasets(limit=effective_limit, debug=debug), start=1):
            progress.set_postfix_str(f"dataset={bundle.name}")
            logger.info("Training dataset %s/%s | %s (%s)", index, effective_limit, bundle.name, bundle.dataset_id)
            record = _process_dataset_with_retries(
                bundle,
                debug=debug,
                timeout_seconds=DATASET_PROCESS_TIMEOUT_SECONDS,
                retries=DATASET_RETRY_COUNT,
            )
            progress.update(1)
            if record is not None:
                records.append(record)
    finally:
        progress.close()

    if not records:
        logger.error("No valid datasets produced a meta-record.")
        return None, pd.DataFrame()

    records.sort(key=lambda record: record.dataset_id)
    df = _meta_records_to_frame(records)
    META_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing meta-dataset with %s records to %s", len(df), META_DATASET_PATH)
    df.to_csv(META_DATASET_PATH, index=False)

    logger.info("Training meta-model on %s records.", len(df))
    predictor = MetaModelPredictor.train(df)
    predictor.save()
    elapsed = perf_counter() - training_start
    logger.info("Training pipeline complete in %.2fs using %s successful datasets.", elapsed, len(records))
    return predictor, df


def recommend_for_dataframe(
    df: pd.DataFrame,
    predictor: MetaModelPredictor,
    target_column: str | None = None,
    mode: str = "accurate",
    benchmark_automl: bool = False,
) -> dict[str, Any]:
    """Generate cost-aware model recommendations for an in-memory DataFrame."""
    if df.empty:
        raise ValueError("Input dataset is empty.")

    logger.info("Recommendation start | mode=%s target=%s rows=%s cols=%s", mode, target_column, len(df), df.shape[1])
    working_df = df.copy()
    if target_column:
        if target_column not in working_df.columns:
            raise ValueError(f"Target column '{target_column}' was not found.")
        y = working_df.pop(target_column)
    else:
        y = pd.Series([0] * len(working_df), name="target")

    meta_features = extract_meta_features(working_df, y)
    top_k = predictor.predict_top_k_models(meta_features, k=DEFAULT_TOP_K, mode=mode)
    comparisons = predictor.compare_modes(meta_features, k=DEFAULT_TOP_K)
    explanation = predictor.get_explanation(meta_features, mode=mode)
    problem_type = detect_task_type(y)
    automl_benchmark = None
    if benchmark_automl and target_column is not None:
        try:
            automl_benchmark = benchmark_against_automl(
                working_df,
                y,
                recommended_model=top_k[0][0],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AutoML benchmark skipped: %s", exc)

    logger.info("Recommendation complete | best_model=%s", top_k[0][0])
    return {
        "best_model": top_k[0][0],
        "mode": mode,
        "top_3": [{"model": model_name, "probability": probability} for model_name, probability in top_k],
        "recommendation_modes": comparisons,
        "meta_features": meta_features,
        "dataset_summary": _summarize_dataset(df),
        "dataset_diagnostics": {
            "problem_type": problem_type,
            "target_column": target_column,
            "n_unique_target": int(y.nunique(dropna=True)),
            "missing_target_values": int(y.isna().sum()),
        },
        "problem_type": problem_type,
        "target_column": target_column,
        "explanation": explanation,
        "meta_model_metrics": predictor.metrics,
        "automl_benchmark": automl_benchmark,
    }


def recommend_for_csv(
    csv_path: str,
    predictor: MetaModelPredictor,
    target_column: str | None = None,
    mode: str = "accurate",
    benchmark_automl: bool = False,
) -> dict[str, Any]:
    """Run recommendation on a user-provided CSV file."""
    logger.info("Loading CSV for prediction: %s", csv_path)
    df = pd.read_csv(csv_path)
    return recommend_for_dataframe(
        df,
        predictor,
        target_column=target_column,
        mode=mode,
        benchmark_automl=benchmark_automl,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training and prediction."""
    parser = argparse.ArgumentParser(description="Meta-learning model recommender")
    parser.add_argument("--train", action="store_true", help="Train the meta-model on OpenML datasets.")
    parser.add_argument(
        "--openml-limit",
        type=int,
        default=DEFAULT_OPENML_SIZE,
        help="Number of OpenML datasets to process.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Requested number of worker processes for dataset processing.",
    )
    parser.add_argument("--predict-csv", type=str, default=None, help="Path to CSV file for prediction.")
    parser.add_argument("--target", type=str, default=None, help="Optional target column name for prediction.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accurate", "fast"],
        default="accurate",
        help="Use the accurate or cost-aware fast ranking objective.",
    )
    parser.add_argument(
        "--benchmark-automl",
        action="store_true",
        help="Optionally benchmark the recommendation against installed AutoML tools.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a tiny, verbose debugging pass: 3 datasets, 1 model, DEBUG logging, no dataset parallelism.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for both training and batch recommendation."""
    args = parse_args()
    setup_logging(debug=args.debug)

    try:
        if args.train:
            predictor, summary = run_training_pipeline(
                openml_limit=args.openml_limit,
                n_jobs=args.n_jobs,
                debug=args.debug,
            )
            if predictor is not None and not summary.empty:
                print("Training summary rows:", len(summary))
                print("Top-1 Accuracy:", round(float(predictor.metrics.get("top_1_accuracy", 0.0)), 4))
                print("Top-3 Accuracy:", round(float(predictor.metrics.get("top_3_accuracy", 0.0)), 4))
                print("NDCG@3:", round(float(predictor.metrics.get("ndcg_at_3", 0.0)), 4))
                print("MAP@3:", round(float(predictor.metrics.get("map_at_3", 0.0)), 4))

        if args.predict_csv:
            predictor = MetaModelPredictor.load()
            result = recommend_for_csv(
                args.predict_csv,
                predictor,
                target_column=args.target,
                mode=args.mode,
                benchmark_automl=args.benchmark_automl,
            )
            print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline CLI failure: %s", exc)
        raise


if __name__ == "__main__":
    if os.name == "nt":
        import multiprocessing as mp

        mp.freeze_support()
    main()
