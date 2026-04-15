"""Meta-model training, ranking, comparison, and explanation utilities."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ndcg_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRanker

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False

from .config import (
    DEFAULT_TOP_K,
    INFERENCE_WEIGHT,
    META_EVALUATION_PATH,
    META_FEATURE_ORDER,
    MODEL_PATH,
    RANK_RELEVANCE_SCALE,
    RANDOM_STATE,
    SCALER_PATH,
    TRAINING_REPORT_PATH,
)

logger = logging.getLogger(__name__)

ModeName = Literal["accurate", "fast"]


def _safe_literal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if total <= 0:
        return np.repeat(1.0 / max(len(values), 1), len(values))
    return exp_values / total


def _top_k_accuracy_from_proba(probabilities: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top_idx = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
    hits = [truth in preds for truth, preds in zip(y_true, top_idx, strict=False)]
    return float(np.mean(hits)) if hits else 0.0


def _average_precision_at_k(relevant: set[str], predicted: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for rank, model_name in enumerate(predicted[:k], start=1):
        if model_name in relevant:
            hits += 1
            score += hits / rank
    return float(score / min(len(relevant), k))


@dataclass
class RankingDataset:
    """Training matrix for learning-to-rank."""

    X: pd.DataFrame
    y: pd.Series
    groups: list[int]
    query_ids: list[int]
    candidate_models: list[str]


@dataclass
class MetaModelPredictor:
    """Ranking-first meta-learning predictor with baseline comparisons."""

    ranking_models: dict[str, XGBRanker]
    baseline_model: Any
    scaler: StandardScaler
    feature_order: list[str]
    model_encoder: dict[str, int]
    fill_values: dict[str, float]
    model_profiles: dict[str, dict[str, float]]
    label_encoder: LabelEncoder
    metrics: dict[str, Any]

    @staticmethod
    def _prepare_classifier_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(
            [{feature: float(record.get(f"meta_{feature}", 0.0)) for feature in META_FEATURE_ORDER} for _, record in df.iterrows()]
        )
        return frame[META_FEATURE_ORDER].fillna(0.0)

    @staticmethod
    def _build_model_profiles(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        buckets: dict[str, dict[str, list[float]]] = {}
        for _, record in df.iterrows():
            scores = _safe_literal(record.get("model_scores", {})) or {}
            timings = _safe_literal(record.get("model_timings", {})) or {}
            if not isinstance(scores, dict):
                continue
            for model_name, score in scores.items():
                bucket = buckets.setdefault(
                    str(model_name),
                    {"score": [], "training_time": [], "inference_time": []},
                )
                if pd.notna(score):
                    bucket["score"].append(float(score))
                timing = timings.get(model_name, {}) if isinstance(timings, dict) else {}
                if isinstance(timing, dict):
                    bucket["training_time"].append(float(timing.get("training_time", 0.0) or 0.0))
                    bucket["inference_time"].append(float(timing.get("inference_time", 0.0) or 0.0))

        profiles: dict[str, dict[str, float]] = {}
        for model_name, values in buckets.items():
            profiles[model_name] = {
                "mean_score": float(np.mean(values["score"])) if values["score"] else 0.0,
                "mean_training_time": float(np.mean(values["training_time"])) if values["training_time"] else 0.0,
                "mean_inference_time": float(np.mean(values["inference_time"])) if values["inference_time"] else 0.0,
            }
        return profiles

    @classmethod
    def _extract_ranking_dataset(
        cls,
        df: pd.DataFrame,
        mode: ModeName,
        model_encoder: dict[str, int],
        model_profiles: dict[str, dict[str, float]],
    ) -> RankingDataset:
        rows: list[dict[str, float]] = []
        labels: list[int] = []
        groups: list[int] = []
        query_ids: list[int] = []
        candidate_models: list[str] = []

        for query_id, (_, record) in enumerate(df.iterrows()):
            scores = _safe_literal(record.get("model_scores", {})) or {}
            timings = _safe_literal(record.get("model_timings", {})) or {}
            if not isinstance(scores, dict):
                continue

            group_count = 0
            for model_name, raw_score in scores.items():
                if pd.isna(raw_score):
                    continue
                score = float(raw_score)
                timing = timings.get(model_name, {}) if isinstance(timings, dict) else {}
                train_time = float(timing.get("training_time", 0.0) or 0.0) if isinstance(timing, dict) else 0.0
                inference_time = (
                    float(timing.get("inference_time", 0.0) or 0.0) if isinstance(timing, dict) else 0.0
                )

                relevance = score
                if mode == "fast":
                    penalty = np.log1p(max(train_time, 0.0) + INFERENCE_WEIGHT * max(inference_time, 0.0))
                    relevance = score - 0.05 * penalty

                row = {feature: float(record.get(f"meta_{feature}", 0.0)) for feature in META_FEATURE_ORDER}
                profile = model_profiles.get(model_name, {})
                row["model_id"] = float(model_encoder[model_name])
                row["model_mean_score"] = float(profile.get("mean_score", 0.0))
                row["model_mean_training_time"] = float(profile.get("mean_training_time", 0.0))
                row["model_mean_inference_time"] = float(profile.get("mean_inference_time", 0.0))
                rows.append(row)
                labels.append(max(0, int(round(relevance * RANK_RELEVANCE_SCALE))))
                candidate_models.append(str(model_name))
                query_ids.append(query_id)
                group_count += 1

            if group_count:
                groups.append(group_count)

        feature_order = META_FEATURE_ORDER + [
            "model_id",
            "model_mean_score",
            "model_mean_training_time",
            "model_mean_inference_time",
        ]
        frame = pd.DataFrame(rows)
        for column in feature_order:
            if column not in frame.columns:
                frame[column] = 0.0
        return RankingDataset(
            X=frame[feature_order],
            y=pd.Series(labels, dtype=int),
            groups=groups,
            query_ids=query_ids,
            candidate_models=candidate_models,
        )

    @staticmethod
    def _fit_ranker(X: pd.DataFrame, y: pd.Series, groups: list[int]) -> XGBRanker:
        model = XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.85,
            n_estimators=220,
            random_state=RANDOM_STATE,
        )
        model.fit(X, y, group=groups, verbose=False)
        return model

    @staticmethod
    def _make_baseline_candidates(min_class_count: int = 5) -> dict[str, Any]:
        stacking_cv = max(2, min(3, min_class_count))
        xgb = XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
        stacking = StackingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=250,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=160,
                        max_depth=4,
                        learning_rate=0.06,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        random_state=RANDOM_STATE,
                        eval_metric="mlogloss",
                    ),
                ),
            ],
            final_estimator=LogisticRegression(max_iter=1200),
            stack_method="predict_proba",
            passthrough=False,
            cv=stacking_cv,
        )
        return {
            "xgboost_classifier": CalibratedClassifierCV(estimator=xgb, method="sigmoid", cv=max(2, min(5, min_class_count))),
            "stacking_ensemble": CalibratedClassifierCV(estimator=stacking, method="sigmoid", cv=max(2, min(5, min_class_count))),
        }

    @classmethod
    def _evaluate_baseline_models(cls, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, Any], Any, LabelEncoder]:
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)
        if y_encoded.nunique() < 2:
            model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
            model.fit(X, y_encoded)
            return (
                {
                    "selected_baseline": "random_fallback",
                    "baselines": {
                        "random_fallback": {
                            "top_1_accuracy": 1.0,
                            "top_3_accuracy": 1.0,
                        }
                    },
                },
                model,
                label_encoder,
            )

        min_class_count = int(y_encoded.value_counts().min())
        candidate_models = cls._make_baseline_candidates(min_class_count=min_class_count)
        n_splits = min(5, min_class_count, len(X))
        if n_splits < 2:
            best_name = "xgboost_classifier"
            best_model = clone(candidate_models[best_name])
            best_model.fit(X, y_encoded)
            return (
                {
                    "selected_baseline": best_name,
                    "baselines": {
                        best_name: {
                            "top_1_accuracy": float(accuracy_score(y_encoded, best_model.predict(X))),
                            "top_3_accuracy": _top_k_accuracy_from_proba(best_model.predict_proba(X), y_encoded.to_numpy(), DEFAULT_TOP_K),
                        }
                    },
                },
                best_model,
                label_encoder,
            )
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        baseline_results: dict[str, Any] = {}
        best_name = ""
        best_score = -np.inf

        for name, estimator in candidate_models.items():
            top1_scores: list[float] = []
            top3_scores: list[float] = []
            for train_idx, test_idx in splitter.split(X, y_encoded):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y_encoded.iloc[train_idx]
                y_test = y_encoded.iloc[test_idx]
                fitted = clone(estimator)
                fitted.fit(X_train, y_train)
                probabilities = fitted.predict_proba(X_test)
                predictions = np.argmax(probabilities, axis=1)
                top1_scores.append(float(accuracy_score(y_test, predictions)))
                top3_scores.append(_top_k_accuracy_from_proba(probabilities, y_test.to_numpy(), DEFAULT_TOP_K))

            baseline_results[name] = {
                "top_1_accuracy": float(np.mean(top1_scores)),
                "top_3_accuracy": float(np.mean(top3_scores)),
            }
            if baseline_results[name]["top_3_accuracy"] > best_score:
                best_name = name
                best_score = baseline_results[name]["top_3_accuracy"]

        best_model = clone(candidate_models[best_name])
        best_model.fit(X, y_encoded)
        return (
            {
                "selected_baseline": best_name,
                "baselines": baseline_results,
            },
            best_model,
            label_encoder,
        )

    @staticmethod
    def _iter_group_slices(groups: list[int]) -> list[tuple[int, int]]:
        slices: list[tuple[int, int]] = []
        start = 0
        for size in groups:
            end = start + size
            slices.append((start, end))
            start = end
        return slices

    @classmethod
    def _evaluate_ranker(
        cls,
        ranking_dataset: RankingDataset,
        scaled_frame: pd.DataFrame,
        query_names: list[str],
    ) -> dict[str, float]:
        unique_queries = np.array(sorted(set(ranking_dataset.query_ids)))
        if len(unique_queries) < 2:
            return {
                "top_1_accuracy": 1.0,
                "top_3_accuracy": 1.0,
                "ndcg_at_3": 1.0,
                "map_at_3": 1.0,
            }

        n_splits = min(5, len(unique_queries))
        splitter = GroupKFold(n_splits=n_splits)
        groups_array = np.array(ranking_dataset.query_ids)
        top1_scores: list[float] = []
        top3_scores: list[float] = []
        ndcg_scores: list[float] = []
        map_scores: list[float] = []

        for train_idx, test_idx in splitter.split(scaled_frame, ranking_dataset.y, groups=groups_array):
            train_X = scaled_frame.iloc[train_idx]
            train_y = ranking_dataset.y.iloc[train_idx]
            test_X = scaled_frame.iloc[test_idx]

            train_group_sizes = (
                pd.Series(groups_array[train_idx]).value_counts().sort_index().astype(int).tolist()
            )
            test_group_sizes = pd.Series(groups_array[test_idx]).value_counts().sort_index().astype(int).tolist()
            model = cls._fit_ranker(train_X, train_y, train_group_sizes)
            predictions = model.predict(test_X)

            start = 0
            for group_size in test_group_sizes:
                end = start + group_size
                group_pred = predictions[start:end]
                group_true = ranking_dataset.y.iloc[test_idx].iloc[start:end].to_numpy()
                group_models = [query_names[idx] for idx in np.array(test_idx)[start:end]]

                pred_order = np.argsort(group_pred)[::-1]
                true_order = np.argsort(group_true)[::-1]
                predicted_models = [group_models[idx] for idx in pred_order[:DEFAULT_TOP_K]]
                best_true_model = group_models[true_order[0]]
                relevant_models = {group_models[idx] for idx in true_order[: min(DEFAULT_TOP_K, len(true_order))]}

                top1_scores.append(float(predicted_models[0] == best_true_model))
                top3_scores.append(float(best_true_model in predicted_models))
                if len(group_true) < 2:
                    ndcg_scores.append(1.0)
                else:
                    ndcg_scores.append(float(ndcg_score([group_true], [group_pred], k=DEFAULT_TOP_K)))
                map_scores.append(_average_precision_at_k(relevant_models, predicted_models, DEFAULT_TOP_K))
                start = end

        return {
            "top_1_accuracy": float(np.mean(top1_scores)) if top1_scores else 0.0,
            "top_3_accuracy": float(np.mean(top3_scores)) if top3_scores else 0.0,
            "ndcg_at_3": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "map_at_3": float(np.mean(map_scores)) if map_scores else 0.0,
        }

    @classmethod
    def train(cls, df: pd.DataFrame) -> "MetaModelPredictor":
        """Train strong classifier baselines and ranking models."""
        df = df.copy()
        model_profiles = cls._build_model_profiles(df)
        if not model_profiles:
            raise ValueError(
                "The meta-dataset is missing ranking metadata. Rebuild it with run_training_pipeline() so "
                "'model_scores' and 'model_timings' are included."
            )
        model_encoder = {model_name: idx for idx, model_name in enumerate(sorted(model_profiles))}
        classifier_frame = cls._prepare_classifier_frame(df)
        baseline_metrics, baseline_model, label_encoder = cls._evaluate_baseline_models(classifier_frame, df["best_model"])

        accurate_ranking = cls._extract_ranking_dataset(df, "accurate", model_encoder, model_profiles)
        fast_ranking = cls._extract_ranking_dataset(df, "fast", model_encoder, model_profiles)

        fill_values = accurate_ranking.X.median(numeric_only=True).fillna(0.0).to_dict()
        scaler = StandardScaler()
        scale_columns = [column for column in accurate_ranking.X.columns if column != "model_id"]

        accurate_frame = accurate_ranking.X.fillna(fill_values).fillna(0.0).copy()
        accurate_frame[scale_columns] = scaler.fit_transform(accurate_frame[scale_columns])
        fast_frame = fast_ranking.X.fillna(fill_values).fillna(0.0).copy()
        fast_frame[scale_columns] = scaler.transform(fast_frame[scale_columns])

        accurate_metrics = cls._evaluate_ranker(accurate_ranking, accurate_frame, accurate_ranking.candidate_models)
        fast_metrics = cls._evaluate_ranker(fast_ranking, fast_frame, fast_ranking.candidate_models)

        accurate_model = cls._fit_ranker(accurate_frame, accurate_ranking.y, accurate_ranking.groups)
        fast_model = cls._fit_ranker(fast_frame, fast_ranking.y, fast_ranking.groups)

        metrics = {
            "mode_metrics": {
                "accurate": accurate_metrics,
                "fast": fast_metrics,
            },
            "baseline_comparison": baseline_metrics,
            "selected_meta_model": "xgboost_lambdamart_ranker",
            "top_1_accuracy": accurate_metrics["top_1_accuracy"],
            "accuracy": accurate_metrics["top_1_accuracy"],
            "top_3_accuracy": accurate_metrics["top_3_accuracy"],
            "ndcg_at_3": accurate_metrics["ndcg_at_3"],
            "map_at_3": accurate_metrics["map_at_3"],
        }

        TRAINING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRAINING_REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        META_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_EVALUATION_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return cls(
            ranking_models={"accurate": accurate_model, "fast": fast_model},
            baseline_model=baseline_model,
            scaler=scaler,
            feature_order=accurate_ranking.X.columns.tolist(),
            model_encoder=model_encoder,
            fill_values={key: float(value) for key, value in fill_values.items()},
            model_profiles=model_profiles,
            label_encoder=label_encoder,
            metrics=metrics,
        )

    def save(self, path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> None:
        """Persist model artifacts to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ranking_models": self.ranking_models,
            "baseline_model": self.baseline_model,
            "feature_order": self.feature_order,
            "model_encoder": self.model_encoder,
            "fill_values": self.fill_values,
            "model_profiles": self.model_profiles,
            "label_encoder": self.label_encoder,
            "metrics": self.metrics,
        }
        joblib.dump(payload, path)
        joblib.dump(self.scaler, scaler_path)
        logger.info("Saved meta-model artifacts to %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> "MetaModelPredictor":
        """Load persisted model artifacts."""
        payload = joblib.load(path)
        scaler = joblib.load(scaler_path)
        return cls(
            ranking_models=payload["ranking_models"],
            baseline_model=payload["baseline_model"],
            scaler=scaler,
            feature_order=payload["feature_order"],
            model_encoder=payload["model_encoder"],
            fill_values=payload.get("fill_values", {}),
            model_profiles=payload.get("model_profiles", {}),
            label_encoder=payload["label_encoder"],
            metrics=payload.get("metrics", {}),
        )

    def _prepare_query_block(self, meta_features: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
        rows: list[dict[str, float]] = []
        candidate_models = sorted(self.model_encoder, key=self.model_encoder.get)
        for model_name in candidate_models:
            row = {feature: float(meta_features.get(feature, 0.0)) for feature in META_FEATURE_ORDER}
            profile = self.model_profiles.get(model_name, {})
            row["model_id"] = float(self.model_encoder[model_name])
            row["model_mean_score"] = float(profile.get("mean_score", 0.0))
            row["model_mean_training_time"] = float(profile.get("mean_training_time", 0.0))
            row["model_mean_inference_time"] = float(profile.get("mean_inference_time", 0.0))
            rows.append(row)

        frame = pd.DataFrame(rows)
        for column in self.feature_order:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[self.feature_order].fillna(self.fill_values).fillna(0.0)
        scale_columns = [column for column in self.feature_order if column != "model_id"]
        frame[scale_columns] = self.scaler.transform(frame[scale_columns])
        return frame, candidate_models

    def predict_top_k_models(self, meta_features: dict[str, float], k: int = 3, mode: ModeName = "accurate") -> list[tuple[str, float]]:
        """Predict ranked recommendations for a dataset."""
        frame, candidate_models = self._prepare_query_block(meta_features)
        scores = self.ranking_models[mode].predict(frame)
        probabilities = _softmax(scores)
        sorted_idx = np.argsort(scores)[::-1][:k]
        return [(candidate_models[idx], float(probabilities[idx])) for idx in sorted_idx]

    def predict_best_model(self, meta_features: dict[str, float], mode: ModeName = "accurate") -> str:
        """Predict the single best model label."""
        return self.predict_top_k_models(meta_features, k=1, mode=mode)[0][0]

    def compare_modes(self, meta_features: dict[str, float], k: int = 3) -> dict[str, list[dict[str, float | str]]]:
        """Return accurate and fast recommendations side by side."""
        return {
            mode: [
                {"model": model_name, "probability": probability}
                for model_name, probability in self.predict_top_k_models(meta_features, k=k, mode=mode)
            ]
            for mode in ("accurate", "fast")
        }

    def _fallback_feature_importance(self) -> list[dict[str, float | str]]:
        model = self.ranking_models["accurate"]
        importances = getattr(model, "feature_importances_", np.zeros(len(self.feature_order)))
        ranked = sorted(
            zip(self.feature_order, importances, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [{"feature": str(name), "importance": float(score)} for name, score in ranked[:5]]

    def get_explanation(self, meta_features: dict[str, float], mode: ModeName = "accurate") -> dict[str, Any]:
        """Explain the top recommendation using SHAP when available."""
        frame, candidate_models = self._prepare_query_block(meta_features)
        scores = self.ranking_models[mode].predict(frame)
        best_idx = int(np.argmax(scores))
        best_model_name = candidate_models[best_idx]

        if HAS_SHAP:
            try:
                explainer = shap.TreeExplainer(self.ranking_models[mode])
                shap_values = explainer.shap_values(frame)
                row_values = shap_values[best_idx]
                top_pairs = sorted(
                    zip(self.feature_order, row_values, strict=False),
                    key=lambda item: abs(float(item[1])),
                    reverse=True,
                )[:5]
                return {
                    "model": best_model_name,
                    "top_features": [
                        {"feature": str(name), "contribution": float(value)} for name, value in top_pairs
                    ],
                    "global_feature_importance": self._fallback_feature_importance(),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("SHAP explanation failed: %s", exc)

        return {
            "model": best_model_name,
            "top_features": self._fallback_feature_importance(),
            "global_feature_importance": self._fallback_feature_importance(),
        }
