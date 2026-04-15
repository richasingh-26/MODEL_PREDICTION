"""Preprocessing and meta-feature extraction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import META_FEATURE_ORDER, Z_SCORE_THRESHOLD
from .logging_utils import log_exception

logger = logging.getLogger(__name__)


def detect_task_type(y: pd.Series) -> str:
    """Infer classification/regression from target properties."""
    y_non_null = y.dropna()
    if y_non_null.empty:
        return "classification"

    if pd.api.types.is_object_dtype(y_non_null) or pd.api.types.is_categorical_dtype(y_non_null):
        return "classification"

    unique_count = y_non_null.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y_non_null), 1)
    if unique_count <= 20 and unique_ratio < 0.2:
        return "classification"

    return "regression"


def clean_X(X: pd.DataFrame) -> pd.DataFrame:
    """Drop constant and fully missing columns; keep tabular DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    X = X.copy()
    full_missing_cols = [c for c in X.columns if X[c].isna().all()]
    if full_missing_cols:
        X = X.drop(columns=full_missing_cols)

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    if X.empty:
        raise ValueError("No usable columns after cleaning features.")

    return X


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a preprocessing transformer for mixed tabular data."""
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def _entropy_from_series(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True, dropna=True)
    if probs.empty:
        return 0.0
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _safe_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if not num_cols:
        return pd.DataFrame(index=X.index)
    num_df = X[num_cols].apply(pd.to_numeric, errors="coerce")
    return num_df.fillna(num_df.mean(numeric_only=True)).fillna(0.0)


def extract_meta_features(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute robust meta-features in a fixed order."""
    try:
        X = clean_X(X)
        n_samples, n_features = X.shape
        n_total_cells = n_samples * n_features
        missing_ratio = float(X.isna().sum().sum() / max(n_total_cells, 1))

        num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        num_df = _safe_numeric_frame(X)

        if not num_df.empty:
            variances = num_df.var(axis=0)
            variances = variances[variances > 0]
            mean_variance = float(variances.mean()) if not variances.empty else 0.0

            skewness = num_df.skew(axis=0).replace([np.inf, -np.inf], np.nan).dropna()
            mean_skewness = float(skewness.mean()) if not skewness.empty else 0.0

            kurtosis = num_df.kurt(axis=0).replace([np.inf, -np.inf], np.nan).dropna()
            mean_kurtosis = float(kurtosis.mean()) if not kurtosis.empty else 0.0

            entropies = [_entropy_from_series(num_df[col].round(4)) for col in num_df.columns]
            mean_entropy = float(np.mean(entropies)) if entropies else 0.0

            corr = num_df.corr().replace([np.inf, -np.inf], np.nan)
            if corr.shape[0] > 1:
                upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
                mean_abs_corr = float(upper_tri.abs().mean()) if not upper_tri.empty else 0.0
            else:
                mean_abs_corr = 0.0

            pca_var_1 = 0.0
            pca_var_2 = 0.0
            if min(num_df.shape) >= 2:
                try:
                    pca = PCA(n_components=2, random_state=42)
                    pca.fit(num_df)
                    ratios = pca.explained_variance_ratio_
                    pca_var_1 = float(ratios[0]) if len(ratios) > 0 else 0.0
                    pca_var_2 = float(ratios[1]) if len(ratios) > 1 else 0.0
                except Exception as exc:  # noqa: BLE001
                    log_exception(logger, "feature extraction", "pca", exc)

            means = num_df.mean(axis=0)
            stds = num_df.std(axis=0).replace(0, np.nan)
            z_scores = ((num_df - means) / stds).abs()
            outlier_percentage = float((z_scores > Z_SCORE_THRESHOLD).sum().sum() / max(n_total_cells, 1))
            
            feature_importance_var = 0.0
            if num_df.shape[1] > 1:
                try:
                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                    if detect_task_type(y) == "classification":
                        dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(num_df, y)
                    else:
                        dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(num_df, y)
                    feature_importance_var = float(np.var(dt.feature_importances_))
                except Exception as exc:  # noqa: BLE001
                    log_exception(logger, "feature extraction", "feature_importance_var", exc)

        else:
            mean_variance = 0.0
            mean_skewness = 0.0
            mean_kurtosis = 0.0
            mean_entropy = 0.0
            mean_abs_corr = 0.0
            pca_var_1 = 0.0
            pca_var_2 = 0.0
            outlier_percentage = 0.0
            feature_importance_var = 0.0

        class_imbalance_ratio = 1.0
        if detect_task_type(y) == "classification":
            counts = y.value_counts(dropna=True)
            if not counts.empty and counts.max() > 0:
                class_imbalance_ratio = float(counts.min() / counts.max())

        non_missing = X.notna().sum().sum()
        sparsity = 1.0 - float(non_missing / max(n_total_cells, 1))
        if num_cols:
            numeric_zero_cells = (X[num_cols].fillna(0) == 0).sum().sum()
            sparsity = max(sparsity, float(numeric_zero_cells / max(n_total_cells, 1)))

        features = {
            "n_samples": float(n_samples),
            "n_features": float(n_features),
            "missing_ratio": float(missing_ratio),
            "n_numeric": float(len(num_cols)),
            "n_categorical": float(len(cat_cols)),
            "mean_variance": float(mean_variance),
            "mean_skewness": float(mean_skewness),
            "mean_kurtosis": float(mean_kurtosis),
            "mean_entropy": float(mean_entropy),
            "mean_abs_correlation": float(mean_abs_corr),
            "pca_component_1_var": float(pca_var_1),
            "pca_component_2_var": float(pca_var_2),
            "feature_sparsity": float(sparsity),
            "outlier_percentage": float(outlier_percentage),
            "feature_importance_var": float(feature_importance_var),
            "class_imbalance_ratio": float(class_imbalance_ratio),
        }
        # Ensure numerical stability
        clean_features = {}
        for k in META_FEATURE_ORDER:
            val = float(features.get(k, 0.0))
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            clean_features[k] = val
        return clean_features
    except Exception as exc:  # noqa: BLE001
        log_exception(logger, "feature extraction", "input_dataset", exc)
        raise
