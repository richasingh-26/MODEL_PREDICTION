import pandas as pd
import numpy as np
import pytest
from meta_recommender.features import detect_task_type, extract_meta_features, clean_X

def test_detect_task_type_classification():
    y = pd.Series(["A", "A", "B", "C"])
    assert detect_task_type(y) == "classification"

def test_detect_task_type_regression():
    y = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1, 20.1, 21.1])
    assert detect_task_type(y) == "regression"

def test_clean_X_removes_constant_and_missing():
    X = pd.DataFrame({
        "valid": [1, 2, 3],
        "constant": [5, 5, 5],
        "missing": [np.nan, np.nan, np.nan]
    })
    X_clean = clean_X(X)
    assert list(X_clean.columns) == ["valid"]

def test_extract_meta_features_numerical_stability():
    X = pd.DataFrame({
        "A": [1, 1, 1], # Constant
        "B": [1.0, np.inf, -np.inf], # Infs
        "C": [np.nan, np.nan, np.nan], # NaNs
        "D": [1, 2, 3] # Valid
    })
    y = pd.Series([0, 1, 0])
    
    # Need to handle ValueError correctly because clean_X might raise if all empty, 
    # but here D is valid so it shouldn't.
    try:
        features = extract_meta_features(X, y)
        assert features["n_samples"] == 3
        # Should not have NaNs
        for k, v in features.items():
            assert not np.isnan(v)
            assert not np.isinf(v)
    except ValueError as e:
        pytest.fail(f"Feature extraction failed: {e}")
