import pandas as pd
import numpy as np
import pytest
from meta_recommender.predictor import MetaModelPredictor

def test_baseline_calibrated():
    X = pd.DataFrame({
        "A": np.random.randn(100),
        "B": np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, size=100))
    
    metrics, best_model, le = MetaModelPredictor._evaluate_baseline_models(X, y)
    
    assert "xgboost_classifier" in metrics["baselines"]
    assert "top_1_accuracy" in metrics["baselines"]["xgboost_classifier"]
