"""
Integration Test for ML Pipeline - End-to-End Testing
======================================================

Tests the ENTIRE pipeline (steps 1-9) on small data to catch API mismatches
BEFORE running 20-hour training jobs.

This test would have caught the get_model_info() bug in <1 second instead of 20+ hours.

Usage:
    pytest tests/test_pipeline_integration.py -v
    # Or run directly:
    python tests/test_pipeline_integration.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import pytest


def create_test_data(n_rows=1000, n_users=50, n_movies=100):
    """Create minimal test dataset with all required columns for hybrid model"""
    np.random.seed(42)

    users = [f"user_{i}" for i in range(n_users)]
    movies = [f"movie_{i}" for i in range(n_movies)]

    # Generate all required columns
    genres_list = [
        "['Action', 'Thriller']",
        "['Drama', 'Romance']",
        "['Comedy']",
        "['Sci-Fi', 'Adventure']",
        "['Horror']"
    ]

    data = {
        'user_id': np.random.choice(users, n_rows),
        'movie_id': np.random.choice(movies, n_rows),
        'rating': np.random.uniform(1, 5, n_rows),
        'watch_minutes': np.random.uniform(0, 180, n_rows),
        'runtime': np.random.uniform(80, 180, n_rows),
        'genres': np.random.choice(genres_list, n_rows),
        'vote_average': np.random.uniform(5, 9, n_rows),
        'popularity': np.random.uniform(1, 100, n_rows),
        'vote_count': np.random.randint(100, 10000, n_rows),
        'ts': pd.date_range('2024-01-01', periods=n_rows, freq='1min')
    }

    return pd.DataFrame(data)


def test_model_api_compatibility():
    """Test that both models have compatible APIs - catches method mismatches immediately"""
    from ml_pipeline.model import ImprovedSVDRecommendationModel, PersonalizedHybridRecommender

    print("\n" + "="*70)
    print("API COMPATIBILITY TEST")
    print("="*70)

    # Required methods for pipeline compatibility
    required_methods = [
        'fit',
        'predict',
        'predict_rating',
        'get_model_info'  # This catches the bug immediately!
    ]

    models = {
        'SVD': ImprovedSVDRecommendationModel,
        'Hybrid': PersonalizedHybridRecommender
    }

    for model_name, model_class in models.items():
        print(f"\nChecking {model_name} Model:")

        for method in required_methods:
            has_method = hasattr(model_class, method)
            status = "PASS" if has_method else "MISSING"
            print(f"  {status} {method}()")

            if not has_method:
                pytest.fail(f"{model_name} model missing required method: {method}()")

    print("\nPASS - All required methods present in both models")


def test_hybrid_pipeline_end_to_end():
    """Test ENTIRE hybrid pipeline (steps 1-9) on 1000 rows"""
    from ml_pipeline.pipeline import run_training_pipeline

    print("\n" + "="*70)
    print("HYBRID PIPELINE INTEGRATION TEST (1000 rows)")
    print("="*70)

    # Create test data
    print("\n[1/6] Creating test dataset...")
    test_df = create_test_data(n_rows=1000)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "test_interactions.parquet"
        test_df.to_parquet(data_path)

        model_path = tmpdir / "hybrid_model.pkl"

        # Run ENTIRE pipeline (all 9 steps)
        print("\n[2/6] Running hybrid training pipeline (steps 1-9)...")
        results = run_training_pipeline(
            data_path=str(data_path),
            model_output_path=str(model_path),
            model_type='hybrid',
            hybrid_config={'n_factors': 10, 'n_epochs': 5},  # Small for speed
            eval_config={'k_values': [5, 10], 'n_inference_samples': 10},
            enable_drift_detection=False
        )

        # Verify ALL pipeline outputs
        print("\n[3/6] Verifying pipeline outputs...")
        assert 'model_path' in results, "Missing model_path in results"
        assert 'model_size_mb' in results, "Missing model_size_mb in results"
        assert 'training_time_sec' in results, "Missing training_time_sec in results"
        assert 'evaluation_metrics' in results, "Missing evaluation_metrics in results"

        # Verify evaluation metrics structure
        print("\n[4/6] Verifying evaluation metrics...")
        metrics = results['evaluation_metrics']
        assert 'rmse' in metrics, "Missing RMSE"
        assert 'precision@k' in metrics, "Missing precision@k"
        assert 'recall@k' in metrics, "Missing recall@k"
        assert 'inference_time' in metrics, "Missing inference_time"
        assert 'model_info' in metrics, "Missing model_info (THIS WOULD CATCH THE BUG!)"

        # Verify model_info structure (critical!)
        print("\n[5/6] Verifying model_info structure...")
        model_info = metrics['model_info']
        assert 'algorithm' in model_info, "Missing 'algorithm' in model_info"
        assert 'total_users' in model_info, "Missing 'total_users' in model_info"
        assert 'total_movies' in model_info, "Missing 'total_movies' in model_info"

        # Test inference
        print("\n[6/6] Testing inference...")
        from ml_pipeline.pipeline import run_inference_pipeline
        recs = run_inference_pipeline(
            model_path=str(model_path),
            user_id="user_0",
            n_recommendations=10
        )
        assert len(recs) > 0, "No recommendations returned"

        print("\n" + "="*70)
        print("PASS - HYBRID PIPELINE TEST PASSED")
        print("="*70)
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - Model size: {results['model_size_mb']:.2f} MB")
        print(f"  - Training time: {results['training_time_sec']:.2f}s")
        print(f"  - Recommendations: {len(recs)}")
        print(f"  - Algorithm: {model_info['algorithm']}")


def test_svd_pipeline_end_to_end():
    """Test ENTIRE SVD pipeline (steps 1-9) on 1000 rows"""
    from ml_pipeline.pipeline import run_training_pipeline

    print("\n" + "="*70)
    print("SVD PIPELINE INTEGRATION TEST (1000 rows)")
    print("="*70)

    # Create test data
    print("\n[1/6] Creating test dataset...")
    test_df = create_test_data(n_rows=1000)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "test_interactions.parquet"
        test_df.to_parquet(data_path)

        model_path = tmpdir / "svd_model.pkl"

        # Run ENTIRE pipeline (all 9 steps)
        print("\n[2/6] Running SVD training pipeline (steps 1-9)...")
        results = run_training_pipeline(
            data_path=str(data_path),
            model_output_path=str(model_path),
            model_type='svd',
            svd_config={'n_factors': 10},  # Small for speed
            eval_config={'k_values': [5, 10], 'n_inference_samples': 10},
            enable_drift_detection=False
        )

        # Verify ALL pipeline outputs
        print("\n[3/6] Verifying pipeline outputs...")
        assert 'model_path' in results
        assert 'model_size_mb' in results
        assert 'evaluation_metrics' in results

        # Verify evaluation metrics
        print("\n[4/6] Verifying evaluation metrics...")
        metrics = results['evaluation_metrics']
        assert 'rmse' in metrics
        assert 'model_info' in metrics

        # Verify model_info
        print("\n[5/6] Verifying model_info structure...")
        model_info = metrics['model_info']
        assert 'algorithm' in model_info
        assert 'total_users' in model_info

        # Test inference
        print("\n[6/6] Testing inference...")
        from ml_pipeline.pipeline import run_inference_pipeline
        recs = run_inference_pipeline(
            model_path=str(model_path),
            user_id="user_0",
            n_recommendations=10
        )
        assert len(recs) > 0

        print("\n" + "="*70)
        print("PASS - SVD PIPELINE TEST PASSED")
        print("="*70)
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - Algorithm: {model_info['algorithm']}")


if __name__ == "__main__":
    print("="*70)
    print("ML PIPELINE INTEGRATION TESTS")
    print("Tests entire pipeline on 1000 rows to catch API bugs in <60s")
    print("="*70)

    try:
        # Test 1: API compatibility (instant)
        test_model_api_compatibility()

        # Test 2: Hybrid pipeline end-to-end (~30s)
        test_hybrid_pipeline_end_to_end()

        # Test 3: SVD pipeline end-to-end (~20s)
        test_svd_pipeline_end_to_end()

        print("\n" + "="*70)
        print("PASS - ALL INTEGRATION TESTS PASSED")
        print("="*70)
        print("Safe to run full pipeline on production data!")
        sys.exit(0)

    except Exception as e:
        print("\n" + "="*70)
        print("✗✗✗ INTEGRATION TEST FAILED ✗✗✗")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nWARNING: DO NOT run on production data until this passes!")
        sys.exit(1)
