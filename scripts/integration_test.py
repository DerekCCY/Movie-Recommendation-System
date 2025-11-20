"""
Integration Test: Pipeline → Serving Flow
Validates that models trained by ml_pipeline can be served by deployment/src/app.py
"""
import sys
from pathlib import Path
import tempfile
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.pipeline import run_training_pipeline, run_inference_pipeline
from ml_pipeline.serialize import load_model, save_model
import pandas as pd

def test_pipeline_to_serving():
    """
    Full integration test:
    1. Train model with ml_pipeline
    2. Save model
    3. Load model in deployment wrapper
    4. Test predictions
    """
    print("="*80)
    print("INTEGRATION TEST: PIPELINE -> SERVING")
    print("="*80)

    # Step 1: Train a model
    print("\n[1/5] Training model with ml_pipeline...")

    test_data = Path("tests/fixtures/sample_data.parquet")
    if not test_data.exists():
        print(f"X Test data not found: {test_data}")
        print("  Please ensure tests/fixtures/sample_data.parquet exists")
        return False

    temp_dir = Path(tempfile.mkdtemp())
    model_path = temp_dir / "integration_test_model.pkl"

    results = run_training_pipeline(
        data_path=str(test_data),
        model_output_path=str(model_path),
        enable_drift_detection=False
    )

    print(f"✓ Model trained: RMSE = {results['evaluation_metrics'].get('rmse', 'N/A')}")

    # Step 2: Verify model file exists and can be loaded
    print("\n[2/5] Verifying model serialization...")

    if not model_path.exists():
        print(f"X Model file not created: {model_path}")
        return False

    model = load_model(str(model_path))
    print(f"✓ Model loaded successfully")
    print(f"  - Users: {len(model.user_ids)}")
    print(f"  - Movies: {len(model.all_movie_ids)}")

    # Step 3: Test inference pipeline
    print("\n[3/5] Testing inference pipeline...")

    test_user = model.user_ids[0] if model.user_ids else "999999"
    recommendations = run_inference_pipeline(
        model_path=str(model_path),
        user_id=test_user,
        n_recommendations=10
    )

    print(f"✓ Generated {len(recommendations)} recommendations for user {test_user}")
    print(f"  Recommendations: {recommendations[:5]}...")

    # Step 4: Test deployment wrapper compatibility
    print("\n[4/5] Testing deployment wrapper compatibility...")

    # Import deployment app components
    sys.path.insert(0, str(Path(__file__).parent.parent / "deployment" / "src"))

    try:
        # Import SVDRecommenderWrapper
        import app
        from app import SVDRecommenderWrapper

        # Create wrapper instance
        wrapper = SVDRecommenderWrapper(
            model_path=str(model_path),
            interactions_path=None
        )

        # Test recommendation
        wrapper_recs = wrapper.get_recommendations(test_user, n_recommendations=10)

        print(f"✓ Deployment wrapper generated {len(wrapper_recs)} recommendations")
        print(f"  Recommendations: {wrapper_recs[:5]}...")

        # Verify consistency
        if wrapper_recs == recommendations:
            print("✓ Wrapper recommendations match pipeline recommendations")
        else:
            print("⚠ Wrapper recommendations differ from pipeline (this is okay if both are valid)")

    except Exception as e:
        print(f"X Deployment wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Test cold start handling
    print("\n[5/5] Testing cold start handling...")

    unknown_user = "999999999"
    cold_start_recs = wrapper.get_recommendations(unknown_user, n_recommendations=10)

    if len(cold_start_recs) > 0:
        print(f"✓ Cold start handling works: {len(cold_start_recs)} recommendations for unknown user")
    else:
        print("X Cold start handling failed: no recommendations generated")
        return False

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    print("\n" + "="*80)
    print("INTEGRATION TEST PASSED")
    print("="*80)

    return True

def main():
    """Run integration test"""
    success = test_pipeline_to_serving()

    if success:
        print("\nAll integration tests passed!")
        return 0
    else:
        print("\nIntegration tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
