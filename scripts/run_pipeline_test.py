"""
Pipeline Test Script
Tests the full ml_pipeline end-to-end with real data
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.pipeline import run_training_pipeline
from ml_pipeline.config import INTERACTIONS_PATH, DEFAULT_MODEL_PATH
import json

def main():
    """
    Run the training pipeline with test configuration
    """
    print("="*80)
    print("PIPELINE TEST - FULL TRAINING RUN")
    print("="*80)

    # Use smaller test data if available
    test_data_path = Path("tests/fixtures/sample_data.parquet")
    if test_data_path.exists():
        data_path = str(test_data_path)
        print(f"Using test data: {data_path}")
    else:
        data_path = str(INTERACTIONS_PATH)
        print(f"Using production data: {data_path}")

    # Output to test directory
    model_output = Path("tests/output/test_model.pkl")
    model_output.parent.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    print("\nStarting pipeline execution...")
    results = run_training_pipeline(
        data_path=data_path,
        model_output_path=str(model_output),
        enable_drift_detection=True
    )

    # Print results
    print("\n" + "="*80)
    print("PIPELINE TEST RESULTS")
    print("="*80)
    print(f"Model path: {results['model_path']}")
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    print(f"Training time: {results['training_time_sec']:.2f} seconds")
    print(f"\nEvaluation Metrics:")
    for metric, value in results['evaluation_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    print(f"\nData Quality:")
    print(f"  Retention rate: {results['data_quality_report']['data_retention_rate']*100:.1f}%")

    if results['drift_report']:
        print(f"\nDrift Detection:")
        print(f"  Drift detected: {results['drift_report']['drift_detected']}")
        if results['drift_report']['drift_detected']:
            print(f"  Issues: {', '.join(results['drift_report']['drift_summary'])}")

    # Save results
    results_file = Path("tests/output/pipeline_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    print("="*80)
    print("PIPELINE TEST COMPLETE")
    print("="*80)

    return results

if __name__ == "__main__":
    main()
