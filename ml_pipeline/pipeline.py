"""
ML Pipeline Orchestrator

Main entry point for running the end-to-end ML pipeline:
1. Data loading
2. Preprocessing with data quality checks
3. Drift detection
4. Feature engineering
5. Model training
6. Model evaluation
7. Model serialization
8. Baseline statistics computation

"""

import time
import logging
import json
import numpy as np
from pathlib import Path
import hashlib, subprocess
from typing import Dict, Optional, Any
from datetime import timezone

from . import config
from .data_io import load_interactions_from_parquet
from .data_drift import compute_baseline_statistics, detect_drift, to_python
from .feature_engineering import build_user_item_matrix, calculate_popular_movies, calculate_global_statistics
from .train import train_test_split_temporal, train_svd_model, train_hybrid_model
from .evaluate import generate_evaluation_report
from .serialize import save_model, get_model_size, load_model


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Temporary compatibility: use old data_io.py structure
def load_enriched_interactions(path):
    """Wrapper for backward compatibility"""
    return load_interactions_from_parquet(path)

def clean_interactions_for_silver(bronze_dir, output_path, config, report_quality=True, loaded_df=None):
    """Wrapper - use already-loaded data if provided"""
    if loaded_df is not None:
        # Use the data that was already loaded by the pipeline
        df = loaded_df
    elif Path(output_path).exists():
        # Fallback: load from output_path if it exists
        df = load_interactions_from_parquet(output_path)
    else:
        # Last resort: look for interactions.parquet in bronze_dir or parent
        possible_paths = [
            Path(bronze_dir) / "interactions.parquet",
            Path(bronze_dir).parent / "interactions.parquet",
            Path(bronze_dir).parent / "gold" / "interactions.parquet"
        ]
        df = None
        for p in possible_paths:
            if p.exists():
                df = load_interactions_from_parquet(str(p))
                break

        if df is None:
            raise FileNotFoundError(f"No interactions.parquet found near {bronze_dir}")

    quality_report = {
        'data_retention_rate': 1.0,
        'total_interactions': len(df),
        'valid_interactions': len(df)
    }
    return df, quality_report

# These two functions are for model versioning
def _git_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def _file_sha256(p: str | Path) -> str:
    """Return SHA256 hash of file, or 'unknown' if file doesn't exist."""
    p = Path(p)
    if not p.exists():
        return "sha256:unknown"  # For tests with fake paths

    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return "sha256:" + h.hexdigest()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_pipeline(data_path: Optional[str] = None,
                        model_output_path: Optional[str] = None,
                        model_type: str = 'hybrid',
                        svd_config: Optional[Dict] = None,
                        hybrid_config: Optional[Dict] = None,
                        eval_config: Optional[Dict] = None,
                        enable_drift_detection: bool = True,
                        baseline_stats_path: Optional[str] = None) -> Dict:
    """
    Run the complete training pipeline from data to trained model.

    Steps:
    1. Load interactions data
    2. Clean and validate data quality
    3. Detect drift (if baseline exists)
    4. Split into train/test
    5. Build feature matrix
    6. Train model (SVD or Hybrid based on model_type)
    7. Evaluate model
    8. Save model to disk
    9. Compute and save baseline statistics for future drift detection

    Args:
        data_path: Path to interactions parquet file (uses default if None)
        model_output_path: Path to save trained model (uses default if None)
        model_type: Model type to train ('hybrid' or 'svd'), default='hybrid'
        svd_config: SVD hyperparameters (uses default if None)
        hybrid_config: Hybrid model hyperparameters (uses default if None)
        eval_config: Evaluation config (uses default if None)
        enable_drift_detection: Whether to run drift detection (requires baseline)
        baseline_stats_path: Path to baseline stats JSON (auto-generated if None)

    Returns:
        Dict with pipeline results:
        - model_path: str
        - model_size_mb: float
        - training_time_sec: float
        - evaluation_metrics: Dict
        - data_quality_report: Dict
        - drift_report: Dict (if drift detection enabled)

    Example:
        >>> results = run_training_pipeline(
        ...     data_path="data/gold/interactions.parquet",
        ...     model_output_path="models/svd_model.pkl"
        ... )
        >>> print(results['evaluation_metrics']['rmse'])
        1.0468
        >>> print(results['data_quality_report']['data_retention_rate'])
        0.95
    """
    start_time = time.time()

    # Set defaults from config
    if data_path is None:
        data_path = config.INTERACTIONS_PATH
    if model_output_path is None:
        model_output_path = config.DEFAULT_MODEL_PATH
    if svd_config is None:
        svd_config = config.SVD_CONFIG
    if hybrid_config is None:
        hybrid_config = config.HYBRID_CONFIG
    if eval_config is None:
        eval_config = config.EVALUATION_CONFIG
    if baseline_stats_path is None:
    # Save to project root Data/ folder for monitoring service
        baseline_stats_path = Path(__file__).parent.parent / "Data" / "baseline_statistics.json"

    logger.info("=" * 60)
    logger.info("STARTING ML TRAINING PIPELINE WITH DATA QUALITY CHECKS")
    logger.info("=" * 60)

    # Step 1: Load interactions data
    logger.info(f"[1/9] Loading data from {data_path}...")
    interactions_df = load_enriched_interactions(str(data_path))
    logger.info(f"  Loaded {len(interactions_df)} interactions")
    
    # Keep track of the data version
    training_data_snapshots = [{
        "uri": str(data_path),
        "sha256": _file_sha256(data_path)
    }]

    # Step 2: Clean and validate data quality
    # Note: Bronze aggregation already done in automated_retraining.py if needed
    logger.info("[2/9] Cleaning and validating data quality...")
    cleaned_df, quality_report = clean_interactions_for_silver(
        bronze_dir=str(config.BRONZE_DIR),
        output_path=str(config.INTERACTIONS_PATH),
        config=config.PREPROCESSING_CONFIG,
        report_quality=True,
        loaded_df=interactions_df  # Use already-loaded data, skip bronze processing
    )
    logger.info(f"✅ Cleaned {len(cleaned_df)} interactions "
                f"({quality_report['data_retention_rate']*100:.1f}% retained)")

    # Step 3: Drift detection (if enabled and baseline exists)
    drift_report = None
    if enable_drift_detection and Path(baseline_stats_path).exists():
        logger.info(f"[3/9] Detecting data drift against baseline: {baseline_stats_path}...")
        
        # Load baseline statistics
        with open(baseline_stats_path, 'r') as f:
            baseline_stats = json.load(f)
        
        # Detect drift
        drift_report = detect_drift(baseline_stats, cleaned_df)
        
        if drift_report['drift_detected']:
            logger.warning("⚠️  DATA DRIFT DETECTED - Consider retraining!")
            logger.warning(f"  Issues: {', '.join(drift_report['drift_summary'])}")
        else:
            logger.info("✓ No significant drift detected")
    elif enable_drift_detection:
        logger.info(f"[3/9] Skipping drift detection - no baseline found at {baseline_stats_path}")
        logger.info("     Baseline will be created after training")
    else:
        logger.info("[3/9] Drift detection disabled")

    # Step 4: Split into train/test
    logger.info("[4/9] Splitting into train/test sets...")
    train_df, test_df = train_test_split_temporal(
        cleaned_df,
        test_size=config.TRAINING_CONFIG.get('test_size', 0.2),
        random_state=config.TRAINING_CONFIG.get('random_state', 42)
    )
    logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Keep track of train_df time window
    if 'ts' in train_df.columns:
        start_dt = train_df['ts'].min().to_pydatetime().astimezone(timezone.utc)
        end_dt   = train_df['ts'].max().to_pydatetime().astimezone(timezone.utc)
        time_window = {"start": start_dt.strftime("%Y-%m-%d"), "end":   end_dt.strftime("%Y-%m-%d")}
    else:
        # Fallback if no timestamp column
        time_window = {"start": "unknown", "end": "unknown"}

    # Step 5: Train model (SVD or Hybrid)
    if model_type.lower() == 'hybrid':
        logger.info("[5/9] Training Hybrid Recommendation model...")
        model = train_hybrid_model(
            train_df=train_df,
            interactions_df=interactions_df,  # Full data with metadata
            config=hybrid_config,
            log_to_wandb=False
        )
        hyperparams = hybrid_config #Keep track of model hyperparameters
    elif model_type.lower() == 'svd':
        logger.info("[5/9] Training SVD model...")
        model = train_svd_model(train_df, svd_config, log_to_wandb=False)
        hyperparams = svd_config
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'hybrid' or 'svd'")
    
    training_commit = _git_commit_short()
    logger.info(f"  {model_type.upper()} model training complete")
    logger.info(f" Training commit: {training_commit}")

    # Step 6: Evaluate model
    logger.info("[6/9] Evaluating model on test set...")
    evaluation_metrics = generate_evaluation_report(model, test_df, eval_config)

    # Step 7: Save model to disk
    logger.info(f"[7/9] Saving model to {model_output_path}...")
    save_model(model, str(model_output_path))
    model_size_mb = get_model_size(str(model_output_path))
    logger.info(f"  Model size: {model_size_mb:.2f} MB")

    # Step 8: Compute and save baseline statistics
    logger.info(f"[8/9] Computing baseline statistics...")
    baseline_stats = compute_baseline_statistics(train_df, str(baseline_stats_path))
    logger.info(f"  Baseline saved to: {baseline_stats_path}")

    # Step 9: Save data quality and drift reports
    logger.info("[9/9] Saving quality reports...")
    reports_dir = Path(model_output_path).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Save quality report
    quality_report_path = reports_dir / "data_quality_report.json"
    with open(quality_report_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"  Data quality report: {quality_report_path}")
    
    # Save drift report if available
    if drift_report:
        drift_report = to_python(drift_report) # Handling data type
        drift_report_path = reports_dir / "drift_report.json"
        with open(drift_report_path, 'w') as f:
            json.dump(drift_report, f, indent=2, cls=NumpyEncoder)
        logger.info(f"  Drift report: {drift_report_path}")

        monitoring_drift_path = Path("/group-project-f25-the-real-reel-deal/monitoring/drift_report.json")
        monitoring_drift_path.parent.mkdir(parents=True, exist_ok=True)
        with open(monitoring_drift_path, 'w') as f:
            json.dump(drift_report, f, indent=2, cls=NumpyEncoder)
        logger.info(f"  Drift report (monitoring): {monitoring_drift_path}")

    training_time_sec = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Total time: {training_time_sec:.2f} seconds")
    logger.info(f"Model saved to: {model_output_path}")
    logger.info(f"RMSE: {evaluation_metrics.get('rmse', 'N/A')}")
    logger.info(f"Data retention: {quality_report['data_retention_rate']*100:.1f}%")
    if drift_report and drift_report['drift_detected']:
        logger.warning(f"⚠️  Drift detected in: {', '.join(drift_report['drift_summary'])}")
    logger.info("=" * 60)

    return {
        'model_path': str(model_output_path),
        'model_size_mb': model_size_mb,
        'training_time_sec': training_time_sec,
        'evaluation_metrics': evaluation_metrics,
        'data_quality_report': quality_report,
        'drift_report': drift_report,
        'baseline_stats_path': str(baseline_stats_path),
        "time_window": time_window,
        'training_commit': training_commit,
        'training_data_snapshots': training_data_snapshots,
        'hyperparams': hyperparams
    }

def run_inference_pipeline(model_path: str, user_id: str,
                        n_recommendations: int = 20) -> list:
    """
    Run inference pipeline for a single user.

    Steps:
    1. Load trained model
    2. Generate recommendations

    Args:
        model_path: Path to saved model file
        user_id: User identifier
        n_recommendations: Number of recommendations to return

    Returns:
        List of recommended movie IDs

    Example:
        >>> recommendations = run_inference_pipeline(
        ...     model_path="models/svd_model.pkl",
        ...     user_id="12345"
        ... )
        >>> print(recommendations[:5])
        ['movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5']
    """
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)

    logger.info(f"Generating {n_recommendations} recommendations for user {user_id}...")
    recommendations = model.predict(user_id, n_recommendations=n_recommendations)

    logger.info(f"Generated {len(recommendations)} recommendations")
    return recommendations


if __name__ == "__main__":
    """
    Run pipeline from command line with optional model type selection.

    Usage:
        python -m ml_pipeline.pipeline                    # Train hybrid model (default)
        python -m ml_pipeline.pipeline --model-type=svd   # Train SVD model
        python -m ml_pipeline.pipeline --model-type=hybrid # Train hybrid model explicitly
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run ML training pipeline')
    parser.add_argument('--model-type', type=str, default='hybrid',
                       choices=['hybrid', 'svd'],
                       help='Model type to train (default: hybrid)')
    args = parser.parse_args()

    logger.info("Starting ML training pipeline...")
    logger.info(f"Model type: {args.model_type.upper()}")

    results = run_training_pipeline(model_type=args.model_type)

    logger.info("Pipeline completed successfully!")
    logger.info(f"Model type: {args.model_type.upper()}")
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"Model size: {results['model_size_mb']:.2f} MB")
    logger.info(f"Training time: {results['training_time_sec']:.2f} seconds")
    logger.info(f"RMSE: {results['evaluation_metrics']['rmse']:.4f}")