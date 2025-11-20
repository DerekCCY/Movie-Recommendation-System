"""
Automated Retraining Script for M3
Suitable for cron jobs or scheduled tasks
"""
import sys
from pathlib import Path
from datetime import datetime
import json
import logging
import psutil
import subprocess
import time
import requests

# Add parent to path FIRST before importing provenance
sys.path.insert(0, str(Path(__file__).parent.parent))

from provenance.versioning import *

from ml_pipeline.pipeline import run_training_pipeline
from ml_pipeline.config import INTERACTIONS_PATH, DEFAULT_MODEL_PATH
from ml_pipeline.serialize import load_model

# Setup logging
log_dir = Path(__file__).parent.parent / "logs" / "retraining"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage):
    """
    Log current memory usage

    Args:
        stage: Description of current stage (e.g., "Before training")

    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[Memory] {stage}: {memory_mb:.2f} MB")
    return memory_mb

def should_retrain(drift_report):
    """
    Decide whether to retrain based on drift detection

    Args:
        drift_report: Drift detection results from previous run

    Returns:
        bool: True if retraining is recommended
    """
    if drift_report is None:
        logger.info("No previous drift report - proceeding with training")
        return True

    if drift_report.get('drift_detected', False):
        logger.warning("Data drift detected - retraining recommended")
        return True

    logger.info("No significant drift detected - retraining may not be necessary")
    # Still allow retraining (could add schedule logic here)
    return True

def backup_current_model(model_path):
    """
    Create backup of current model before retraining

    Args:
        model_path: Path to current model
    """
    if not Path(model_path).exists():
        logger.warning(f"No existing model found at {model_path}")
        return None

    backup_dir = Path(model_path).parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f"model_backup_{timestamp}.pkl"

    import shutil
    shutil.copy(model_path, backup_path)
    logger.info(f"Backed up current model to {backup_path}")

    return backup_path

def verify_model_integrity(model_path):
    """
    Verify that the trained model can be loaded and has required attributes

    Args:
        model_path: Path to model file

    Returns:
        bool: True if model is valid
    """
    try:
        model = load_model(str(model_path))

        # Check required attributes
        assert hasattr(model, 'user_ids'), "Model missing user_ids"
        assert hasattr(model, 'all_movie_ids'), "Model missing all_movie_ids"
        assert hasattr(model, 'predict'), "Model missing predict method"
        assert len(model.user_ids) > 0, "Model has no users"
        assert len(model.all_movie_ids) > 0, "Model has no movies"

        logger.info(f"Model integrity check passed: {len(model.user_ids)} users, {len(model.all_movie_ids)} movies")
        return True

    except Exception as e:
        logger.error(f"Model integrity check failed: {e}")
        return False

# restart_flask_service() removed - Docker handles deployment now

def main():
    """
    Main retraining workflow
    """
    logger.info("="*80)
    logger.info("AUTOMATED RETRAINING STARTED")
    logger.info("="*80)

    # Generate versioned model directory
    training_timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    versioned_model_dir = Path(__file__).parent.parent / "models" / f"recsys-{training_timestamp}"
    versioned_model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Versioned model directory: {versioned_model_dir}")

    # Configuration
    data_path = str(INTERACTIONS_PATH)
    versioned_model_path = versioned_model_dir / "improved_hybrid_model.pkl"
    model_output_path = str(versioned_model_path)  # Train to versioned path
    default_model_path = Path(__file__).parent.parent / "models" / "improved_hybrid_model.pkl"
    baseline_stats_path = Path(data_path).parent / "baseline_statistics.json"

    logger.info(f"Data path: {data_path}")
    logger.info(f"Versioned model: {model_output_path}")
    logger.info(f"Default model: {default_model_path}")
    logger.info(f"Baseline stats: {baseline_stats_path}")

    # Aggregate bronze data to silver before training (if available)
    logger.info("="*80)
    logger.info("CHECKING FOR NEW BRONZE DATA")
    logger.info("="*80)
    try:
        bronze_dir = Path(data_path).parent.parent / "bronze"
        watch_events_dir = bronze_dir / "watch_events"

        # Check if bronze layer has the proper structure AND data
        if watch_events_dir.exists() and list(watch_events_dir.rglob("*.parquet")):
            logger.info(f"Found bronze data - aggregating to silver")
            from ml_pipeline.data_io_refactored.silver_cleaning import clean_interactions_for_silver

            clean_interactions_for_silver(
                bronze_dir=str(bronze_dir),
                output_path=data_path,
                report_quality=True
            )
            logger.info("✓ Silver layer updated with fresh bronze data")
        else:
            logger.info("No bronze data found - using existing silver data")
            logger.info(f"  (Checked: {watch_events_dir})")

    except Exception as e:
        logger.warning(f"Bronze aggregation failed: {e}")
        logger.warning("Continuing with existing silver data...")

    # Check if data exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("RETRAINING FAILED")
        return 1

    # Backup current default model (the one Flask is using)
    backup_path = backup_current_model(str(default_model_path))

    try:
        # Log memory before training
        log_memory_usage("Before training")

        # Run training pipeline
        logger.info("Starting training pipeline...")
        results = run_training_pipeline(
            data_path=data_path,
            model_output_path=model_output_path,
            enable_drift_detection=True,
            baseline_stats_path=str(baseline_stats_path)
        )

        # Log memory after training
        peak_memory = log_memory_usage("After training")
        results['peak_memory_mb'] = peak_memory

        # Log results
        logger.info("="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Model size: {results['model_size_mb']:.2f} MB")
        logger.info(f"Training time: {results['training_time_sec']:.2f} seconds")
        logger.info(f"Peak memory: {peak_memory:.2f} MB")
        logger.info(f"RMSE: {results['evaluation_metrics'].get('rmse', 'N/A')}")
        logger.info(f"Data retention: {results['data_quality_report']['data_retention_rate']*100:.1f}%")

        # Check drift
        if results['drift_report'] and results['drift_report']['drift_detected']:
            logger.warning(f"Drift detected: {', '.join(results['drift_report']['drift_summary'])}")

        # Verify model integrity
        if verify_model_integrity(model_output_path):
            logger.info("Model integrity verified")
        else:
            logger.error("Model integrity check failed - restoring backup")
            if backup_path and Path(backup_path).exists():
                import shutil
                shutil.copy(backup_path, model_output_path)
                logger.info(f"Restored model from backup: {backup_path}")
            return 1
        
        new_version = save_versioned_artifacts(results, model_output_path)
        set_candidate(new_version)
        logger.info(f"[Provenance] Candidate version -> {new_version} (registry.json updated)")

        # Copy versioned model to default path for Flask
        logger.info("="*80)
        logger.info("COPYING MODEL TO DEFAULT PATH")
        logger.info("="*80)
        import shutil
        shutil.copy(model_output_path, default_model_path)
        logger.info(f"✓ Copied to: {default_model_path}")
        logger.info(f"✓ Versioned model: {model_output_path}")
        logger.info(f"Flask will load from: {default_model_path}")

        # Save retraining report
        report_path = log_dir / f"retraining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'versioned_model_path': str(model_output_path),
                'default_model_path': str(default_model_path),
                'versioned_directory': str(versioned_model_dir),
                'training_timestamp': training_timestamp,
                'backup_path': str(backup_path) if backup_path else None,
                'results': results
            }, f, indent=2, default=str)
        logger.info(f"Report saved to: {report_path}")

        # Deploy new model to Docker containers
        logger.info("="*80)
        logger.info("DEPLOYING TO DOCKER CONTAINERS")
        logger.info("="*80)

        deploy_script = Path(__file__).parent / "deploy_model.sh"
        model_version = versioned_model_dir.name  # e.g., "recsys-20251116-1520"

        logger.info(f"Model version: {model_version}")
        logger.info(f"Deploy script: {deploy_script}")

        try:
            result = subprocess.run(
                ["sudo", "-E", "sh", str(deploy_script), model_version],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for rolling deployment
            )

            if result.returncode == 0:
                logger.info("="*80)
                logger.info("✓ DOCKER DEPLOYMENT SUCCESSFUL")
                logger.info("="*80)
                if result.stdout:
                    logger.info(f"Deployment output:\n{result.stdout}")
            else:
                logger.error("="*80)
                logger.error("✗ DOCKER DEPLOYMENT FAILED")
                logger.error("="*80)
                logger.error(f"Exit code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output:\n{result.stderr}")
                logger.warning("⚠️  Model trained but not deployed!")
                logger.warning("   Check deploy_model.sh or deploy manually")

        except subprocess.TimeoutExpired:
            logger.error("✗ Docker deployment timed out after 10 minutes")
            logger.warning("⚠️  Model trained but deployment may be incomplete")
        except Exception as e:
            logger.error(f"✗ Docker deployment error: {e}", exc_info=True)
            logger.warning("⚠️  Model trained but not deployed!")

        logger.info("="*80)
        logger.info("AUTOMATED RETRAINING COMPLETE")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        logger.error("AUTOMATED RETRAINING FAILED")

        # Restore backup if training failed
        if backup_path and Path(backup_path).exists():
            import shutil
            shutil.copy(backup_path, model_output_path)
            logger.info(f"Restored model from backup: {backup_path}")

        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
