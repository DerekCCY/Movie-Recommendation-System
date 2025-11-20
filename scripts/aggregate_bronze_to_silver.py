"""
Aggregate Bronze to Silver Data Layer

Merges bronze Parquet files from Kafka ingestion into the silver layer.
This script should run before automated retraining to ensure fresh data.

Usage:
    python scripts/aggregate_bronze_to_silver.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_pipeline.data_io_refactored.silver_cleaning import clean_interactions_for_silver
from ml_pipeline.config import BRONZE_DIR, SILVER_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Aggregate bronze data to silver layer."""
    logger.info("="*80)
    logger.info("AGGREGATING BRONZE → SILVER DATA LAYER")
    logger.info("="*80)

    bronze_path = Path(BRONZE_DIR)
    silver_path = Path(SILVER_DIR)
    output_file = silver_path / "interactions.parquet"

    # Check if bronze directory exists and has data
    if not bronze_path.exists():
        logger.warning(f"Bronze directory does not exist: {bronze_path}")
        logger.warning("Skipping aggregation - no new data to process")
        return

    # Check for bronze data files
    bronze_files = list(bronze_path.rglob("*.parquet"))
    if not bronze_files:
        logger.warning(f"No Parquet files found in bronze directory: {bronze_path}")
        logger.warning("Skipping aggregation - no new data to process")
        return

    logger.info(f"Found {len(bronze_files)} Parquet files in bronze layer")
    logger.info(f"Bronze directory: {bronze_path}")
    logger.info(f"Output file: {output_file}")

    try:
        # Run aggregation
        clean_interactions_for_silver(
            bronze_dir=str(bronze_path),
            output_path=str(output_file),
            report_quality=True
        )

        logger.info("="*80)
        logger.info("✓ SILVER LAYER UPDATED SUCCESSFULLY")
        logger.info("="*80)

        # Report output file stats
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Output file size: {size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Failed to aggregate bronze to silver: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
