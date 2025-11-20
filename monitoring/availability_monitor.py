"""
Availability Monitor
===================

Monitors service availability every 15 seconds using health endpoint only.

Usage:
    python availability_monitor.py
    nohup python availability_monitor.py > availability.log 2>&1 &
"""

import sys
import time
import logging
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [AVAIL] %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'availability_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from metrics_utils import write_metrics_file, format_gauge

CHECK_INTERVAL = 15  # seconds
HEALTH_ENDPOINT = "http://17645-team02.isri.cmu.edu:8082/health"


def check_health_endpoint():
    """
    Check /health endpoint.
    Returns 1 if healthy, 0 if not.
    """
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Check if status is "healthy" and model is loaded
            if data.get('status') == 'healthy' and data.get('model_loaded') == True:
                logger.info(f"Health check: OK (model_version={data.get('model_version', 'unknown')})")
                return 1
            else:
                logger.warning(f"Health check: Unhealthy status or model not loaded: {data}")
                return 0
        else:
            logger.warning(f"Health check: Bad status code {response.status_code}")
            return 0
            
    except requests.exceptions.Timeout:
        logger.error("Health check: Timeout")
        return 0
    except requests.exceptions.ConnectionError:
        logger.error("Health check: Connection error")
        return 0
    except Exception as e:
        logger.error(f"Health check: Error: {e}")
        return 0


def compute_availability():
    """Check health endpoint and write metric."""
    try:
        # Check health endpoint
        health_status = check_health_endpoint()
        
        # Write metric
        metrics = format_gauge('service_health_check', health_status)
        write_metrics_file(metrics)
        
        logger.info(f"✓ Metric updated: health_check={health_status}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def main():
    logger.info("=" * 70)
    logger.info("AVAILABILITY MONITOR - Starting")
    logger.info("=" * 70)
    logger.info(f"Check interval: {CHECK_INTERVAL}s")
    logger.info(f"Health endpoint: {HEALTH_ENDPOINT}")
    logger.info("=" * 70)
    
    logger.info("Entering loop (Ctrl+C to stop)...")
    
    try:
        while True:
            compute_availability()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")


if __name__ == '__main__':
    main()