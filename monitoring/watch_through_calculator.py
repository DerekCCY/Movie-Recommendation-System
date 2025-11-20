"""
Watch-Through Rate Calculator
=============================

Calculates watch-through rate for completed weeks (Thu-Wed).

Usage:
    # First run (uses hardcoded historical value)
    python watch_through_calculator.py --first-run
    
    # Weekly run (samples data for efficiency)
    python watch_through_calculator.py
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import random

import pandas as pd

from metrics_utils import write_metrics_file, format_gauge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WATCH] %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'watch_through.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BRONZE_DIR = Path.home() / 'group-project-f25-the-real-reel-deal' / 'data' / 'bronze'
SAMPLE_RATE = 0.3  # Sample 30% of files for efficiency


def sample_files(files, sample_rate=SAMPLE_RATE):
    """Randomly sample files to reduce computation."""
    if len(files) == 0:
        return []
    
    n_sample = max(1, int(len(files) * sample_rate))
    sampled = random.sample(files, n_sample)
    logger.info(f"Sampled {len(sampled)} files from {len(files)} total ({sample_rate*100}%)")
    return sampled


def load_data_between_dates(folder_name, start_date, end_date, use_sampling=True):
    """Load data between two dates with optional sampling."""
    folder = BRONZE_DIR / folder_name
    if not folder.exists():
        return pd.DataFrame()
    
    # Collect all files
    all_files = list(folder.glob('**/*.parquet'))
    
    if use_sampling and len(all_files) > 100:
        # Sample files if too many
        files_to_load = sample_files(all_files)
    else:
        files_to_load = all_files
    
    dfs = []
    for f in files_to_load:
        try:
            df = pd.read_parquet(f)
            if 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
                df = df[(df['ts'] >= start_date) & (df['ts'] <= end_date)]
            if len(df) > 0:
                dfs.append(df)
        except:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(result)} records from {folder_name} ({start_date.date()} to {end_date.date()})")
    return result


def compute_watch_through_first_run():
    """
    First run: Use hardcoded historical metric from Oct 26 evaluation.
    
    Historical evaluation (Oct 26, 2025):
    - 10K parquet files of watch minutes
    - 10K parquet files of recommendation requests
    - 151,795 total recommendations
    - 4,038 watched through (≥20 min within 7 days)
    - Watch-through rate: 2.66%
    """
    try:
        logger.info("First run: using historical evaluation metric")
        logger.info("Historical evaluation from Oct 26, 2025:")
        logger.info("  Total recommendations: 151,795")
        logger.info("  Watched through: 4,038")
        logger.info("  Watch-through rate: 2.66%")
        
        # Use historical metric
        wtr = 2.66
        
        metrics = format_gauge('model_watch_through_rate', wtr, labels={'week': 'historical-oct26'})
        write_metrics_file(metrics)
        
        logger.info(f"✓ Watch-through rate (historical): {wtr:.2f}%")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def compute_watch_through_weekly():
    """Weekly run: proper logic for last complete week with sampling."""
    try:
        logger.info("Weekly run: computing last complete week (with sampling)")
        
        # Get last complete week (Thu-Wed)
        now = datetime.now(timezone.utc)
        days_since_wed = (now.weekday() - 2) % 7
        if days_since_wed == 0:
            days_since_wed = 7
        
        week_end = now - timedelta(days=days_since_wed)
        week_end = week_end.replace(hour=23, minute=59, second=59)
        week_start = week_end - timedelta(days=6)
        week_start = week_start.replace(hour=0, minute=0, second=0)
        
        # Recommendations: 14 days before week_end (2-week lookback)
        rec_start = week_end - timedelta(days=14)
        rec_end = week_end
        
        logger.info(f"Week: {week_start.date()} to {week_end.date()}")
        logger.info(f"Recs lookback: {rec_start.date()} to {rec_end.date()}")
        
        # Load with sampling
        requests_df = load_data_between_dates('request_events', rec_start, rec_end, use_sampling=True)
        watch_df = load_data_between_dates('watch_events', week_start, week_end, use_sampling=True)
        
        if len(requests_df) == 0:
            logger.warning("No requests")
            return
        
        requests_df = requests_df[
            requests_df['result_raw'].notna() & 
            (requests_df['result_raw'].astype(str).str.strip() != '')
        ].copy()
        
        if len(requests_df) == 0:
            logger.warning("No valid recommendations")
            return
        
        requests_df['movie_ids'] = requests_df['result_raw'].str.split(',')
        recs = requests_df.explode('movie_ids')[['ts', 'user_id', 'movie_ids']].copy()
        recs.rename(columns={'ts': 'rec_ts', 'movie_ids': 'movie_id'}, inplace=True)
        recs['movie_id'] = recs['movie_id'].str.strip()
        recs = recs[recs['movie_id'] != '']
        
        logger.info(f"Analyzing {len(recs)} recommendations (sampled)")
        
        if len(watch_df) == 0:
            logger.warning("No watches")
            week_label = week_start.strftime('%Y-%m-%d')
            metrics = format_gauge('model_watch_through_rate', 0, labels={'week': week_label})
            write_metrics_file(metrics)
            return
        
        watch_summary = watch_df.groupby(['user_id', 'movie_id']).agg(
            first_watch_ts=('ts', 'min'),
            minutes_watched=('minute', 'count')
        ).reset_index()
        
        watch_20min = watch_summary[watch_summary['minutes_watched'] >= 20].copy()
        logger.info(f"Found {len(watch_20min)} movies watched ≥20 min")
        
        merged = recs.merge(
            watch_20min[['user_id', 'movie_id', 'first_watch_ts']],
            on=['user_id', 'movie_id'],
            how='left'
        )
        
        merged['time_diff'] = merged['first_watch_ts'] - merged['rec_ts']
        merged['watched_through'] = (
            merged['first_watch_ts'].notna() & 
            (merged['time_diff'] >= timedelta(0)) & 
            (merged['time_diff'] <= timedelta(days=7))
        )
        
        total = len(merged)
        watched = int(merged['watched_through'].sum())
        wtr = (watched / total * 100) if total > 0 else 0
        
        week_label = week_start.strftime('%Y-%m-%d')
        metrics = format_gauge('model_watch_through_rate', wtr, labels={'week': week_label})
        write_metrics_file(metrics)
        
        logger.info(f"✓ {week_label}: {wtr:.2f}% ({watched}/{total}) [sampled estimate]")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-run', action='store_true', help='Use historical metric')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("WATCH-THROUGH CALCULATOR")
    logger.info("=" * 70)
    
    if args.first_run:
        compute_watch_through_first_run()
    else:
        compute_watch_through_weekly()
    
    logger.info("Complete!")


if __name__ == '__main__':
    main()