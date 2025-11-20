import pandas as pd
import glob
from datetime import timedelta
import sys

print("=" * 70)
print("WATCH-THROUGH RATE CALCULATION (SAMPLED)")
print("=" * 70)

print("\n[Step 1] Loading recommendation requests...")
requests_files = sorted(glob.glob("bronze/requests/*.parquet"))
if not requests_files:
    print("ERROR: No parquet files found in bronze/requests/")
    sys.exit(1)

print(f"Found {len(requests_files)} request files")
requests_files = requests_files[-10000:]
print(f"Using latest {len(requests_files)} request files")

requests_dfs = []
errors = 0
total_rows_skipped = 0
for f in requests_files:
    try:
        df = pd.read_parquet(f)
        original_len = len(df)
        if 'ts' in df.columns:
            valid_mask = (df['ts'] >= pd.Timestamp('1970-01-01')) & (df['ts'] <= pd.Timestamp('2100-01-01'))
            df = df[valid_mask].copy()
            df['ts'] = df['ts'].astype(str)
            total_rows_skipped += (original_len - len(df))
        if len(df) > 0:
            requests_dfs.append(df)
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"Warning: Skipping file: {str(e)[:100]}")

if not requests_dfs:
    print("ERROR: No valid request data loaded")
    sys.exit(1)

requests_df = pd.concat(requests_dfs, ignore_index=True)
requests_df['ts'] = pd.to_datetime(requests_df['ts'])
print(f"Total request rows: {len(requests_df):,} (skipped {errors} files, {total_rows_skipped:,} bad rows)")

requests_df = requests_df[requests_df['result_raw'].notna() & (requests_df['result_raw'].astype(str).str.strip() != '')]
print(f"Requests with recommendations: {len(requests_df):,}")

if len(requests_df) == 0:
    print("ERROR: No recommendations found in result_raw. Check data format.")
    sys.exit(1)

print("Parsing and exploding recommendations...")
requests_df['movie_ids'] = requests_df['result_raw'].str.split(',')
recommendations = requests_df.explode('movie_ids')[['ts', 'user_id', 'movie_ids']].copy()
recommendations.rename(columns={'ts': 'recommendation_ts', 'movie_ids': 'movie_id'}, inplace=True)
recommendations['movie_id'] = recommendations['movie_id'].str.strip()
recommendations = recommendations[recommendations['movie_id'] != '']

print(f"Total recommendations (exploded): {len(recommendations):,}")
print(f"Unique users who got recommendations: {recommendations['user_id'].nunique():,}")

print("\n[Step 2] Loading watch events...")
watch_files = sorted(glob.glob("bronze/watch_events/*.parquet"))
if not watch_files:
    print("ERROR: No parquet files found in bronze/watch_events/")
    sys.exit(1)

print(f"Found {len(watch_files)} watch event files")
watch_files = watch_files[-10000:]
print(f"Using latest {len(watch_files)} watch event files")

watch_dfs = []
errors = 0
total_rows_skipped = 0
for i, f in enumerate(watch_files):
    try:
        df = pd.read_parquet(f)
        original_len = len(df)
        if 'ts' in df.columns:
            valid_mask = (df['ts'] >= pd.Timestamp('1970-01-01')) & (df['ts'] <= pd.Timestamp('2100-01-01'))
            df = df[valid_mask].copy()
            df['ts'] = df['ts'].astype(str)
            total_rows_skipped += (original_len - len(df))
        if len(df) > 0:
            watch_dfs.append(df)
    except Exception as e:
        errors += 1
        if errors <= 10:
            print(f"Warning: Skipping file: {str(e)[:80]}")
    
    if (i + 1) % 5000 == 0:
        print(f"Loaded {i + 1}/{len(watch_files)} files (skipped {total_rows_skipped:,} bad rows so far)...")

if not watch_dfs:
    print("ERROR: No valid watch data loaded")
    sys.exit(1)

print("Concatenating watch events...")
watch_df = pd.concat(watch_dfs, ignore_index=True)
watch_df['ts'] = pd.to_datetime(watch_df['ts'])
print(f"Total watch event rows: {len(watch_df):,} (skipped {errors} files, {total_rows_skipped:,} bad rows)")

print("Grouping watch events by user and movie...")
watch_summary = watch_df.groupby(['user_id', 'movie_id']).agg(first_watch_ts=('ts', 'min'), minutes_watched=('minute', 'count')).reset_index()

print(f"Unique (user, movie) watch combinations: {len(watch_summary):,}")

watch_summary_20min = watch_summary[watch_summary['minutes_watched'] >= 20].copy()
print(f"Movies watched ≥20 minutes: {len(watch_summary_20min):,}")

print("\n[Step 3] Joining recommendations with watch events...")

merged = recommendations.merge(watch_summary_20min[['user_id', 'movie_id', 'first_watch_ts', 'minutes_watched']], on=['user_id', 'movie_id'], how='left')

print(f"Merged rows: {len(merged):,}")

merged['time_diff'] = merged['first_watch_ts'] - merged['recommendation_ts']
merged['watched_through'] = (merged['first_watch_ts'].notna() & (merged['time_diff'] >= timedelta(0)) & (merged['time_diff'] <= timedelta(days=7)))

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

total_recommendations = len(merged)
watched_through = merged['watched_through'].sum()
watch_through_rate = (watched_through / total_recommendations * 100) if total_recommendations > 0 else 0

print(f"\nTotal Recommendations: {total_recommendations:,}")
print(f"Watched Through (≥20 min within 7 days): {watched_through:,}")
print(f"Watch-Through Rate: {watch_through_rate:.2f}%")

print(f"\n--- Additional Insights ---")
print(f"Unique users who received recommendations: {merged['user_id'].nunique():,}")
print(f"Unique movies recommended: {merged['movie_id'].nunique():,}")
print(f"Avg recommendations per user: {total_recommendations / merged['user_id'].nunique():.1f}")

watched_recs = merged[merged['watched_through']]
if len(watched_recs) > 0:
    print(f"Avg minutes watched (for watched-through movies): {watched_recs['minutes_watched'].mean():.1f}")
    print(f"Median time to watch after recommendation: {watched_recs['time_diff'].median()}")

output_file = "watch_through_rate_results.csv"
merged.to_csv(output_file, index=False)
print(f"\nDetailed results saved to: {output_file}")

print("\n" + "=" * 70)
print("CALCULATION COMPLETE")
print("=" * 70)