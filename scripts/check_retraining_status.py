#!/usr/bin/env python3
"""
Check status of automated retraining
Shows: last run time, success/failure, model versions, next scheduled run

Usage: python scripts/check_retraining_status.py
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    print("="*60)
    print("AUTOMATED RETRAINING STATUS")
    print("="*60)

    # Check registry for latest models
    registry_path = Path("models/registry.json")
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                registry = json.load(f)

            candidate = registry.get('candidate', 'None')
            production = registry.get('production', 'None')

            print(f"\nCandidate Model: {candidate}")
            print(f"Production Model: {production}")

            if candidate != production:
                print("WARNING: Candidate model not yet promoted to production")
            else:
                print("Status: Candidate and production are in sync")
        except Exception as e:
            print(f"\nERROR reading registry: {e}")
    else:
        print("\nERROR: No registry found at models/registry.json")

    # Check recent training logs
    log_dir = Path("logs/retraining")
    if log_dir.exists():
        log_files = sorted(log_dir.glob("retraining_*.log"), reverse=True)

        if log_files:
            latest_log = log_files[0]
            print(f"\nLatest Training Log: {latest_log.name}")

            try:
                with open(latest_log) as f:
                    content = f.read()

                # Check training status
                if "AUTOMATED RETRAINING COMPLETE" in content:
                    print("Last Training: SUCCESS")
                elif "RETRAINING FAILED" in content or "Training failed" in content:
                    print("Last Training: FAILED")
                else:
                    print("Last Training: IN PROGRESS or UNKNOWN")

                # Extract timestamp
                lines = content.split('\n')
                for line in lines[:10]:
                    if "AUTOMATED RETRAINING STARTED" in line:
                        timestamp_part = line.split(' [INFO]')[0] if ' [INFO]' in line else line.split(' -')[0]
                        print(f"Started At: {timestamp_part}")
                        break

                # Extract RMSE if available
                for line in lines:
                    if "RMSE:" in line and "[INFO]" in line:
                        rmse_value = line.split("RMSE:")[1].strip()
                        print(f"Model RMSE: {rmse_value}")
                        break

            except Exception as e:
                print(f"ERROR reading log: {e}")
        else:
            print("\nNo training logs found")
    else:
        print("\nNo training logs directory found")

    # Check for recent retraining reports
    report_dir = Path("logs/retraining")
    if report_dir.exists():
        report_files = sorted(report_dir.glob("retraining_report_*.json"), reverse=True)
        if report_files:
            latest_report = report_files[0]
            print(f"\nLatest Report: {latest_report.name}")
            try:
                with open(latest_report) as f:
                    report = json.load(f)
                print(f"Timestamp: {report.get('timestamp', 'Unknown')}")
                print(f"Model Path: {report.get('model_path', 'Unknown')}")
            except:
                pass

    # Check cron job installation
    print("\n" + "-"*60)
    print("CRON JOB STATUS")
    print("-"*60)
    try:
        result = subprocess.run(
            ['crontab', '-l'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            cron_lines = [line for line in result.stdout.split('\n') if 'automated_retraining.py' in line]

            if cron_lines:
                print("Cron Job: INSTALLED")
                for line in cron_lines:
                    # Extract schedule (first 5 fields)
                    parts = line.split()
                    if len(parts) >= 5:
                        schedule = ' '.join(parts[:5])
                        print(f"Schedule: {schedule}")
                        print(f"Full Entry: {line}")
            else:
                print("Cron Job: NOT INSTALLED")
                print("Run: ./scripts/setup_cron.sh to install")
        else:
            print("Cron Job: ERROR checking crontab")
    except FileNotFoundError:
        print("Cron Job: Not available (crontab command not found)")
    except Exception as e:
        print(f"Cron Job: ERROR ({e})")

    # Check production model file
    print("\n" + "-"*60)
    print("PRODUCTION MODEL STATUS")
    print("-"*60)
    production_model = Path("models/production/model.pkl")
    if production_model.exists():
        size_mb = production_model.stat().st_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(production_model.stat().st_mtime)
        print(f"Model File: EXISTS")
        print(f"Size: {size_mb:.2f} MB")
        print(f"Last Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Check version file
        version_file = Path("models/production/current_version.txt")
        if version_file.exists():
            version = version_file.read_text().strip()
            print(f"Version: {version}")
    else:
        print("Model File: NOT FOUND")
        print("Run automated_retraining.py and deploy_model.sh")

    print("\n" + "="*60)
    print("END OF STATUS REPORT")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
