#!/bin/bash
# Setup cron job for automated retraining
# Run this once on the VM to install the cron job
# Schedule: Every 3 days at 2:00 AM

# Get absolute paths
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_PATH="$PROJECT_DIR/scripts/automated_retraining.py"
PYTHON_PATH=$(which python3 || which python)
LOG_DIR="$PROJECT_DIR/logs/cron"

echo "============================================================"
echo "AUTOMATED RETRAINING CRON JOB SETUP"
echo "============================================================"

# Verify script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Retraining script not found at $SCRIPT_PATH"
    exit 1
fi

# Verify python exists
if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: Python not found in PATH"
    exit 1
fi

echo "Project Directory: $PROJECT_DIR"
echo "Script Path: $SCRIPT_PATH"
echo "Python Path: $PYTHON_PATH"

# Create log directory
mkdir -p "$LOG_DIR"
echo "Log Directory: $LOG_DIR"

# Cron schedule: Every 3 days at 2:00 AM
CRON_SCHEDULE="0 2 */3 * *"

# Create cron entry
CRON_ENTRY="$CRON_SCHEDULE cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/cron_retraining.log 2>&1"

echo ""
echo "Cron Schedule: $CRON_SCHEDULE (Every 3 days at 2:00 AM)"
echo "Full Cron Entry:"
echo "  $CRON_ENTRY"
echo ""

# Check if cron job already exists
crontab -l 2>/dev/null | grep -q "automated_retraining.py"
if [ $? -eq 0 ]; then
    echo "WARNING: Cron job already exists!"
    echo ""
    echo "Current crontab entries containing 'automated_retraining.py':"
    crontab -l 2>/dev/null | grep "automated_retraining.py"
    echo ""
    echo "To remove existing entry:"
    echo "  1. Run: crontab -e"
    echo "  2. Delete the line containing 'automated_retraining.py'"
    echo "  3. Save and exit"
    echo "  4. Run this script again"
    exit 1
fi

# Add to crontab
echo "Installing cron job..."
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

# Verify installation
if crontab -l 2>/dev/null | grep -q "automated_retraining.py"; then
    echo "SUCCESS: Cron job installed!"
    echo ""
    echo "============================================================"
    echo "CRON JOB DETAILS"
    echo "============================================================"
    echo "Schedule: Every 3 days at 2:00 AM"
    echo "Script: $SCRIPT_PATH"
    echo "Logs: $LOG_DIR/cron_retraining.log"
    echo ""
    echo "Verify installation:"
    echo "  crontab -l"
    echo ""
    echo "View logs:"
    echo "  tail -f $LOG_DIR/cron_retraining.log"
    echo ""
    echo "Remove cron job:"
    echo "  crontab -e  (then delete the line)"
    echo ""
    echo "Check retraining status:"
    echo "  python scripts/check_retraining_status.py"
    echo "============================================================"
else
    echo "ERROR: Failed to install cron job"
    exit 1
fi
