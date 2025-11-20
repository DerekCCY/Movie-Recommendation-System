#!/bin/bash
# Setup script for Ubuntu VM deployment

echo "Setting up Movie Recommendation API on VM..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip if not already installed
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies
sudo apt install -y build-essential python3-dev

# Create project directory
mkdir -p /home/team02/deployment
cd /home/team02/deployment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p models data logs

# Set up systemd service (optional)
cat > recommendation-api.service << EOF
[Unit]
Description=Movie Recommendation API
After=network.target

[Service]
Type=simple
User=team02
WorkingDirectory=/home/team02/deployment
Environment=PATH=/home/team02/deployment/venv/bin
ExecStart=/home/team02/deployment/venv/bin/python run_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Setup complete!"
echo "Next steps:"
echo "1. Copy SVD model to models/svd_model_colab.pkl"
echo "2. Copy interactions.parquet to data/"
echo "3. Run: python run_server.py"
echo "4. Test: curl http://localhost:8082/health"