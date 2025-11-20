"""
Metrics Server
=============

HTTP server that exposes metrics.prom file to Prometheus.

Usage:
    python metrics_server.py
    nohup python metrics_server.py > metrics_server.log 2>&1 &
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SERVER] %(message)s'
)
logger = logging.getLogger(__name__)

METRICS_FILE = Path(__file__).parent / 'metrics.prom'
PORT = 8765


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            try:
                if METRICS_FILE.exists():
                    with open(METRICS_FILE, 'r') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; version=0.0.4')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'# No metrics available yet\n')
            except Exception as e:
                logger.error(f"Error: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress verbose logging


def main():
    logger.info("=" * 70)
    logger.info("METRICS SERVER - Starting")
    logger.info("=" * 70)
    logger.info(f"Metrics file: {METRICS_FILE}")
    logger.info(f"Port: {PORT}")
    logger.info("=" * 70)
    
    server = HTTPServer(('0.0.0.0', PORT), MetricsHandler)
    logger.info(f"✓ Server running on http://0.0.0.0:{PORT}/metrics")
    logger.info("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")


if __name__ == '__main__':
    main()