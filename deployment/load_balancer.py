from flask import Flask, request
import itertools
import requests
import os

app = Flask(__name__)

BACKEND_SERVERS = [
    "http://localhost:8083",
    "http://localhost:8084",
    "http://localhost:8085"
]

server_pool = itertools.cycle(BACKEND_SERVERS)

@app.route('/health', methods=['GET'])
def health_check():
    for i in range(3):
        backend_url = next(server_pool)
        try:
            response = requests.get(f"{backend_url}/health")
            data = response.json()
            return data, response.status_code
        except ValueError:
            return {"error": "Invalid JSON from backend", "raw": response.text}, 502
        except:
            continue
    return {"error": "network error"}, 500

@app.route('/recommend/<user_id>', methods=['GET'])
def recommend_movies(user_id):
    for i in range(3):
        backend_url = next(server_pool)
        try:
            response = requests.get(f"{backend_url}/recommend/{user_id}")
            data = response.text
            return data, response.status_code
        except:
            continue
    return {"error": "network error"}, 500

@app.route('/', methods=['GET'])
def root():
    backend_url = next(server_pool)
    for i in range(3):
        try:
            response = requests.get(f"{backend_url}/")
            data = response.json()
            return data, response.status_code
        except ValueError:
            return {"error": "Invalid JSON from backend", "raw": response.text}, 502
        except:
            continue
    return {"error": "network error"}, 500
    
if __name__ == '__main__':
    # Run on port 8082 as required
    port = int(os.getenv('PORT', 8082))
    app.run(host='0.0.0.0', port=port, debug=False)
