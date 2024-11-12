from flask import Flask
import threading

app = Flask(__name__)
server = None

def is_ready():
    """Check if backend is ready"""
    return True

def start(port=8502):
    """Start the backend server in a separate thread"""
    global server
    
    def run_server():
        try:
            app.run(host='localhost', port=port, debug=False)
        except Exception as e:
            raise RuntimeError(f"Failed to start backend server: {e}")
    
    if server is None:
        server = threading.Thread(target=run_server, daemon=True)
        server.start()
