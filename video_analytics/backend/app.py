from flask import Flask

app = Flask(__name__)

def is_ready():
    """Check if backend is ready"""
    return True

def start(port=8502):
    """Start the backend server"""
    app.run(port=port)
