import os
import sys
import yaml
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Initialize eventlet at the very beginning
import eventlet
eventlet.monkey_patch(socket=True, select=True, thread=True)

from backend.app import backend_app

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the video analytics backend server')
    parser.add_argument('--config', type=str, 
                       default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                       help='Path to config.yaml file')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Get server config
    api_config = config.get("api", {})
    ws_config = config.get("websocket", {})
    
    # Force local IP address
    host = "127.0.0.1"
    try:
        port = int(api_config.get("port", "8000"))
    except ValueError:
        port = 8000
        
    debug = api_config.get("debug", False)
    
    try:
        # Run the application
        backend_app.run(host=host, port=port, debug=debug)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
