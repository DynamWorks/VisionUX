import os
import sys
import yaml

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.app import app

if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get server config
    api_config = config.get("api", {})
    ws_config = config.get("websocket", {})
    
    # Get host and ports with fallback values
    host = api_config.get("host", "localhost")
    try:
        port = int(api_config.get("port", "8000"))
    except ValueError:
        port = 8000
        
    try:
        ws_port = int(ws_config.get("port", "8001"))
    except ValueError:
        ws_port = 8001
        
    debug = api_config.get("debug", False)
    
    # Run server
    app.run(host=host, port=port, ws_port=ws_port, debug=debug)
