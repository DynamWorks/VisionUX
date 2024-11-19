from app import app
import yaml
import os

if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get server config
    api_config = config.get("api", {})
    ws_config = config.get("websocket", {})
    host = api_config.get("host", "localhost")
    port = api_config.get("port", 8000)
    ws_port = ws_config.get("port", 8001)
    debug = api_config.get("debug", False)
    
    # Run server
    app.run(host=host, port=port, ws_port=ws_port, debug=debug)
