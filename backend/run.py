from app import app
import yaml
import os

if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get server config
    backend_config = config.get("backend", {})
    host = backend_config.get("host", "localhost")
    port = backend_config.get("port", 8000)
    debug = backend_config.get("debug", False)
    
    # Run server
    app.run(host=host, port=port, debug=debug)
