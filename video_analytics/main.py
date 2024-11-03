from flask import Flask
from .api.routes import api
from .utils.config import Config
import logging

def create_app(config_path: str = None) -> Flask:
    """
    Create and configure Flask application
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configured Flask application
    """
    # Initialize app
    app = Flask(__name__)
    
    # Load configuration
    config = Config(config_path)
    app.config.update(config.config)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    
    return app

def main():
    """Run the application"""
    import argparse
    parser = argparse.ArgumentParser(description='Video Analytics API')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    app = create_app(args.config)
    app.run(
        host=app.config['api']['host'],
        port=app.config['api']['port'],
        debug=app.config['api']['debug']
    )

if __name__ == '__main__':
    main()
