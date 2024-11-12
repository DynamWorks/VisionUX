from flask import Flask, jsonify
from video_analytics.api.routes import api
from .utils.config import Config
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
        
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('video_analytics.log')
        ]
    )

def create_app(config_path: Optional[str] = None) -> Flask:
    """Create and configure Flask application
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configured Flask application
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Initialize app
    app = Flask(__name__)
    
    try:
        # Load configuration
        if config_path:
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
        config = Config(config_path)
        app.config.update(config.config)
        
        # Configure logging
        setup_logging(app.config.get('logging', {}).get('level', 'INFO'))
        
        # Error handlers
        @app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404
            
        @app.errorhandler(500)
        def server_error(error):
            return jsonify({'error': 'Internal server error'}), 500
        
        # Health check endpoint
        @app.route('/api/health')
        def health_check():
            return jsonify({
                'status': 'ok',
                'version': '1.0.0'
            }), 200

        # Register blueprints
        app.register_blueprint(api, url_prefix='/api')
        
        logging.info(f"Application created successfully with config: {config_path}")
        return app
        
    except Exception as e:
        logging.error(f"Failed to create application: {str(e)}")
        raise

def main():
    """Run the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Analytics API Server')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--host', type=str, help='Host to run server on')
    parser.add_argument('--port', type=int, help='Port to run server on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    try:
        app = create_app(args.config)
        
        # Command line args override config file
        if args.host:
            app.config['api']['host'] = args.host
        if args.port:
            app.config['api']['port'] = args.port
        if args.debug:
            app.config['api']['debug'] = True
            
        setup_logging(args.log_level)
        
        logging.info(f"Starting server on {app.config['api']['host']}:{app.config['api']['port']}")
        
        app.run(
            host=app.config['api']['host'],
            port=app.config['api']['port'],
            debug=app.config['api']['debug']
        )
        
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
