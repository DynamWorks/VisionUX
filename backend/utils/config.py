import os
from pathlib import Path
from typing import Dict, Any
import yaml

class Config:
    """Configuration management for video analytics"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional path to YAML config file. Environment variables
                        override config file values when prefixed with VIDEO_ANALYTICS_.
        """
        # Initialize default config
        self._config = {
            'models': {
                'clip': {
                    'name': 'openai/clip-vit-base-patch32',
                    'local_path': 'video_analytics/models/clip'
                },
                'yolo': {
                    'name': 'yolov8x.pt',
                    'local_path': 'video_analytics/models/yolo'
                },
                'traffic_signs': {
                    'name': 'yolov8n.pt',
                    'local_path': 'video_analytics/models/traffic_signs'
                }
            },
            'processing': {
                'sample_rate': 1,
                'max_workers': 4,
                'confidence_threshold': 0.5
            },
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            }
        }
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML config file
            
        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If config structure is invalid
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if not isinstance(yaml_config, dict):
                    raise ValueError("Config file must contain a YAML dictionary")
                self._update_config(yaml_config)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading config file: {str(e)}")
            
    def _update_config(self, new_config: Dict[str, Any]):
        """Recursively update configuration"""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self.config:
                self._config[key].update(value)
            else:
                self._config[key] = value
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
        
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict syntax"""
        return self._config[key]
        
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using dict syntax"""
        self._config[key] = value
        
    def _load_env_vars(self):
        """Load configuration from environment variables"""
        import os
        
        # Look for environment variables prefixed with VIDEO_ANALYTICS_
        prefix = 'VIDEO_ANALYTICS_'
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert key to config path (e.g., VIDEO_ANALYTICS_API_PORT -> api.port)
                config_path = key[len(prefix):].lower().replace('_', '.')
                
                # Try to convert value to appropriate type
                try:
                    # Try to parse as int
                    value = int(value)
                except ValueError:
                    try:
                        # Try to parse as float
                        value = float(value)
                    except ValueError:
                        # Try to parse as boolean
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                
                # Set value in config
                current = self._config
                *parts, last = config_path.split('.')
                for part in parts:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[last] = value
