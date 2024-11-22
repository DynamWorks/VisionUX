import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
from yaml.parser import ParserError

class Config:
    """Configuration management for video analytics"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional path to YAML config file
        """
        self._config = {}
        
        # Try to find config file in standard locations
        if not config_path:
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'config.yaml'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'config.yaml'),
                'config.yaml'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
                    
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            self._config_path = config_path
            
    def load_config(self, config_path: Union[str, Path]) -> None:
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
                
            # Validate required sections
            required_sections = ['api', 'websocket', 'rerun']
            missing_sections = [s for s in required_sections if s not in yaml_config]
            if missing_sections:
                raise ValueError(f"Missing required config sections: {', '.join(missing_sections)}")
                
            self._update_config(yaml_config)
            
        except ParserError as e:
            raise ValueError(f"Invalid YAML syntax in config file: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except Exception as e:
            raise ValueError(f"Error loading config file: {str(e)}")
            
    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Recursively update configuration with validation
        
        Args:
            new_config: New configuration dictionary to merge
        """
        def _validate_value(value: Any) -> bool:
            if isinstance(value, (dict, list, str, int, float, bool)):
                return True
            return False
            
        for key, value in new_config.items():
            if not _validate_value(value):
                raise ValueError(f"Invalid config value type for {key}: {type(value)}")
                
            if isinstance(value, dict) and key in self._config:
                if not isinstance(self._config[key], dict):
                    self._config[key] = {}
                self._config[key].update(value)
            else:
                self._config[key] = value
                
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        
        Args:
            *keys: Sequence of keys to traverse
            default: Default value if path not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get('api', 'host', default='localhost')
        """
        current = self._config
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current
        
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict syntax"""
        return self._config[key]
        
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using dict syntax"""
        self._config[key] = value
        
    def reload(self) -> None:
        """Reload configuration from file"""
        if hasattr(self, '_config_path') and self._config_path:
            self.load_config(self._config_path)
        
    def reset(self) -> None:
        """Reset configuration to defaults"""
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
        
