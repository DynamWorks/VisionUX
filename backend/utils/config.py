import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
from yaml.parser import ParserError

class Config:
    """Configuration management for video analytics"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with environment variable support
        
        Args:
            config_path: Optional path to YAML config file
        """
        self._config = {}
        self._env_prefix = 'DXV_'
        
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
                    
        # Load .env file if present
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
                    
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            self._config_path = config_path
            
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file with environment variable substitution
        
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
            # Read raw YAML content
            with open(config_path, 'r') as f:
                content = f.read()
                
            # Replace environment variables
            import os
            import re
            
            def env_var_replacer(match):
                var_name = match.group(1)
                default = match.group(2) if match.group(2) else None
                return os.getenv(var_name, default if default else '')
                
            # Replace ${VAR:-default} or ${VAR} patterns
            content = re.sub(r'\${([^:-]+)(?::-([^}]+))?}', env_var_replacer, content)
            
            # Parse YAML
            yaml_config = yaml.safe_load(content)
                
            if not isinstance(yaml_config, dict):
                raise ValueError("Config file must contain a YAML dictionary")
                
            # Validate required sections
            required_sections = ['api', 'websocket']
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
            'api': {
                'host': '127.0.0.1',
                'port': 8000,
                'debug': False,
                'cors_origins': '*'
            },
            'websocket': {
                'host': '127.0.0.1',
                'port': 8000,
                'debug': False,
                'max_buffer_size': 104857600,
                'ssl_enabled': False,
                'ping_interval': 25,
                'ping_timeout': 60,
                'max_connections': 100,
                'reconnection': {
                    'attempts': 10,
                    'delay': 1000,
                    'max_delay': 5000
                }
            },
            'storage': {
                'base_path': 'tmp_content',
                'subdirs': ['uploads', 'analysis', 'chat_history', 'visualizations']
            },
            'logging': {
                'level': 'INFO',
                'file': 'video_analytics.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
