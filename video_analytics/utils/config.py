import yaml
from pathlib import Path
from typing import Dict, Optional

class Config:
    """Configuration manager for video analytics"""
    
    DEFAULT_CONFIG = {
        'api': {
            'host': 'localhost',
            'port': 8001,
            'debug': False
        },
        'frontend': {
            'port': 8501
        },
        'backend': {
            'port': 8502
        },
        'logging': {
            'level': 'INFO'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional path to YAML config file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self._load_config(config_path)
            
    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(path) as f:
            file_config = yaml.safe_load(f)
            
        # Update default config with file values
        self._update_config(self.config, file_config)
        
    def _update_config(self, base: Dict, update: Dict) -> None:
        """Recursively update configuration dictionary"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
