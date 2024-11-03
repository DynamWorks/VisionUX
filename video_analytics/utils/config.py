import os
from typing import Dict, Any
import yaml

class Config:
    """Configuration management for video analytics"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional path to YAML config file
        """
        self.config = {
            'models': {
                'clip': {
                    'name': 'openai/clip-vit-base-patch32',
                    'local_path': 'video_analytics/models/clip-vit-base-patch32'
                },
                'yolo': {
                    'name': 'yolov8x.pt',
                    'local_path': 'video_analytics/models/yolo/yolov8x.pt'
                },
                'traffic_signs': {
                    'name': 'yolov8n.pt',
                    'local_path': 'video_analytics/models/traffic_signs/yolov8n.pt'
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
            
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self._update_config(yaml_config)
        except Exception as e:
            raise ValueError(f"Error loading config file: {str(e)}")
            
    def _update_config(self, new_config: Dict[str, Any]):
        """Recursively update configuration"""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
        
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict syntax"""
        return self.config[key]
        
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using dict syntax"""
        self.config[key] = value
