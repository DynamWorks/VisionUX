import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class EnvValidator:
    """Utility for validating environment variables"""
    
    REQUIRED_VARS = [
        'API_HOST',
        'API_PORT',
        'OPENAI_API_KEY',
        'GEMINI_API_KEY'
    ]
    
    OPTIONAL_VARS = {
        'API_DEBUG': 'false',
        'API_CORS_ORIGINS': '*',
        'WS_HOST': 'localhost',
        'WS_PORT': '8000',
        'WS_DEBUG': 'false',
        'OPENAI_MODEL': 'gpt-4o-mini',
        'GEMINI_MODEL': 'gemini-1.5-flash',
        'LOG_LEVEL': 'INFO'
    }
    
    @classmethod
    def validate_env(cls) -> Dict[str, str]:
        """Validate environment variables and return config dict"""
        config = {}
        
        # Check required vars
        missing_vars = []
        for var in cls.REQUIRED_VARS:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            config[var] = value
            
        if missing_vars:
            logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        # Set optional vars with defaults
        for var, default in cls.OPTIONAL_VARS.items():
            config[var] = os.getenv(var, default)
            
        return config
