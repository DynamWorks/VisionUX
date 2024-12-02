import logging
from typing import Any, Dict, Optional
from .handler_interface import HandlerInterface

class BaseHandler(HandlerInterface):
    """Base class for all handlers implementing common functionality"""
    
    def __init__(self, name: str, handler_type: str):
        self.name = name
        self.type = handler_type
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.status = {
            'active': False,
            'error': None,
            'last_handled': None
        }

    def handle(self, data: Any) -> Dict:
        """Base handle method with error handling"""
        try:
            if not self.validate(data):
                raise ValueError("Invalid data format")
            
            self.status['active'] = True
            result = self._handle_impl(data)
            self.status['last_handled'] = result.get('timestamp')
            self.status['error'] = None
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling data: {e}", exc_info=True)
            self.status['error'] = str(e)
            raise

    def validate(self, data: Any) -> bool:
        """Base validation method"""
        return data is not None

    def get_name(self) -> str:
        """Get handler name"""
        return self.name

    def get_type(self) -> str:
        """Get handler type"""
        return self.type

    def get_status(self) -> Dict:
        """Get handler status"""
        return self.status

    def cleanup(self) -> None:
        """Base cleanup method"""
        self.status['active'] = False
        self.status['error'] = None
        self.status['last_handled'] = None

    def _handle_impl(self, data: Any) -> Dict:
        """Implementation specific handling - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _handle_impl")

    def _update_status(self, updates: Dict) -> None:
        """Update handler status"""
        self.status.update(updates)

    def _log_error(self, error: Exception, message: str = "Error in handler") -> None:
        """Log error and update status"""
        self.logger.error(f"{message}: {error}", exc_info=True)
        self.status['error'] = str(error)

    def _reset_error(self) -> None:
        """Reset error status"""
        self.status['error'] = None

    def _is_active(self) -> bool:
        """Check if handler is active"""
        return self.status['active']

    def _set_active(self, active: bool) -> None:
        """Set handler active state"""
        self.status['active'] = active
