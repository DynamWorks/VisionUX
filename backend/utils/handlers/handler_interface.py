from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class HandlerInterface(ABC):
    """Interface for all handlers"""
    
    @abstractmethod
    def handle(self, data: Any) -> Dict:
        """Handle incoming data and return response"""
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate incoming data"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get handler name"""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Get handler type"""
        pass

    @abstractmethod
    def get_status(self) -> Dict:
        """Get handler status"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup handler resources"""
        pass
