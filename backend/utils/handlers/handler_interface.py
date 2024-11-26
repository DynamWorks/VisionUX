from abc import ABC, abstractmethod
import logging
from typing import Any, Dict

class MessageHandler(ABC):
    """Base interface for WebSocket message handlers"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def handle(self, websocket: Any, message: Dict) -> None:
        """Handle incoming WebSocket message"""
        pass
    
    async def send_response(self, websocket: Any, data: Dict) -> None:
        """Send response through WebSocket"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            
    async def send_error(self, websocket: Any, error: str) -> None:
        """Send error response"""
        await self.send_response(websocket, {
            'type': 'error',
            'error': str(error)
        })
