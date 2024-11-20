import logging
from abc import ABC, abstractmethod
import json
import websockets

class BaseMessageHandler(ABC):
    """Base class for WebSocket message handlers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    async def handle(self, websocket, message_data):
        """Handle a specific type of message"""
        pass
        
    async def send_response(self, websocket, data):
        """Send a response through the WebSocket"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Connection closed while sending response")
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            
    async def send_error(self, websocket, error_message):
        """Send an error response"""
        await self.send_response(websocket, {
            'type': 'error',
            'error': str(error_message)
        })
import logging
from abc import ABC, abstractmethod
import json
import websockets

class BaseMessageHandler(ABC):
    """Base class for WebSocket message handlers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    async def handle(self, websocket, message_data):
        """Handle a specific type of message"""
        pass
        
    async def send_response(self, websocket, data):
        """Send a response through the WebSocket"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Connection closed while sending response")
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            
    async def send_error(self, websocket, error_message):
        """Send an error response"""
        await self.send_response(websocket, {
            'type': 'error',
            'error': str(error_message)
        })
