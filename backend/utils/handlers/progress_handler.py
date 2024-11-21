from .base_handler import BaseMessageHandler
import json
import logging

class ProgressHandler(BaseMessageHandler):
    """Handles progress update messages"""
    
    def __init__(self):
        super().__init__()
        
    async def handle(self, websocket, message_data):
        """Handle progress update message"""
        try:
            # Forward progress update to client
            await self.send_response(websocket, {
                'type': 'upload_progress',
                'progress': message_data.get('progress', 0),
                'chunk': message_data.get('chunk'),
                'totalChunks': message_data.get('totalChunks')
            })
        except Exception as e:
            self.logger.error(f"Error handling progress update: {e}")
            await self.send_error(websocket, str(e))
