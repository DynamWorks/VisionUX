from .base_handler import BaseMessageHandler
from .file_list_handler import FileListHandler
from .video_upload_handler import VideoUploadHandler
import json
import logging

class MessageRouter:
    """Routes WebSocket messages to appropriate handlers"""
    
    def __init__(self, uploads_path):
        self.logger = logging.getLogger(__name__)
        self.handlers = {
            'get_uploaded_files': FileListHandler(uploads_path),
            'video_upload_start': VideoUploadHandler(uploads_path),
            'video_upload_chunk': VideoUploadHandler(uploads_path),
            'video_upload_complete': VideoUploadHandler(uploads_path)
        }
        
    async def route_message(self, websocket, message):
        """Route message to appropriate handler"""
        try:
            if isinstance(message, str):
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type in self.handlers:
                    handler = self.handlers[message_type]
                    await handler.handle(websocket, data)
                else:
                    self.logger.warning(f"No handler for message type: {message_type}")
                    
            elif isinstance(message, bytes):
                # Handle binary data (e.g., video chunks)
                if 'video_upload_chunk' in self.handlers:
                    await self.handlers['video_upload_chunk'].handle(websocket, message)
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
from .base_handler import BaseMessageHandler
from .file_list_handler import FileListHandler
from .video_upload_handler import VideoUploadHandler
import json
import logging

class MessageRouter:
    """Routes WebSocket messages to appropriate handlers"""
    
    def __init__(self, uploads_path):
        self.logger = logging.getLogger(__name__)
        self.handlers = {
            'get_uploaded_files': FileListHandler(uploads_path),
            'video_upload_start': VideoUploadHandler(uploads_path),
            'video_upload_chunk': VideoUploadHandler(uploads_path),
            'video_upload_complete': VideoUploadHandler(uploads_path)
        }
        
    async def route_message(self, websocket, message):
        """Route message to appropriate handler"""
        try:
            if isinstance(message, str):
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type in self.handlers:
                    handler = self.handlers[message_type]
                    await handler.handle(websocket, data)
                else:
                    self.logger.warning(f"No handler for message type: {message_type}")
                    
            elif isinstance(message, bytes):
                # Handle binary data (e.g., video chunks)
                if 'video_upload_chunk' in self.handlers:
                    await self.handlers['video_upload_chunk'].handle(websocket, message)
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
