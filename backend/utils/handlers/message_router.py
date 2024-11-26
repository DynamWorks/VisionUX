from .handler_interface import MessageHandler
from .file_list_handler import FileListHandler
from .video_upload_handler import VideoUploadHandler
from .camera_stream_handler import CameraStreamHandler
from .progress_handler import ProgressHandler
import json
import logging

class MessageRouter:
    """Routes WebSocket messages to appropriate handlers"""
    
    def __init__(self, uploads_path, config=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.handlers = {
            'get_uploaded_files': FileListHandler(uploads_path),
            'video_upload_start': VideoUploadHandler(uploads_path),
            'video_upload_chunk': VideoUploadHandler(uploads_path),
            'video_upload_complete': VideoUploadHandler(uploads_path),
            'upload_progress': VideoUploadHandler(uploads_path),
            'camera_frame': CameraStreamHandler(),
            'start_camera_stream': CameraStreamHandler(),
            'start_video_stream': CameraStreamHandler(),
            'stop_video_stream': CameraStreamHandler(),
            'pause_video_stream': CameraStreamHandler(),
            'resume_video_stream': CameraStreamHandler(),
            'trigger_scene_analysis': CameraStreamHandler(),
            'toggle_edge_detection': CameraStreamHandler()
        }
        
    async def route_message(self, websocket, message):
        """Route message to appropriate handler"""
        try:
            if isinstance(message, str):
                data = json.loads(message) if isinstance(message, str) else message
                message_type = data.get('type')
                self.logger.info(f"Routing message type: {message_type}")
                
                if message_type in self.handlers:
                    handler = self.handlers[message_type]
                    self.logger.debug(f"Found handler for {message_type}")
                    try:
                        await handler.handle(websocket, data)
                        self.logger.info(f"Successfully handled {message_type}")
                    except Exception as e:
                        self.logger.error(f"Handler error for {message_type}: {e}")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'error': f'Handler error: {str(e)}'
                        }))
                else:
                    self.logger.warning(f"No handler for message type: {message_type}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'error': f'No handler for message type: {message_type}'
                    }))
                    
            elif isinstance(message, bytes):
                # Handle binary data (e.g., video chunks)
                if 'video_upload_chunk' in self.handlers:
                    await self.handlers['video_upload_chunk'].handle(websocket, message)
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
