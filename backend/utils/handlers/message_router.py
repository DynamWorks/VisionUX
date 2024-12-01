import logging
from typing import Dict, Any
from .handler_interface import HandlerInterface
from .file_list_handler import FileListHandler
from .video_upload_handler import VideoUploadHandler
from .progress_handler import ProgressHandler
from .camera_stream_handler import CameraStreamHandler

class MessageRouter:
    """Routes messages to appropriate handlers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.handlers: Dict[str, HandlerInterface] = {}
        self.progress_handler = ProgressHandler()
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Initialize all handlers"""
        try:
            # Initialize core handlers
            self.handlers['progress'] = self.progress_handler
            self.handlers['file_list'] = FileListHandler()
            self.handlers['video_upload'] = VideoUploadHandler(
                progress_handler=self.progress_handler
            )
            
            # Initialize stream handler
            stream_handler = CameraStreamHandler(
                progress_handler=self.progress_handler
            )
            self.handlers['camera_stream'] = stream_handler

            # Register stream control handlers
            self.handlers['start_stream'] = stream_handler
            self.handlers['stop_stream'] = stream_handler
            self.handlers['pause_stream'] = stream_handler
            self.handlers['resume_stream'] = stream_handler

            self.logger.info("All handlers initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing handlers: {e}", exc_info=True)
            raise

    def route_message(self, message_type: str, data: Any) -> Dict:
        """Route message to appropriate handler"""
        try:
            if message_type not in self.handlers:
                raise ValueError(f"Unknown message type: {message_type}")

            handler = self.handlers[message_type]
            operation_id = f"{message_type}_{id(data)}"
            self.progress_handler.start_operation(operation_id, message_type)

            try:
                # Special handling for stream control messages
                if message_type.endswith('_stream'):
                    if message_type == 'start_stream':
                        result = handler.start_stream(data)
                    elif message_type == 'stop_stream':
                        result = handler.stop_stream()
                    elif message_type == 'pause_stream':
                        result = handler.pause_stream()
                    elif message_type == 'resume_stream':
                        result = handler.resume_stream()
                    else:
                        result = handler.handle(data)
                else:
                    # Validate data for non-stream messages
                    if not handler.validate(data):
                        raise ValueError(f"Invalid data for handler {message_type}")
                    result = handler.handle(data)

                # If confirmation provided and there's a pending action, execute it
                if state.get('confirmed') and state.get('pending_action'):
                    tool = self._get_tool(state['pending_action'])
                    if tool:
                        try:
                            tool_result = tool.run(state.get('tool_input', {}))
                            state['action_executed'] = True
                            state['executed_action'] = state['pending_action']
                            state['response'] = f"Action completed: {tool_result}"
                        except Exception as e:
                            state['response'] = f"Error executing action: {str(e)}"
                            state['action_executed'] = False
                    else:
                        state['response'] = f"Tool {state['pending_action']} not found"
                        state['action_executed'] = False

                if result.get('status') == 'error':
                    self.progress_handler.fail_operation(
                        operation_id,
                        result.get('error', 'Unknown error')
                    )
                else:
                    self.progress_handler.complete_operation(
                        operation_id,
                        f"Successfully handled {message_type}"
                    )
                return result
            except Exception as e:
                self.progress_handler.fail_operation(operation_id, str(e))
                raise

        except Exception as e:
            self.logger.error(f"Error routing message: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'message_type': message_type
            }

    def get_handler(self, handler_type: str) -> HandlerInterface:
        """Get handler by type"""
        return self.handlers.get(handler_type)

    def get_handler_status(self, handler_type: str) -> Dict:
        """Get status of a specific handler"""
        handler = self.get_handler(handler_type)
        if not handler:
            return {'error': f'Handler not found: {handler_type}'}
        return handler.get_status()

    def get_all_handlers_status(self) -> Dict[str, Dict]:
        """Get status of all handlers"""
        return {
            handler_type: handler.get_status()
            for handler_type, handler in self.handlers.items()
        }

    def cleanup_handlers(self) -> None:
        """Cleanup all handlers"""
        for handler in self.handlers.values():
            try:
                handler.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up handler: {e}")

    def register_handler(self, handler_type: str, handler: HandlerInterface) -> None:
        """Register a new handler"""
        if handler_type in self.handlers:
            raise ValueError(f"Handler type already registered: {handler_type}")
        self.handlers[handler_type] = handler
        self.logger.info(f"Registered new handler: {handler_type}")

    def unregister_handler(self, handler_type: str) -> None:
        """Unregister a handler"""
        if handler_type in self.handlers:
            handler = self.handlers[handler_type]
            handler.cleanup()
            del self.handlers[handler_type]
            self.logger.info(f"Unregistered handler: {handler_type}")

    def get_active_operations(self) -> Dict[str, Dict]:
        """Get all active operations"""
        return self.progress_handler.get_active_operations()

    def get_handler_stats(self) -> Dict[str, Dict]:
        """Get statistics from all handlers"""
        stats = {}
        for handler_type, handler in self.handlers.items():
            if hasattr(handler, 'get_stats'):
                stats[handler_type] = handler.get_stats()
        return stats
