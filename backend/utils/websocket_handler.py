import asyncio
import websockets
import logging
import json
from pathlib import Path
import rerun as rr
from .handlers.message_router import MessageRouter
from .rerun_manager import RerunManager
from .video_stream import VideoStream

class WebSocketHandler:
    def __init__(self):
        self.clients = set()
        self.uploads_path = Path("tmp_content/uploads")
        self.logger = logging.getLogger(__name__)
        self.heartbeat_interval = 30
        self.rerun_manager = RerunManager()
        
        # Initialize message router and ensure uploads directory exists
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.message_router = MessageRouter(self.uploads_path)
        
    async def _keep_alive(self):
        """Send periodic heartbeats to keep connection alive"""
        while True:
            try:
                await asyncio.sleep(30)  # 30 second interval
                if self.clients:  # Only send if there are active clients
                    for client in self.clients:
                        await client.send(json.dumps({"type": "ping"}))
            except Exception as e:
                self.logger.error(f"Error in keep-alive: {e}")
                await asyncio.sleep(5)  # Brief pause before retry


    async def _setup_connection(self, websocket):
        """Setup initial WebSocket connection"""
        self.clients.add(websocket)
        websocket.max_size = 1024 * 1024 * 100  # 100MB limit
        
        # Initialize Rerun only if not already initialized
        from .rerun_manager import RerunManager
        rerun_manager = RerunManager()
        if not hasattr(rerun_manager, '_server_started'):
            rerun_manager.initialize()
        rerun_manager.register_connection()
        
        heartbeat_task = asyncio.create_task(self.send_heartbeat(websocket))
        return heartbeat_task

    async def _cleanup_connection(self, websocket):
        """Cleanup when connection closes"""
        if websocket in self.clients:
            self.clients.remove(websocket)
            from .rerun_manager import RerunManager
            rerun_manager = RerunManager()
            rerun_manager.unregister_connection()

    async def get_uploaded_files(self):
        """Get list of uploaded files with their metadata"""
        try:
            if not self.uploads_path.exists():
                self.logger.warning(f"Creating uploads directory: {self.uploads_path}")
                self.uploads_path.mkdir(parents=True, exist_ok=True)
                return []
                
            files = []
            for file_path in self.uploads_path.glob('*.mp4'):
                try:
                    stat = file_path.stat()
                    files.append({
                        'name': file_path.name,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'path': str(file_path)
                    })
                except OSError as e:
                    self.logger.error(f"Error accessing file {file_path}: {e}")
                    continue
                    
            self.logger.info(f"Found {len(files)} uploaded files")
            return sorted(files, key=lambda x: x['modified'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing uploaded files: {e}")
            return []

    async def handle_connection(self, websocket):
        """Handle incoming WebSocket connections"""
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.clients.add(websocket)
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.send_heartbeat(websocket))
        
        # Send initial list of uploaded files
        try:
            files = await self.get_uploaded_files()
            await websocket.send(json.dumps({
                'type': 'uploaded_files',
                'files': files
            }))
            self.logger.info(f"Sent initial file list: {len(files)} files")
        except Exception as e:
            self.logger.error(f"Error sending initial file list: {e}")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    if message == "pong":
                        continue
                        
                    try:
                        data = json.loads(message)
                        message_type = data.get('type')
                        self.logger.info(f"Received WebSocket message type: {message_type}")
                        
                        # Handle video control messages directly
                        if message_type in ['stop_video_stream', 'pause_video_stream', 'resume_video_stream']:
                            await self.message_router.route_message(websocket, data)
                        # Route other messages through the message router
                        elif message_type == 'video_upload_start':
                            # Ensure keep-alive task is running
                            if not hasattr(self.rerun_manager, '_keep_alive_task') or \
                               (hasattr(self.rerun_manager._keep_alive_task, 'done') and 
                                self.rerun_manager._keep_alive_task.done()):
                                self.rerun_manager._keep_alive_task = asyncio.create_task(
                                    self.rerun_manager._keep_alive())
                            
                            await websocket.send(json.dumps({
                                'type': 'upload_start_ack'
                            }))
                            
                        elif message_type == 'video_upload_complete':
                            await self.message_router.route_message(websocket, message)
                    except json.JSONDecodeError:
                        self.logger.error("Failed to decode JSON message")
                elif isinstance(message, bytes):
                        await self.message_router.route_message(websocket, message)
                        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            # Unregister connection from RerunManager
            from .rerun_manager import RerunManager
            rerun_manager = RerunManager()
            rerun_manager.unregister_connection()

    async def send_heartbeat(self, websocket):
        """Send periodic heartbeat to keep connection alive"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await websocket.send(json.dumps({"type": "ping"}))
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logging.error(f"Error sending heartbeat: {e}")
                break

    async def _init_rerun(self):
        """Initialize Rerun visualization"""
        try:
            self.rerun_manager.initialize()
            self.logger.info("Rerun initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Rerun: {e}")
            raise

    async def start_server(self, host='localhost', port=8001):
        """Start the WebSocket server"""
        try:
            # Initialize Rerun when starting the server
            await self._init_rerun()
            
            async with websockets.serve(
                ws_handler=self.handle_connection, 
                host=host, 
                port=port,
                max_size=1024 * 1024 * 100,  # 100MB max message size
                ping_interval=20,
                ping_timeout=60
            ) as server:
                self.logger.info(f"WebSocket server started on ws://{host}:{port}/ws")
                await asyncio.Future()  # run forever
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
    async def _handle_text_message(self, websocket, message):
        """Handle text-based WebSocket messages"""
        if message == "pong":
            return
            
        data = json.loads(message)
        message_type = data.get('type')
        self.logger.info(f"Received message type: {message_type}")
        
        handlers = {
            'video_upload_start': lambda: self.upload_handler.handle_upload_start(websocket, data),
            'video_upload_complete': self._handle_upload_complete,
            'reset_rerun': lambda: self._handle_rerun_reset(websocket)
        }
        
        handler = handlers.get(message_type)
        if handler:
            await handler()

    async def _handle_binary_message(self, websocket, message):
        """Handle binary WebSocket messages"""
        if hasattr(self.upload_handler, 'current_upload'):
            await self.upload_handler.handle_upload_chunk(message)
        else:
            await self.frame_handler.handle_frame(message)

    async def _handle_upload_complete(self, websocket):
        """Handle video upload completion"""
        file_path = await self.upload_handler.handle_upload_complete(websocket)
        if file_path:
            await self.handle_new_video(file_path)

    async def _handle_rerun_reset(self, websocket):
        """Handle Rerun reset request"""
        try:
            # Only clear topics if Rerun is already initialized
            if hasattr(rr, '_recording'):
                rr.log("world", rr.Clear(recursive=True))
                self.logger.info("Cleared Rerun topics on reset request")
            else:
                # Initialize only if not already initialized
                await self._init_rerun()
                
            # Ensure keep-alive task is running
            if not hasattr(self, '_keep_alive_task') or \
               (self._keep_alive_task and self._keep_alive_task.done()):
                self._keep_alive_task = asyncio.create_task(self._keep_alive())
            
            await websocket.send(json.dumps({
                'type': 'rerun_reset_complete'
            }))
        except Exception as e:
            self.logger.error(f"Error resetting Rerun: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'error': f"Failed to reset Rerun: {str(e)}"
            }))

            
    async def handle_new_video(self, file_path):
        """Handle setup for a newly uploaded video"""
        try:
            # Stop and clean up any existing video stream
            if self.video_streamer:
                self.video_streamer.stop()
                self.video_streamer = None
            
            # Clean up old video files
            uploads_dir = Path("tmp_content/uploads")
            if uploads_dir.exists():
                for old_file in uploads_dir.glob("*.mp4"):
                    if old_file != file_path:
                        try:
                            old_file.unlink()
                            self.logger.info(f"Removed old video: {old_file}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove old video {old_file}: {e}")
                
            # Create new video stream
            self.video_streamer = VideoStream(str(file_path))
            self.video_streamer.start()
            self.logger.info(f"Started streaming video: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to start video streaming: {e}")
            raise
