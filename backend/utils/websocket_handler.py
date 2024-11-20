import asyncio
import websockets
import json
import logging
import rerun as rr
import time
import numpy as np
import cv2
from pathlib import Path
from .video_upload_handler import VideoUploadHandler
from .camera_frame_handler import CameraFrameHandler
from utils.video_stream import VideoStream

class WebSocketHandler:
    def __init__(self):
        self.clients = set()
        self.uploads_path = Path("tmp_content/uploads")
        self.logger = logging.getLogger(__name__)
        
        # Constants
        self.chunk_timeout = 30  # seconds
        
        # Initialize handlers and services
        self.upload_handler = VideoUploadHandler(self.uploads_path)
        self.frame_handler = CameraFrameHandler()
        self.video_streamer = None
        
        # Heartbeat configuration
        self.heartbeat_interval = 30  # seconds
        
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

    async def _init_rerun(self):
        """Initialize Rerun for websocket handling"""
        try:
            # Initialize through RerunManager
            from .rerun_manager import RerunManager
            rerun_manager = RerunManager()
            rerun_manager.initialize()
            
            # Start keep-alive task only once
            if not hasattr(self, '_keep_alive_task'):
                self._keep_alive_task = asyncio.create_task(self._keep_alive())
                self.logger.info("Started Rerun keep-alive task")
            elif self._keep_alive_task.done():
                self._keep_alive_task = asyncio.create_task(self._keep_alive())
                self.logger.info("Restarted Rerun keep-alive task")
        except Exception as e:
            self.logger.error(f"Error initializing Rerun: {e}")
            raise

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
        # Ensure uploads directory exists
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        
        heartbeat_task = await self._setup_connection(websocket)
        
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
                if isinstance(message, str) and message == "pong":
                    continue
                # Try to parse as JSON first
                try:
                    if isinstance(message, str):
                        data = json.loads(message)
                        message_type = data.get('type')
                        logging.info(f"Received message type: {message_type}")
                        
                        if message_type == 'video_upload_start':
                            self.current_upload = {
                                'filename': data.get('filename'),
                                'size': data.get('size'),
                                'file_handle': open(self.uploads_path / data.get('filename'), 'wb'),
                                'bytes_received': 0
                            }
                            logging.info(f"Starting video upload: {data.get('filename')} ({data.get('size')} bytes)")
                            await websocket.send(json.dumps({'type': 'upload_start_ack'}))
                            
                        elif message_type == 'video_upload_chunk':
                            if not hasattr(self, 'current_upload'):
                                raise ValueError("No active upload session")
                            logging.info(f"Received chunk: offset={data.get('offset')}, size={data.get('size')}, progress={data.get('progress')}%")
                            
                        elif message_type == 'reset_rerun':
                            # Clear all topics
                            rr.log("world", rr.Clear(recursive=True))
                            rr.log("camera", rr.Clear(recursive=True))
                            rr.log("edge_detection", rr.Clear(recursive=True))
                            rr.log("heartbeat", rr.Clear(recursive=True))
                            self.logger.info("Cleared all Rerun topics on frontend refresh")
                            # Ensure Rerun stays alive
                            from .rerun_manager import RerunManager
                            rerun_manager = RerunManager()
                            if not hasattr(rerun_manager, '_keep_alive_task') or rerun_manager._keep_alive_task.done():
                                rerun_manager._keep_alive_task = asyncio.create_task(rerun_manager._keep_alive())
                            await websocket.send(json.dumps({
                                'type': 'rerun_reset_complete'
                            }))
                            
                        elif message_type == 'video_upload_complete':
                            if hasattr(self, 'current_upload'):
                                try:
                                    self.current_upload['file_handle'].flush()
                                    self.current_upload['file_handle'].close()
                                    file_path = self.uploads_path / self.current_upload['filename']
                                    if file_path.exists():
                                        file_size = file_path.stat().st_size
                                        logging.info(f"Upload completed successfully: {self.current_upload['filename']} ({file_size} bytes)")
                                        await websocket.send(json.dumps({
                                            'type': 'upload_complete_ack',
                                            'filename': self.current_upload['filename'],
                                            'size': file_size
                                        }))
                                        
                                        # Start streaming the uploaded video
                                        try:
                                            # Stop any existing stream
                                            if self.video_streamer:
                                                self.video_streamer.stop()
                                            # Create new video stream
                                            self.video_streamer = VideoStream(str(file_path))
                                            self.video_streamer.start()
                                            logging.info(f"Started streaming video: {file_path}")
                                        except Exception as e:
                                            logging.error(f"Failed to start video streaming: {e}")
                                            
                                        # Keep connection alive for a moment to ensure client receives the acknowledgment
                                        await asyncio.sleep(1)
                                    else:
                                        raise FileNotFoundError(f"Uploaded file not found: {file_path}")
                                except Exception as e:
                                    logging.error(f"Error finalizing upload: {str(e)}")
                                    await websocket.send(json.dumps({
                                        'type': 'upload_error',
                                        'error': f"Failed to finalize upload: {str(e)}"
                                    }))
                                    await asyncio.sleep(1)  # Give time for error message to be sent
                                finally:
                                    delattr(self, 'current_upload')
                            
                    else:
                        # Handle binary chunk data
                        if hasattr(self, 'current_upload'):
                            self.current_upload['file_handle'].write(message)
                            self.current_upload['bytes_received'] += len(message)
                            logging.info(f"Wrote {len(message)} bytes to {self.current_upload['filename']}")
                except Exception as e:
                    logging.error(f"Error processing message: {str(e)}")
                    if hasattr(self, 'current_upload'):
                        self.current_upload['file_handle'].close()
                        delattr(self, 'current_upload')
                    await websocket.send(json.dumps({
                        'type': 'upload_error',
                        'error': str(e)
                    }))

                if message_type == 'video_upload':
                    try:
                        # Delegate upload handling to upload handler
                        file_path = await self.upload_handler.handle_upload(websocket)
                        if file_path:
                            # Handle successful upload
                            await self.handle_new_video(file_path)
                            await websocket.send(json.dumps({
                                "type": "upload_complete",
                                "path": str(file_path),
                                "success": True
                            }))
                            # Refresh file list
                            files = await self.get_uploaded_files()
                            await websocket.send(json.dumps({
                                'type': 'uploaded_files',
                                'files': files
                            }))
                    except Exception as e:
                        self.logger.error(f"Error handling video upload: {e}")
                        await websocket.send(json.dumps({
                            "type": "upload_error",
                            "error": str(e)
                        }))
                
                elif message_type == 'start_video_stream':
                    filename = data.get('filename')
                    file_path = self.uploads_path / filename
                    if file_path.exists():
                        # Stop any existing stream
                        if self.video_streamer:
                            self.video_streamer.stop()
                        # Create new video stream
                        self.video_streamer = VideoStream(str(file_path))
                        self.video_streamer.start()
                        self.logger.info(f"Started streaming video: {file_path}")
                    else:
                        self.logger.error(f"Video file not found: {file_path}")
                        
                elif message_type == 'pause_video_stream':
                    if self.video_streamer:
                        self.video_streamer.pause()
                        self.logger.info("Video stream paused")
                        
                elif message_type == 'stop_video_stream':
                    if self.video_streamer:
                        self.video_streamer.stop()
                        self.video_streamer = None
                        self.logger.info("Video stream stopped")
                        
                elif message_type == 'start_camera_stream':
                    device_id = data.get('deviceId')
                    if self.video_streamer:
                        self.video_streamer.stop()
                        self.video_streamer = None
                        self.logger.info("Video stream stopped")
                    logging.info(f"Starting camera stream from device: {device_id}")
                    # Reset Rerun for new stream
                    # rr.log("world", rr.Clear(recursive=True))
                    # rr.log("camera", rr.Clear(recursive=True))
                    # rr.log("edge_detection", rr.Clear(recursive=True))

                    await websocket.send(json.dumps({
                        'type': 'camera_stream_started'
                    }))
                    
                elif message_type == 'camera_frame':
                    # Next message will be the frame data
                    message = await websocket.recv()
                    if isinstance(message, bytes):
                        # Handle live stream frame
                        nparr = np.frombuffer(message, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            logging.info(f"Received camera frame: {frame.shape}")
                            # Convert BGR to RGB for Rerun
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Log frame to Rerun using same topic as video stream
                            timestamp = time.time()
                            rr.log("world/video", 
                                  rr.Image(frame_rgb),
                                  timeless=False,
                                  timestamp=timestamp)
                        else:
                            logging.error("Failed to decode camera frame")
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

    async def start_server(self, host='localhost', port=8001):
        # Initialize Rerun when starting the server
        await self._init_rerun()
        
        async with websockets.serve(
            ws_handler=self.handle_connection, 
            host=host, 
            port=port,
            process_request=lambda path, headers: None if path == "/ws" else (404, [], b'Not Found')
        ):
            self.logger.info(f"WebSocket server started on ws://{host}:{port}/ws")
            await asyncio.Future()  # run forever
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
            # Initialize Rerun if needed
            if not hasattr(rr, '_recording'):
                await self._init_rerun()
            
            # Clear all topics
            rr.log("world", rr.Clear(recursive=True))
            rr.log("camera", rr.Clear(recursive=True))
            rr.log("edge_detection", rr.Clear(recursive=True))
            rr.log("heartbeat", rr.Clear(recursive=True))
            self.logger.info("Cleared all Rerun topics on frontend refresh")
            
            # Ensure keep-alive task is running
            if not hasattr(self, '_keep_alive_task'):
                self._keep_alive_task = asyncio.create_task(self._keep_alive())
            elif self._keep_alive_task and self._keep_alive_task.done():
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

    async def handle_message(self, websocket, message):
        """Route incoming messages to appropriate handlers"""
        try:
            if isinstance(message, str):
                await self._handle_text_message(websocket, message)
            else:  # Binary message
                await self._handle_binary_message(websocket, message)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'error': str(e)
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
