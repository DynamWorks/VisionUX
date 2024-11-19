import asyncio
import websockets
import json
import logging
import rerun as rr
from pathlib import Path
from .video_upload_handler import VideoUploadHandler
from .camera_frame_handler import CameraFrameHandler
from utils.video_stream import VideoStream

class WebSocketHandler:
    def __init__(self):
        self.clients = set()
        self.uploads_path = Path("tmp_content/uploads")
        self.logger = logging.getLogger(__name__)
        
        # Initialize handlers and services
        self.upload_handler = VideoUploadHandler(self.uploads_path)
        self.frame_handler = CameraFrameHandler()
        self.video_streamer = None
        
        # Initialize Rerun
        self._init_rerun()
        
        # Heartbeat configuration
        self.heartbeat_interval = 30  # seconds
        
    def _init_rerun(self):
        """Initialize or reinitialize Rerun"""
        rr.init("video_analytics")
        rr.serve(
            open_browser=False,
            ws_port=4321,
            default_blueprint=rr.blueprint.Vertical(
                rr.blueprint.Spatial2DView(origin="world/video", name="Video Stream")
            ),
            blocking=False
        )

    async def handle_connection(self, websocket):
        """Handle incoming WebSocket connections"""
        self.clients.add(websocket)
        heartbeat_task = asyncio.create_task(self.send_heartbeat(websocket))
        
        # Increase WebSocket message size limit (100MB)
        websocket.max_size = 1024 * 1024 * 100
        
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
                            # Reinitialize Rerun
                            if hasattr(self, '_rerun_initialized'):
                                delattr(self, '_rerun_initialized')
                            rr.init("video_analytics")#, spawn=True)
                            rr.serve(
                                open_browser=False,
                                ws_port=4321,
                                default_blueprint=rr.blueprint.Vertical(
                                    rr.blueprint.Spatial2DView(origin="world/video", name="Video Stream")
                                )
                            )
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
                    # Next message will be the video file
                    try:
                        message = await websocket.recv()
                        if isinstance(message, str):
                            try:
                                data = json.loads(message)
                                if data.get('type') == 'video_upload_start':
                                    # Ensure uploads directory exists
                                    self.uploads_path.mkdir(parents=True, exist_ok=True)
                                
                                    # Get file metadata
                                    filename = data.get('filename', f"video_{len(self.clients)}_{int(time.time())}.mp4")
                                    file_path = self.uploads_path / filename
                                
                                    # Create uploads directory if it doesn't exist
                                    self.uploads_path.mkdir(parents=True, exist_ok=True)
                                    
                                    total_size = data.get('size', 0)
                                    bytes_received = 0
                                    chunk_timeout = 30  # seconds
                                    
                                    logging.info(f"Starting upload of {filename} ({total_size} bytes)")
                                    
                                    # Send acknowledgment
                                    await websocket.send(json.dumps({
                                        "type": "upload_start_ack",
                                        "filename": filename
                                    }))
                                    
                                    # Open file for writing chunks
                                    with open(file_path, 'wb') as f:
                                        while True:
                                            try:
                                                chunk_meta = await asyncio.wait_for(
                                                    websocket.recv(),
                                                    timeout=chunk_timeout
                                                )
                                                if not isinstance(chunk_meta, str):
                                                    logging.error("Invalid chunk metadata format")
                                                    continue
                                                    
                                                chunk_data = json.loads(chunk_meta)
                                                if chunk_data.get('type') == 'video_upload_complete':
                                                    logging.info(f"Upload complete: {filename}")
                                                    break
                                            
                                                if chunk_data.get('type') == 'video_upload_chunk':
                                                    chunk = await websocket.recv()
                                                    if isinstance(chunk, bytes):
                                                        offset = chunk_data.get('offset', 0)
                                                        size = chunk_data.get('size', 0)
                                                        
                                                        f.write(chunk)
                                                        bytes_received += len(chunk)
                                                        
                                                        progress = (bytes_received / total_size) * 100
                                                        logging.info(f"Progress: {progress:.1f}% ({bytes_received}/{total_size} bytes)")
                                                    else:
                                                        logging.error("Invalid chunk data format")
                                            except Exception as e:
                                                logging.error(f"Error during upload: {str(e)}")
                                                await websocket.send(json.dumps({
                                                    "type": "upload_error",
                                                    "error": str(e)
                                                }))
                                                break
                                
                                    file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
                                    logging.info(f"Video file saved to: {file_path}")
                                    logging.info(f"Video file size: {file_size:.2f} MB")
                                
                                    await websocket.send(json.dumps({
                                        "type": "upload_complete",
                                        "path": str(file_path),
                                        "success": True
                                    }))
                            except IOError as e:
                                logging.error(f"Failed to save video file: {e}")
                                await websocket.send(json.dumps({
                                    "type": "upload_complete",
                                    "success": False,
                                    "error": "Failed to save video file"
                                }))
                    except Exception as e:
                        logging.error(f"Error handling video upload: {e}")
                        await websocket.send(json.dumps({
                            "type": "upload_complete",
                            "success": False,
                            "error": "Upload failed"
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
                        else:
                            logging.error("Failed to decode camera frame")
                            continue
                            
                        # Convert BGR to RGB for Rerun
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process frame with edge detection
                        #edges_rgb = self.edge_detector.detect_edges(frame)
                        
                        # Log original frame to Rerun
                        timestamp = time.time()
                        rr.log("camera/original", 
                              rr.Image(frame_rgb),
                              timeless=False,
                              timestamp=timestamp)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)

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
        async with websockets.serve(self.handle_connection, host, port):
            await asyncio.Future()  # run forever
    async def handle_message(self, websocket, message):
        """Route incoming messages to appropriate handlers"""
        try:
            if isinstance(message, str):
                if message == "pong":
                    return
                    
                data = json.loads(message)
                message_type = data.get('type')
                self.logger.info(f"Received message type: {message_type}")
                
                if message_type == 'video_upload_start':
                    await self.upload_handler.handle_upload_start(websocket, data)
                    
                elif message_type == 'video_upload_complete':
                    file_path = await self.upload_handler.handle_upload_complete(websocket)
                    if file_path:
                        await self.handle_new_video(file_path)
                        
                elif message_type == 'reset_rerun':
                    self._init_rerun()
                    await websocket.send(json.dumps({
                        'type': 'rerun_reset_complete'
                    }))
                    
            else:  # Binary message
                if hasattr(self.upload_handler, 'current_upload'):
                    await self.upload_handler.handle_upload_chunk(message)
                else:
                    await self.frame_handler.handle_frame(message)
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'error': str(e)
            }))
            
    async def handle_new_video(self, file_path):
        """Handle setup for a newly uploaded video"""
        try:
            # Stop any existing video stream
            if self.video_streamer:
                self.video_streamer.stop()
                
            # Create new video stream
            self.video_streamer = VideoStream(str(file_path))
            self.video_streamer.start()
            self.logger.info(f"Started streaming video: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to start video streaming: {e}")
            raise
