from flask_socketio import SocketIO, emit
import logging
import json
from pathlib import Path
import time

class SocketHandler:
    """Handles WebSocket events"""
    
    def __init__(self, app):
        self.socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            max_http_buffer_size=50 * 1024 * 1024,  # 50MB
            async_mode='gevent',
            logger=True,
            ping_timeout=60,
            ping_interval=25,
            transports=['websocket', 'polling'],  # Allow fallback to polling
            always_connect=True,
            engineio_logger=True
        )
        self.uploads_path = Path("tmp_content/uploads")
        self.logger = logging.getLogger(__name__)
        self.clients = set()
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.setup_handlers()

    def setup_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected")
            self.clients.add(self.socketio)
            self._send_file_list()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected")
            if self.socketio in self.clients:
                self.clients.remove(self.socketio)

        @self.socketio.on('video_upload_start')
        def handle_upload_start(data):
            try:
                filename = data.get('filename')
                size = data.get('size')
                file_path = self.uploads_path / filename

                if file_path.exists():
                    emit('error', {'message': 'File already exists'})
                    return

                max_size = 500 * 1024 * 1024  # 500MB
                if size > max_size:
                    emit('error', {'message': f'File too large (max {max_size/1024/1024}MB)'})
                    return

                emit('upload_start_ack')
            except Exception as e:
                self.logger.error(f"Upload start error: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('video_chunk')
        def handle_chunk(data):
            try:
                # Process video chunk
                emit('upload_progress', {
                    'progress': data.get('progress', 0),
                    'chunk': data.get('chunk'),
                    'totalChunks': data.get('totalChunks')
                })
            except Exception as e:
                self.logger.error(f"Chunk processing error: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('video_upload_complete')
        def handle_upload_complete(data):
            try:
                filename = data.get('filename')
                self.logger.info(f"Upload complete: {filename}")
                self._send_file_list()
                emit('upload_complete_ack', {'filename': filename})
            except Exception as e:
                self.logger.error(f"Upload completion error: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('camera_frame')
        def handle_camera_frame(data):
            try:
                frame_timestamp = data.get('timestamp')
                self.logger.info(f"Received camera frame metadata at {frame_timestamp}")
                
                # Log incoming data structure
                self.logger.debug(f"Frame metadata: {data}")
                
                # Set timeout for binary data receipt
                self.logger.debug("Waiting for binary frame data...")
                try:
                    binary_data = self.socketio.receive(binary=True, timeout=5.0)
                    self.logger.info(f"Received binary frame data of size: {len(binary_data)} bytes")
                    
                    if not binary_data:
                        self.logger.error("No binary frame data received within timeout")
                        emit('frame_error', {
                            'timestamp': frame_timestamp,
                            'error': 'Frame data timeout'
                        })
                        return
                except Exception as e:
                    self.logger.error(f"Error receiving binary data: {e}")
                    emit('frame_error', {
                        'timestamp': frame_timestamp,
                        'error': f'Binary data receive error: {str(e)}'
                    })
                    return
                
                self.logger.info(f"Successfully received binary frame data")
                
                frame_size = len(binary_data)
                self.logger.info(f"Processing binary frame data: {frame_size} bytes")
                
                # Validate frame size
                if frame_size < 1024:  # Minimum 1KB for valid frame
                    raise ValueError(f"Frame too small: {frame_size} bytes")
                
                # Process frame with Rerun
                self._process_frame(binary_data)
                
                # Send detailed acknowledgment
                process_time = time.time() - frame_timestamp
                emit('frame_processed', {
                    'timestamp': frame_timestamp,
                    'process_time_ms': round(process_time * 1000, 2),
                    'frame_size': frame_size,
                    'status': 'success'
                })
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}", exc_info=True)
                emit('error', {'message': str(e)})


    def _send_file_list(self):
        """Send list of uploaded files to client"""
        try:
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

            emit('uploaded_files', {'files': sorted(files, key=lambda x: x['modified'], reverse=True)})
        except Exception as e:
            self.logger.error(f"Error sending file list: {e}")

    def _process_frame(self, frame_data):
        """Process frame with appropriate viewer"""
        try:
            import numpy as np
            import cv2
            import time
            
            process_start = time.time()
            frame_size = len(frame_data)
            self.logger.debug(f"Processing frame data of size: {frame_size} bytes")

            # Decode frame with validation
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode frame")
                
            # Validate frame properties
            if frame.size == 0 or len(frame.shape) != 3:
                raise ValueError(f"Invalid frame shape: {frame.shape}")
                
            if not np.isfinite(frame).all():
                raise ValueError("Frame contains invalid values")

            # Convert BGR to RGB for visualization
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get viewer type from config
            from .config import Config
            config = Config()
            viewer_type = config.get('api', 'viewer', default='rerun')

            if viewer_type == 'rerun':
                # Use RerunManager for visualization
                from .rerun_manager import RerunManager
                rerun_manager = RerunManager()
                
                # Track frame metrics
                frame_number = getattr(self, '_frame_count', 0)
                process_time = time.time() - process_start
                
                # Log frame with metrics
                rerun_manager.log_frame(
                    frame=frame_rgb,
                    frame_number=frame_number,
                    source="socket_stream",
                    metadata={
                        'process_time_ms': round(process_time * 1000, 2),
                        'frame_size': frame_size,
                        'shape': frame.shape
                    }
                )
            else:
                # Stream directly to frontend for custom viewer
                # Convert frame to JPEG for efficient streaming
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                frame_bytes = buffer.tobytes()
                
                # Send binary frame data directly
                self.socketio.emit('binary_frame', frame_bytes, binary=True)
            
            self._frame_count = frame_number + 1
            
            # Log performance metrics periodically
            if frame_number % 100 == 0:
                self.logger.info(
                    f"Frame processing metrics - "
                    f"Count: {frame_number}, "
                    f"Time: {process_time*1000:.1f}ms, "
                    f"Size: {frame_size/1024:.1f}KB"
                )

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            raise
