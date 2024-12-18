from flask_socketio import SocketIO, emit
from flask import request
import logging
import json
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from .video_streaming.stream_manager import StreamManager
from .video_streaming.websocket_publisher import WebSocketPublisher
from .video_streaming.stream_publisher import Frame
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from .video_streaming.stream_manager import StreamManager
from .video_streaming.websocket_publisher import WebSocketPublisher
from .video_streaming.stream_publisher import Frame
from pathlib import Path
import time
import cv2
import numpy as np
from typing import Optional, Dict
from .video_streaming.stream_manager import StreamManager
from .video_streaming.websocket_publisher import WebSocketPublisher
from .video_streaming.stream_publisher import Frame
from backend.services import SceneAnalysisService

class SocketHandler:
    """Handles Socket.IO events and manages stream connections"""
    
    def __init__(self, app):
        # Initialize main Socket.IO for control messages
        self.socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode='eventlet',
            logger=True,
            engineio_logger=True,
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=1024 * 1024,  # Smaller buffer for control messages
            path='/socket.io/',
            always_connect=True,
            transports=['websocket']
        )
        
        # Initialize streaming Socket.IO on separate port
        from flask import Flask
        stream_app = Flask('stream_app')
        self.stream_socketio = SocketIO(
            stream_app,
            cors_allowed_origins="*", 
            async_mode='eventlet',
            logger=True,
            engineio_logger=True,
            ping_timeout=120,  # Longer timeout for streaming
            ping_interval=30,
            max_http_buffer_size=100 * 1024 * 1024,  # 100MB buffer for video frames
            path='/socket.io/',
            always_connect=True,
            transports=['websocket'],
            manage_session=False,  # Disable session management for better performance
            async_handlers=True,   # Enable async handlers
            max_queue_size=50      # Limit queue size to prevent memory issues
        )
        self.logger = logging.getLogger(__name__)
        self.clients = {}  # Store client info with last ping time
        
        # Initialize stream manager and publisher with both socket instances
        self.stream_manager = StreamManager()
        self.publisher = WebSocketPublisher(self.socketio, self.stream_socketio)
        self.stream_manager.register_publisher(self.publisher)
        
        self.setup_handlers()

    def _cleanup_client_stream(self, client_id: str):
        """Clean up client's streaming resources"""
        try:
            if client_id in self.clients:
                self.clients[client_id]['streaming'] = False
                self.clients[client_id]['paused'] = False
            self.publisher.remove_client(client_id)
            self.stream_manager.remove_client(client_id)
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up client stream: {e}")
            return False
        
    def _cleanup_client(self, client_id: str):
        """Clean up client resources"""
        if client_id in self.clients:
            # Stop any ongoing analysis
            if self.clients[client_id].get('analyzing'):
                self.logger.info(f"Cleaning up analysis for client: {client_id}")
                self.clients[client_id]['analyzing'] = False

            # Clean up streaming
            if self.clients[client_id].get('streaming'):
                self.logger.info(f"Cleaning up stream for client: {client_id}")
                self.stream_manager.remove_client(client_id)

            # Remove client
            del self.clients[client_id]

    def setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.logger.info(f"Client connected: {client_id}")
            self.clients[client_id] = {
                'connected_at': time.time(),
                'last_ping': time.time(),
                'streaming': False,
                'analyzing': False,
                'edge_detection': False
            }
            emit('connection_established', {'status': 'success'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.logger.info(f"Client disconnected: {client_id}")
            self._cleanup_client(client_id)

        @self.socketio.on('request_scene_analysis')
        def handle_scene_analysis_request(data=None):
            """Handle request for scene analysis"""
            try:
                client_id = request.sid
                if not client_id in self.clients:
                    emit('error', {'message': 'Client not registered'})
                    return

                # Check if client is streaming
                if not self.clients[client_id].get('streaming'):
                    emit('error', {'message': 'No active stream for analysis'})
                    return

                # Check if already analyzing
                if self.clients[client_id].get('analyzing'):
                    emit('error', {'message': 'Analysis already in progress'})
                    return

                # Mark client as analyzing
                self.clients[client_id]['analyzing'] = True
                emit('analysis_started', {'timestamp': time.time()})

                try:
                    # Get frames from stream manager
                    # Get frames based on source type
                    source_type = data.get('source_type') if data else None
                    if source_type == 'stream':
                        # For streaming, capture consecutive frames
                        frames = self.stream_manager.capture_frames_for_analysis(
                            num_frames=8,
                            consecutive=True
                        )
                    else:
                        # For video files, use interval sampling
                        frames = self.stream_manager.capture_frames_for_analysis(
                            num_frames=8,
                            interval_frames=4
                        )

                    if not frames:
                        raise ValueError('Failed to capture frames for analysis')

                    # Extract frame data
                    frame_data = []
                    frame_numbers = []
                    timestamps = []

                    for frame, timestamp, frame_number in frames:
                        frame_data.append(frame)
                        timestamps.append(timestamp)
                        frame_numbers.append(frame_number)

                    # Perform analysis
                    scene_service = SceneAnalysisService()
                    analysis = scene_service.analyze_scene(
                        frame_data,
                        context=f"Analyzing stream for client {client_id}",
                        frame_numbers=frame_numbers,
                        timestamps=timestamps
                    )

                    # Save results
                    from backend.content_manager import ContentManager
                    content_manager = ContentManager()
                    analysis_id = f"scene_analysis_{int(time.time())}"
                    saved_path = content_manager.save_analysis(analysis, analysis_id)

                    # Prepare response
                    response_data = {
                        'analysis_id': analysis_id,
                        'scene_description': analysis['scene_analysis']['description'],
                        'timestamp': time.time(),
                        'storage_path': str(saved_path)
                    }

                    # Emit results
                    emit('analysis_complete', response_data)

                    # Add system message to chat
                    emit('chat_message', {
                        'role': 'system',
                        'content': f"Scene Analysis:\n{analysis['scene_analysis']['description']}"
                    })

                except Exception as e:
                    self.logger.error(f"Analysis error: {e}")
                    emit('error', {'message': f"Analysis failed: {str(e)}"})

                finally:
                    # Mark client as no longer analyzing
                    self.clients[client_id]['analyzing'] = False

            except Exception as e:
                self.logger.error(f"Scene analysis request error: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('analysis_status')
        def handle_analysis_status_request(data=None):
            """Handle request for analysis status"""
            try:
                client_id = request.sid
                if not client_id in self.clients:
                    return

                # Get recent analyses
                try:
                    analysis_path = Path("tmp_content/analysis")
                    if not analysis_path.exists():
                        return

                    recent_analyses = []
                    for file_path in sorted(
                        analysis_path.glob("scene_analysis_*.json"),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )[:5]:
                        try:
                            with open(file_path) as f:
                                analysis_data = json.load(f)
                            recent_analyses.append({
                                'id': file_path.stem,
                                'timestamp': file_path.stat().st_mtime,
                                'description': analysis_data.get('scene_analysis', {}).get('description', '')
                            })
                        except Exception as e:
                            self.logger.error(f"Error reading analysis file {file_path}: {e}")

                    emit('analysis_status', {
                        'recent_analyses': recent_analyses,
                        'analyzing': self.clients[client_id].get('analyzing', False),
                        'timestamp': time.time()
                    })

                except Exception as e:
                    self.logger.error(f"Error getting analysis status: {e}")
                    emit('error', {'message': f"Failed to get analysis status: {str(e)}"})

            except Exception as e:
                self.logger.error(f"Analysis status request error: {e}")

        @self.socketio.on_error()
        def error_handler(e):
            """Handle WebSocket errors"""
            self.logger.error(f"WebSocket error: {e}")
            emit('error', {'message': 'Internal server error'})

        @self.socketio.on('client_info')
        def handle_client_info(data):
            """Handle client information"""
            client_id = request.sid
            self.logger.info(f"Client info received from {client_id}: {data}")
            if client_id in self.clients:
                self.clients[client_id].update({
                    'info': data,
                    'last_ping': time.time()
                })

        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping from client"""
            client_id = request.sid
            if client_id in self.clients:
                self.clients[client_id]['last_ping'] = time.time()
            emit('pong')

        @self.socketio.on('start_edge_detection')
        def handle_start_edge_detection():
            """Handle edge detection start request"""
            client_id = request.sid
            if client_id in self.clients:
                self.clients[client_id]['edge_detection'] = True
                emit('edge_detection_started')

        @self.socketio.on('stop_edge_detection')
        def handle_stop_edge_detection():
            """Handle edge detection stop request"""
            client_id = request.sid
            if client_id in self.clients:
                self.clients[client_id]['edge_detection'] = False
                emit('edge_detection_stopped')

        @self.socketio.on('frame')
        def handle_frame(frame_data):
            """Handle incoming frame data"""
            try:
                client_id = request.sid
                if not client_id in self.clients:
                    return

                if not self.clients[client_id].get('streaming'):
                    if self.stream_manager.start_streaming():
                        self.clients[client_id]['streaming'] = True
                    else:
                        return

                # Update client's last activity
                self.clients[client_id]['last_ping'] = time.time()

                # Handle different frame data formats with validation
                if isinstance(frame_data, (bytes, bytearray)):
                    frame_bytes = frame_data
                elif isinstance(frame_data, str):
                    import base64
                    frame_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
                elif isinstance(frame_data, dict) and '_placeholder' in frame_data:
                    # Handle placeholder frame data
                    self.logger.warning("Received placeholder frame data, skipping")
                    return
                else:
                    raise ValueError(f"Unsupported frame data type: {type(frame_data)}")

                if not frame_bytes:
                    raise ValueError("Empty frame data")

                # Decode frame
                nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
                if frame is None or frame.size == 0:
                    raise ValueError("Failed to decode frame or empty frame")
                
                # Validate frame dimensions
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    raise ValueError("Invalid frame dimensions")

                # Create frame object with metadata
                frame_obj = Frame(
                    data=frame,
                    timestamp=time.time(),
                    frame_number=self.stream_manager.frame_count,
                    metadata={
                        'source': 'camera',
                        'client_id': client_id,
                        'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                        'channels': frame.shape[2] if len(frame.shape) > 2 else 1,
                        'frame_type': 'camera'
                    }
                )
                
                # Apply edge detection if enabled
                if self.clients[client_id].get('edge_detection', False):
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Apply Gaussian blur
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    # Apply Canny edge detection
                    edges = cv2.Canny(blurred, 100, 200)
                    # Convert back to BGR for visualization
                    frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                # Create frame object with processed frame
                try:
                    frame_obj = Frame(
                        data=frame,
                        timestamp=time.time(),
                        frame_number=self.stream_manager.frame_count,
                        metadata={
                            'source': 'camera',
                            'client_id': client_id,
                            'edge_detection': self.clients[client_id].get('edge_detection', False),
                            'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                            'channels': frame.shape[2] if len(frame.shape) > 2 else 1
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Error creating Frame object: {e}")
                    raise

                # Publish frame
                if self.stream_manager.publish_frame(frame_obj):
                    self.logger.debug(f"Published frame from {client_id}")
                else:
                    self.logger.warning(f"Failed to publish frame from {client_id}")
                
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}", exc_info=True)
                emit('error', {'message': str(e)})

        @self.socketio.on('frame_metadata')
        def handle_frame_metadata(metadata):
            """Handle frame metadata"""
            try:
                client_id = request.sid
                if not client_id in self.clients:
                    return

                if not self.clients[client_id].get('streaming'):
                    return

                self.logger.debug(f"Received frame metadata from {client_id}: {metadata}")
                # Store metadata for next frame
                metadata['client_id'] = client_id
                self.stream_manager.set_frame_metadata(metadata)
            except Exception as e:
                self.logger.error(f"Error processing frame metadata: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('start_stream')
        def handle_start_stream(data=None):
            """Handle stream start request"""
            try:
                client_id = request.sid
                if client_id not in self.clients:
                    raise ValueError("Client not registered")

                if not self.clients[client_id].get('streaming'):
                    if self.stream_manager.start_streaming():
                        self.clients[client_id]['streaming'] = True
                        self.clients[client_id]['paused'] = False
                        self.stream_manager.add_client(client_id)
                        emit('stream_started', {'status': 'success'})
                        self.logger.info(f"Stream started for client: {client_id}")
                    else:
                        emit('error', {'message': 'Failed to start stream'})
            except Exception as e:
                self.logger.error(f"Failed to start stream: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('stop_stream')
        def handle_stop_stream(data=None):
            """Handle stream stop request"""
            try:
                client_id = request.sid
                if client_id not in self.clients:
                    emit('error', {'message': 'Client not registered'})
                    return

                # Clean up client's stream resources
                if self._cleanup_client_stream(client_id):
                    emit('stream_stopped', {'status': 'success'})
                    self.logger.info(f"Stream stopped for client: {client_id}")
                else:
                    emit('error', {'message': 'Failed to stop stream'})
            except Exception as e:
                self.logger.error(f"Failed to stop stream: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('pause_stream')
        def handle_pause_stream(data=None):
            """Handle stream pause request"""
            try:
                client_id = request.sid
                if client_id in self.clients and self.clients[client_id]['streaming']:
                    if self.stream_manager.pause_streaming():
                        self.clients[client_id]['paused'] = True
                        emit('stream_paused', {'status': 'success'})
                        self.logger.info(f"Stream paused for client: {client_id}")
                    else:
                        emit('error', {'message': 'Failed to pause stream'})
            except Exception as e:
                self.logger.error(f"Failed to pause stream: {e}")
                emit('error', {'message': str(e)})

        @self.socketio.on('resume_stream')
        def handle_resume_stream(data=None):
            """Handle stream resume request"""
            try:
                client_id = request.sid
                if client_id in self.clients and self.clients[client_id]['streaming']:
                    if self.stream_manager.resume_streaming():
                        self.clients[client_id]['paused'] = False
                        emit('stream_resumed', {'status': 'success'})
                        self.logger.info(f"Stream resumed for client: {client_id}")
                    else:
                        emit('error', {'message': 'Failed to resume stream'})
            except Exception as e:
                self.logger.error(f"Failed to resume stream: {e}")
                emit('error', {'message': str(e)})
