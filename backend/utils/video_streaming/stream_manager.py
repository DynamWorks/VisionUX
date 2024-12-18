import logging
import time
import cv2
import numpy as np
from typing import Optional, Dict, List, Generator, Tuple
from pathlib import Path
from collections import deque
from threading import Lock
from .stream_publisher import StreamPublisher, Frame

class StreamManager:
    """Enhanced video stream manager with frame buffer and analysis capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.publishers: List[StreamPublisher] = []
        self.subscribers = []
        self.is_streaming = False
        self.is_paused = False
        self.frame_count = 0
        self.current_file: Optional[str] = None
        self.frame_metadata: Optional[Dict] = None
        self.last_frame_time = time.time()
        self.fps = 0
        self.active_clients = set()
        self.video_capture = None
        
        # Frame buffer configuration
        self.frame_buffer = deque(maxlen=10)  # Reduce buffer size for analysis
        self.buffer_lock = Lock()

        # Separate video capture for analysis
        self.analysis_capture = None

        
        # Stream metrics
        self.metrics = {
            'fps': 0,
            'frame_count': 0,
            'resolution': '',
            'start_time': None,
            'last_frame_time': None,  # Explicitly None until first frame
            'dropped_frames': 0,
            'total_frames_received': 0,
            'last_error': None
        }

    def register_publisher(self, publisher: StreamPublisher):
        """Register a publisher for frame distribution"""
        self.publishers.append(publisher)
        self.logger.info(f"Registered publisher: {publisher.__class__.__name__}")

    def register_subscriber(self, subscriber):
        """Register a subscriber for frame processing"""
        self.subscribers.append(subscriber)
        self.logger.info(f"Registered subscriber: {subscriber.__class__.__name__}")

    def get_video_path(self) -> Optional[str]:
        """Get current video file path"""
        if self.current_file:
            return str(Path("tmp_content/uploads") / self.current_file)
        return None

    def capture_frames_for_analysis(self, num_frames: int = 8, consecutive: bool = False, interval_frames: int = 4) -> List[Tuple[np.ndarray, float, int]]:
        """
        Capture frames for analysis from either live stream or video file
        """
        try:
            captured_frames = []
            video_path = self.get_video_path()

            if not video_path:
                # If no video file but streaming is active, capture from buffer
                if self.is_streaming and self.frame_buffer:
                    with self.buffer_lock:
                        buffer_frames = list(self.frame_buffer)
                        step = max(1, len(buffer_frames) // num_frames)
                        for frame in buffer_frames[::step][:num_frames]:
                            frame_data = frame.data
                            if not isinstance(frame_data, np.ndarray):
                                frame_data = cv2.imdecode(
                                    np.frombuffer(frame_data, np.uint8), 
                                    cv2.IMREAD_COLOR
                                )
                            captured_frames.append((
                                frame_data.copy(),  # Make a copy to avoid reference issues
                                frame.timestamp,
                                frame.frame_number
                            ))
                    return captured_frames
                raise ValueError("No video file or active stream available")
                
            # Otherwise try to capture from video file
            video_path = self.get_video_path()
            if not video_path:
                raise ValueError("No video file or active stream available")
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
                
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < num_frames:
                    num_frames = total_frames
                    
                if consecutive:
                    # For streaming, capture consecutive frames from current position
                    frame_positions = list(range(
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        min(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + num_frames, total_frames)
                    ))
                else:
                    # For video files, use evenly spaced positions
                    interval = total_frames // num_frames
                    frame_positions = [i * interval for i in range(num_frames)]
                
                captured_frames = []
                for pos in frame_positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if ret:
                        timestamp = pos / cap.get(cv2.CAP_PROP_FPS)
                        captured_frames.append((
                            frame,
                            timestamp,
                            pos
                        ))
                        
                return captured_frames
                
            finally:
                cap.release()
                
        except Exception as e:
            self.logger.error(f"Error capturing frames: {e}")
            raise

    def publish_frame(self, frame: Frame, frame_type: str = 'frame') -> bool:
        """Publish frame to subscribers - independent of analysis"""
        if not self.is_streaming or self.is_paused:
            return False
            
        try:
            # Process frame through subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber.process_frame(frame)
                except Exception as e:
                    self.logger.error(f"Error in subscriber {subscriber.__class__.__name__}: {e}")

            # Ensure frame data is in correct format
            if isinstance(frame.data, np.ndarray):
                success, buffer = cv2.imencode('.jpg', frame.data)
                if not success:
                    raise ValueError("Failed to encode frame")
                frame.data = buffer.tobytes()

            # Publish to all publishers
            for publisher in self.publishers:
                try:
                    publisher.publish_frame(frame)
                except Exception as e:
                    self.logger.error(f"Error in publisher {publisher.__class__.__name__}: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Error publishing frame: {e}")
            return False
            
    def start_streaming(self, filename: Optional[str] = None) -> bool:
        """Start video streaming"""
        try:
            if filename:
                self.current_file = filename
            
            self.is_streaming = True
            self.is_paused = False
            
            # Start publishers
            for publisher in self.publishers:
                publisher.start_publishing()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting stream: {e}")
            return False
            
    def stop_streaming(self) -> bool:
        """Stop streaming"""
        try:
            self.is_streaming = False
            self.is_paused = False
            
            # Stop publishers
            for publisher in self.publishers:
                try:
                    publisher.stop_publishing()
                except Exception as e:
                    self.logger.error(f"Error stopping publisher: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping stream: {e}")
            return False

    def pause_streaming(self) -> bool:
        """Pause streaming"""
        if not self.is_streaming:
            return False

        if self.is_paused:
            return False

        self.is_paused = True
        self.logger.info("Streaming paused")
        return True

    def resume_streaming(self) -> bool:
        """Resume streaming"""
        if not self.is_streaming or not self.is_paused:
            return False

        self.is_paused = False
        self.metrics['last_frame_time'] = time.time()
        self.logger.info("Streaming resumed")
        return True


    def publish_frame(self, frame: Frame, frame_type: str = 'frame') -> bool:
        """
        Process and publish a frame independently
        
        Args:
            frame: Frame object to publish
            frame_type: Type of frame ('frame' or 'edge_frame')
        """
        if not self.is_streaming or self.is_paused:
            return False

        try:
            # Update metrics in a thread-safe way
            current_time = time.time()
            with self.buffer_lock:
                if self.metrics['last_frame_time'] is not None:
                    time_diff = current_time - self.metrics['last_frame_time']
                    if time_diff >= 1.0:
                        self.metrics['fps'] = round(self.frame_count / time_diff)
                        self.frame_count = 0
                        self.metrics['last_frame_time'] = current_time
                else:
                    self.metrics['last_frame_time'] = current_time
                    self.metrics['fps'] = 0
                
                self.frame_count += 1
                self.metrics['frame_count'] += 1

                # Add metadata
                if not frame.metadata:
                    frame.metadata = {}
                    
                frame.metadata.update({
                    'fps': self.metrics['fps'],
                    'frame_number': self.metrics['frame_count'],
                    'timestamp': current_time
                })

                # Add to frame buffer
                self.frame_buffer.append(frame)

            # Process frame through subscribers asynchronously
            def process_subscribers():
                for subscriber in self.subscribers:
                    try:
                        subscriber.process_frame(frame)
                    except Exception as e:
                        self.logger.error(f"Error in subscriber {subscriber.__class__.__name__}: {e}")
                        with self.buffer_lock:
                            self.metrics['dropped_frames'] += 1

            import threading
            subscriber_thread = threading.Thread(target=process_subscribers)
            subscriber_thread.daemon = True
            subscriber_thread.start()

            # Prepare frame data for publishing
            publish_data = frame.data
            if isinstance(publish_data, np.ndarray):
                success, buffer = cv2.imencode('.jpg', publish_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    raise ValueError("Failed to encode frame")
                publish_data = buffer.tobytes()

            # Publish to each publisher independently
            def publish_to_publisher(publisher):
                try:
                    if hasattr(publisher, 'publish_frame_with_type'):
                        publisher.publish_frame_with_type(Frame(
                            data=publish_data,
                            timestamp=frame.timestamp,
                            frame_number=frame.frame_number,
                            metadata=frame.metadata.copy()
                        ), frame_type)
                    else:
                        publisher.publish_frame(Frame(
                            data=publish_data,
                            timestamp=frame.timestamp,
                            frame_number=frame.frame_number,
                            metadata=frame.metadata.copy()
                        ))
                except Exception as e:
                    self.logger.error(f"Error in publisher {publisher.__class__.__name__}: {e}")
                    with self.buffer_lock:
                        self.metrics['dropped_frames'] += 1

            # Start a thread for each publisher
            publisher_threads = []
            for publisher in self.publishers:
                thread = threading.Thread(target=publish_to_publisher, args=(publisher,))
                thread.daemon = True
                thread.start()
                publisher_threads.append(thread)

            return True

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return False

    def get_stream_metrics(self) -> Dict:
        """Get current stream metrics"""
        current_time = time.time()
        metrics = self.metrics.copy()
        
        if metrics['start_time']:
            metrics['uptime'] = current_time - metrics['start_time']
            
        if self.video_capture:
            metrics['video_fps'] = self.video_capture.get(cv2.CAP_PROP_FPS)
            metrics['total_frames'] = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            metrics['current_position'] = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            
        metrics['buffer_size'] = len(self.frame_buffer)
        metrics['active_clients'] = len(self.active_clients)
        
        return metrics

    def cleanup_resources(self):
        """Clean up streaming resources"""
        self.is_streaming = False
        self.is_paused = False
        self.frame_count = 0
        self.current_file = None
        self.metrics.update({
            'start_time': None,
            'last_frame_time': None,
            'fps': 0,
            'frame_count': 0,
            'dropped_frames': 0
        })

        # Clear frame buffer
        with self.buffer_lock:
            self.frame_buffer.clear()

        # Stop publishers
        for publisher in self.publishers:
            try:
                publisher.stop_publishing()
            except Exception as e:
                self.logger.error(f"Error stopping publisher: {e}")

        # Stop subscribers
        for subscriber in self.subscribers:
            try:
                if hasattr(subscriber, 'cleanup'):
                    subscriber.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up subscriber: {e}")

        # Release video capture
        if self.video_capture:
            try:
                self.video_capture.release()
            except Exception as e:
                self.logger.error(f"Error releasing video capture: {e}")
            finally:
                self.video_capture = None

    def add_client(self, client_id: str):
        """Add a streaming client"""
        self.active_clients.add(client_id)
        self.logger.info(f"Added client: {client_id}")

    def remove_client(self, client_id: str):
        """Remove a streaming client"""
        self.active_clients.discard(client_id)
        self.logger.info(f"Removed client: {client_id}")
        if not self.active_clients:
            self.stop_streaming()

    def get_frame_sample(self, sample_rate: int = 5) -> Optional[Frame]:
        """Get a sample frame from current stream"""
        if not self.is_streaming or not self.frame_buffer:
            return None
            
        try:
            # Get every nth frame
            frames = list(self.frame_buffer)[::sample_rate]
            if not frames:
                return None
                
            # Return middle frame
            return frames[len(frames) // 2]
        except Exception as e:
            self.logger.error(f"Error getting frame sample: {e}")
            return None

    def clear_buffer(self):
        """Clear frame buffer"""
        self.frame_buffer.clear()

    def get_buffer_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'buffer_size': len(self.frame_buffer),
            'buffer_max_size': self.frame_buffer.maxlen,
            'is_locked': self.buffer_lock,
            'timestamp': time.time()
        }
    
    def set_frame_metadata(self, metadata: Dict):
        """Set metadata for next frame"""
        try:
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            
            self.frame_metadata = metadata.copy()  # Make a copy to avoid reference issues
            
            # Update metrics if resolution is provided
            if 'resolution' in metadata:
                self.metrics['resolution'] = metadata['resolution']
                
            # Update timestamp if provided
            if 'timestamp' in metadata:
                self.metrics['last_frame_time'] = metadata['timestamp']
                
            self.logger.debug(f"Set frame metadata: {metadata}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting frame metadata: {e}")
            return False

    def get_frame_metadata(self) -> Optional[Dict]:
        """Get current frame metadata"""
        return self.frame_metadata.copy() if self.frame_metadata else None
