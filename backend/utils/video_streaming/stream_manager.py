import logging
import time
from typing import Optional, Dict, List
from .stream_publisher import StreamPublisher, Frame

class StreamManager:
    """Manages video streaming and frame processing"""
    
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

    def register_publisher(self, publisher: StreamPublisher):
        """Register a publisher for frame distribution"""
        self.publishers.append(publisher)
        self.logger.info(f"Registered publisher: {publisher.__class__.__name__}")

    def register_subscriber(self, subscriber):
        """Register a subscriber for frame processing"""
        self.subscribers.append(subscriber)
        self.logger.info(f"Registered subscriber: {subscriber.__class__.__name__}")

    def clear_subscribers(self):
        """Remove all subscribers"""
        self.subscribers.clear()
        self.logger.info("Cleared all subscribers")

    def add_client(self, client_id: str):
        """Add a streaming client"""
        self.active_clients.add(client_id)
        self.logger.info(f"Added client: {client_id}")

    def remove_client(self, client_id: str):
        """Remove a streaming client"""
        self.active_clients.discard(client_id)
        self.logger.info(f"Removed client: {client_id}")

    def start_streaming(self, filename: Optional[str] = None):
        """Start streaming with optional filename"""
        if self.is_streaming and not self.is_paused:
            self.logger.warning("Streaming already in progress")
            return False

        self.is_streaming = True
        self.is_paused = False
        self.frame_count = 0
        self.current_file = filename
        self.last_frame_time = time.time()
        self.fps = 0

        # Start publishers
        for publisher in self.publishers:
            publisher.start_publishing()

        self.logger.info(f"Started streaming{' for file: ' + filename if filename else ''}")
        return True

    def pause_streaming(self):
        """Pause streaming"""
        if not self.is_streaming:
            self.logger.warning("No active stream to pause")
            return False

        if self.is_paused:
            self.logger.warning("Stream already paused")
            return False

        self.is_paused = True
        self.logger.info("Streaming paused")
        return True

    def resume_streaming(self):
        """Resume streaming"""
        if not self.is_streaming:
            self.logger.warning("No active stream to resume")
            return False

        if not self.is_paused:
            self.logger.warning("Stream not paused")
            return False

        self.is_paused = False
        self.last_frame_time = time.time()
        self.logger.info("Streaming resumed")
        return True

    def stop_streaming(self):
        """Stop streaming"""
        if not self.is_streaming:
            self.logger.warning("No active stream to stop")
            return False

        self.is_streaming = False
        self.is_paused = False
        self.frame_count = 0
        self.current_file = None
        self.frame_metadata = None
        self.fps = 0

        # Stop publishers
        for publisher in self.publishers:
            publisher.stop_publishing()

        self.logger.info("Streaming stopped")
        return True

    def set_frame_metadata(self, metadata: Dict):
        """Set metadata for next frame"""
        self.frame_metadata = metadata

    def publish_frame(self, frame: Frame):
        """Process and publish a frame"""
        if not self.is_streaming or self.is_paused:
            return False

        try:
            # Calculate FPS
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            if time_diff >= 1.0:
                self.fps = round(self.frame_count / time_diff)
                self.frame_count = 0
                self.last_frame_time = current_time
            self.frame_count += 1

            # Add metadata if available
            if self.frame_metadata:
                frame.metadata.update(self.frame_metadata)
                self.frame_metadata = None  # Clear after use

            # Add FPS to metadata
            frame.metadata['fps'] = self.fps

            # Process frame through subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber.process_frame(frame)
                except Exception as e:
                    self.logger.error(f"Error in subscriber {subscriber.__class__.__name__}: {e}")

            # Publish processed frame
            for publisher in self.publishers:
                try:
                    publisher.publish_frame(frame)
                except Exception as e:
                    self.logger.error(f"Error in publisher {publisher.__class__.__name__}: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return False

    def get_status(self) -> Dict:
        """Get current streaming status"""
        return {
            'is_streaming': self.is_streaming,
            'is_paused': self.is_paused,
            'frame_count': self.frame_count,
            'current_file': self.current_file,
            'fps': self.fps,
            'active_clients': len(self.active_clients),
            'subscribers': len(self.subscribers),
            'publishers': len(self.publishers)
        }
