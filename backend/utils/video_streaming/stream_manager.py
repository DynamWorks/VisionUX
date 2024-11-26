import logging
from typing import Dict, Set, Optional
import threading
import queue
import cv2
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class Frame:
    data: np.ndarray
    timestamp: float
    frame_number: int
    metadata: Dict = None

class StreamManager:
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self.logger = logging.getLogger(__name__)
            self.subscribers: Set[StreamSubscriber] = set()
            self.publishers: Set[StreamPublisher] = set()
            self.frame_queue = queue.Queue(maxsize=30)  # Buffer 30 frames
            self.current_frame: Optional[Frame] = None
            self.frame_count = 0
            self.is_streaming = False
            self.stream_thread = None
            self._initialized = True
            
    def register_subscriber(self, subscriber: StreamSubscriber) -> None:
        """Register a new subscriber"""
        self.subscribers.add(subscriber)
        self.logger.info(f"Registered subscriber: {subscriber.__class__.__name__}")
        
    def unregister_subscriber(self, subscriber: StreamSubscriber) -> None:
        """Unregister a subscriber"""
        self.subscribers.discard(subscriber)
        self.logger.info(f"Unregistered subscriber: {subscriber.__class__.__name__}")
        
    def register_publisher(self, publisher: StreamPublisher) -> None:
        """Register a new publisher"""
        self.publishers.add(publisher)
        self.logger.info(f"Registered publisher: {publisher.__class__.__name__}")
        
    def unregister_publisher(self, publisher: StreamPublisher) -> None:
        """Unregister a publisher"""
        self.publishers.discard(publisher)
        self.logger.info(f"Unregistered publisher: {publisher.__class__.__name__}")
        
    def publish_frame(self, frame: np.ndarray, metadata: Dict = None) -> None:
        """Publish a new frame to all subscribers"""
        try:
            if self.frame_queue.full():
                # Drop oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            frame_obj = Frame(
                data=frame,
                timestamp=time.time(),
                frame_number=self.frame_count,
                metadata=metadata or {}
            )
            
            self.frame_queue.put(frame_obj)
            self.current_frame = frame_obj
            self.frame_count += 1
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber.on_frame(frame_obj)
                except Exception as e:
                    self.logger.error(f"Error in subscriber {subscriber.__class__.__name__}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error publishing frame: {e}")
            
    def start_streaming(self) -> None:
        """Start the streaming process"""
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._stream_frames)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
    def stop_streaming(self) -> None:
        """Stop the streaming process"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
            
    def _stream_frames(self) -> None:
        """Main streaming loop"""
        while self.is_streaming:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    # Notify publishers
                    for publisher in self.publishers:
                        try:
                            publisher.publish_frame(frame)
                        except Exception as e:
                            self.logger.error(f"Error in publisher {publisher.__class__.__name__}: {e}")
                else:
                    time.sleep(0.001)  # Small sleep to prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Error in streaming loop: {e}")
                
    def get_current_frame(self) -> Optional[Frame]:
        """Get the most recent frame"""
        return self.current_frame
