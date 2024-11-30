import logging
from typing import Dict, Any, Set
import cv2
import numpy as np
from .stream_publisher import StreamPublisher, Frame

class WebSocketPublisher(StreamPublisher):
    """Publishes frames to WebSocket clients"""
    
    def __init__(self, socketio, stream_socketio):
        super().__init__()
        self.socketio = socketio  # For control messages
        self.stream_socketio = stream_socketio  # For frame streaming
        self.clients: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        self.frame_buffer_size = 5
        self.frame_buffer = []
        self.jpeg_quality = 85

    def _publish_impl(self, frame: Frame, frame_type: str = 'frame') -> None:
        """Publish frame to all connected clients"""
        try:
            if not self.clients:
                return

            # Always ensure frame data is in proper JPEG format
            if isinstance(frame.data, np.ndarray):
                success, buffer = cv2.imencode('.jpg', frame.data, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not success:
                    raise ValueError("Failed to encode frame")
                frame_data = buffer.tobytes()
            elif isinstance(frame.data, bytes):
                frame_data = frame.data
            else:
                raise ValueError(f"Unsupported frame data type: {type(frame.data)}")

            # Convert to base64 for WebSocket transmission
            import base64
            frame_data = base64.b64encode(frame_data).decode('utf-8')

            # Prepare metadata
            metadata = {
                'timestamp': frame.timestamp,
                'frame_number': frame.frame_number,
                'fps': self.fps,
                'quality': self.jpeg_quality
            }

            # Add any processing results
            if frame.metadata:
                metadata.update(frame.metadata)

            # Buffer frame
            self.frame_buffer.append((frame_data, metadata))
            if len(self.frame_buffer) > self.frame_buffer_size:
                self.frame_buffer.pop(0)

            # Log frame info
            self.logger.debug(
                f"Publishing frame: {len(frame_data)} bytes, "
                f"clients: {len(self.clients)}"
            )

            # Use broadcast for efficiency when sending to all clients
            try:
                if self.clients:  # Only emit if there are clients
                    self.stream_socketio.emit(frame_type, frame_data, broadcast=True)
            except Exception as e:
                self.logger.error(f"Error broadcasting frame: {e}")

        except Exception as e:
            self.logger.error(f"Error publishing frame: {e}", exc_info=True)

    def add_client(self, client_id: str) -> None:
        """Add a client to publish frames to"""
        self.clients.add(client_id)
        self.logger.info(f"Added client: {client_id}")
        # Send any buffered frames to new client
        self._send_buffered_frames(client_id)

    def remove_client(self, client_id: str) -> None:
        """Remove a client"""
        self.clients.discard(client_id)
        self.logger.info(f"Removed client: {client_id}")

    def _send_buffered_frames(self, client_id: str) -> None:
        """Send buffered frames to new client"""
        try:
            for frame_data, metadata in self.frame_buffer:
                self.socketio.emit('frame', frame_data, to=client_id)
        except Exception as e:
            self.logger.error(f"Error sending buffered frames to {client_id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics"""
        stats = super().get_stats()
        stats.update({
            'clients': len(self.clients),
            'buffered_frames': len(self.frame_buffer),
            'jpeg_quality': self.jpeg_quality
        })
        return stats

    def _send_frame_to_client(self, client_id: str, frame_data: bytes, metadata: Dict):
        """Send frame to specific client"""
        try:
            # Emit binary frame data directly
            self.socketio.emit('frame', frame_data, room=client_id)
        except Exception as e:
            raise Exception(f"Failed to send frame to client: {e}")

    def start_publishing(self) -> None:
        """Start publishing frames"""
        super().start_publishing()
        self.frame_buffer.clear()
        self.logger.info("Started publishing frames")

    def stop_publishing(self) -> None:
        """Stop publishing frames"""
        super().stop_publishing()
        self.frame_buffer.clear()
        self.logger.info("Stopped publishing frames")

