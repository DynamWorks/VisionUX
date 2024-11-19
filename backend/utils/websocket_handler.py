import asyncio
import websockets
import json
import logging
import time
from pathlib import Path
import numpy as np
import cv2
import base64
import rerun as rr
from services.edge_detection_service import EdgeDetectionService

class WebSocketHandler:
    def __init__(self):
        self.clients = set()
        self.uploads_path = Path("tmp_content/uploads")
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Initialized WebSocket handler with uploads path: {self.uploads_path}")
        
        # Initialize services
        self.edge_detector = EdgeDetectionService()
        
        # Initialize Rerun
        rr.init("video_analytics")

    async def handle_connection(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Try to parse as JSON first
                try:
                    data = json.loads(message) if isinstance(message, str) else None
                    message_type = data.get('type') if data else None
                    logging.info(f"Received message type: {message_type}")
                except:
                    message_type = None
                    data = None
                    logging.warning("Could not parse message type")

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
                                    
                                    logging.info(f"Starting upload of {filename} ({total_size} bytes)")
                                    
                                    # Open file for writing chunks
                                    with open(file_path, 'wb') as f:
                                        while True:
                                            try:
                                                chunk_meta = await websocket.recv()
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

    async def start_server(self, host='localhost', port=8001):
        async with websockets.serve(self.handle_connection, host, port):
            await asyncio.Future()  # run forever
