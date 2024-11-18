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
                except:
                    message_type = None
                    data = None

                if message_type == 'video_upload':
                    # Next message will be the video file
                    message = await websocket.recv()
                    if isinstance(message, bytes):
                        filename = f"video_{len(self.clients)}_{int(time.time())}.mp4"
                        file_path = self.uploads_path / filename
                        with open(file_path, 'wb') as f:
                            f.write(message)
                        await websocket.send(json.dumps({
                            "type": "upload_complete",
                            "path": str(file_path)
                        }))
                
                elif message_type == 'camera_frame':
                    # Next message will be the frame data
                    message = await websocket.recv()
                    if isinstance(message, bytes):
                        # Handle live stream frame
                        nparr = np.frombuffer(message, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
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

    async def start_server(self, host='localhost', port=8000):
        async with websockets.serve(self.handle_connection, host, port):
            await asyncio.Future()  # run forever
