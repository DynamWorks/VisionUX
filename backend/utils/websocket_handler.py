import asyncio
import websockets
import json
import logging
import time
from pathlib import Path
import numpy as np
import cv2
import base64

class WebSocketHandler:
    def __init__(self):
        self.clients = set()
        self.rrd_path = Path("tmp_content/rrd")
        self.uploads_path = Path("tmp_content/uploads")
        self.rrd_path.mkdir(parents=True, exist_ok=True)
        self.uploads_path.mkdir(parents=True, exist_ok=True)

    async def handle_connection(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message) if isinstance(message, str) else None
                
                if isinstance(message, bytes):
                    if data and data.get('type') == 'video_upload':
                        # Handle video file upload
                        filename = f"video_{len(self.clients)}_{int(time.time())}.mp4"
                        file_path = self.uploads_path / filename
                        with open(file_path, 'wb') as f:
                            f.write(message)
                        response = {
                            "type": "upload_complete",
                            "path": str(file_path)
                        }
                    else:
                        # Handle live stream frame
                        nparr = np.frombuffer(message, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # Process frame here
                        rrd_file = self.rrd_path / f"frame_{len(self.clients)}.rrd"
                        
                        response = {
                            "type": "frame_processed",
                            "rrdUrl": str(rrd_file)
                        }
                    
                    await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)

    async def start_server(self, host='localhost', port=8000):
        async with websockets.serve(self.handle_connection, host, port):
            await asyncio.Future()  # run forever
