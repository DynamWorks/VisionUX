import asyncio
import websockets
import json
import logging
from pathlib import Path
import numpy as np
import cv2
import base64

class WebSocketHandler:
    def __init__(self):
        self.clients = set()
        self.rrd_path = Path("tmp_content/rrd")
        self.rrd_path.mkdir(parents=True, exist_ok=True)

    async def handle_connection(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Process frame here
                    # For now, just create a simple RRD file
                    rrd_file = self.rrd_path / f"frame_{len(self.clients)}.rrd"
                    
                    # Send back the RRD URL
                    response = {
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
