import logging
import json
from pathlib import Path
import asyncio

class VideoUploadHandler:
    def __init__(self, uploads_path: Path):
        self.uploads_path = uploads_path
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.current_upload = None

    async def handle_upload_start(self, websocket, data):
        """Handle the start of a video upload"""
        try:
            self.current_upload = {
                'filename': data.get('filename'),
                'size': data.get('size'),
                'file_handle': open(self.uploads_path / data.get('filename'), 'wb'),
                'bytes_received': 0
            }
            self.logger.info(f"Starting video upload: {data.get('filename')} ({data.get('size')} bytes)")
            await websocket.send(json.dumps({'type': 'upload_start_ack'}))
        except Exception as e:
            self.logger.error(f"Error starting upload: {e}")
            await websocket.send(json.dumps({
                'type': 'upload_error',
                'error': str(e)
            }))

    async def handle_upload_chunk(self, chunk):
        """Handle an incoming chunk of video data"""
        if not self.current_upload:
            raise ValueError("No active upload session")
            
        self.current_upload['file_handle'].write(chunk)
        self.current_upload['bytes_received'] += len(chunk)
        self.logger.info(f"Wrote {len(chunk)} bytes to {self.current_upload['filename']}")

    async def handle_upload_complete(self, websocket):
        """Handle upload completion"""
        if not self.current_upload:
            return
            
        try:
            self.current_upload['file_handle'].flush()
            self.current_upload['file_handle'].close()
            file_path = self.uploads_path / self.current_upload['filename']
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                self.logger.info(f"Upload completed successfully: {self.current_upload['filename']} ({file_size} bytes)")
                await websocket.send(json.dumps({
                    'type': 'upload_complete_ack',
                    'filename': self.current_upload['filename'],
                    'size': file_size
                }))
                return file_path
            else:
                raise FileNotFoundError(f"Uploaded file not found: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error finalizing upload: {str(e)}")
            await websocket.send(json.dumps({
                'type': 'upload_error',
                'error': f"Failed to finalize upload: {str(e)}"
            }))
            await asyncio.sleep(1)
        finally:
            self.current_upload = None
