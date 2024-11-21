import logging
import json
from pathlib import Path
import asyncio
from .base_handler import BaseMessageHandler

class VideoUploadHandler(BaseMessageHandler):
    def __init__(self, uploads_path: Path):
        super().__init__()
        self.uploads_path = uploads_path
        self.current_upload = None
        
    async def handle(self, websocket, message_data):
        """Handle video upload messages"""
        try:
            if isinstance(message_data, dict):
                if message_data.get('type') == 'video_upload_start':
                    await self.handle_upload_start(websocket, message_data)
                elif message_data.get('type') == 'video_upload_complete':
                    await self.handle_upload_complete(websocket)
            elif isinstance(message_data, bytes):
                await self.handle_upload_chunk(websocket, message_data)
        except Exception as e:
            self.logger.error(f"Error handling video upload: {e}")
            await self.send_error(websocket, str(e))

    async def handle_upload_start(self, websocket, data):
        """Handle the start of a video upload"""
        try:
            filename = data.get('filename')
            size = data.get('size')
            file_path = self.uploads_path / filename

            # Check if file already exists
            if file_path.exists():
                self.logger.warning(f"File already exists: {filename}")
                await self.send_error(websocket, 'File already exists')
                return

            # Validate file size (e.g., max 500MB)
            max_size = 500 * 1024 * 1024  # 500MB in bytes
            if size > max_size:
                self.logger.warning(f"File too large: {size} bytes")
                await self.send_error(websocket, 
                    f'File size exceeds maximum allowed size of {max_size/1024/1024}MB')
                return

            self.current_upload = {
                'filename': filename,
                'size': size,
                'file_handle': open(file_path, 'wb'),
                'bytes_received': 0
            }
            self.logger.info(f"Starting video upload: {filename} ({size} bytes)")
            await self.send_response(websocket, {'type': 'upload_start_ack'})
            
        except Exception as e:
            self.logger.error(f"Error starting upload: {e}")
            await self.send_error(websocket, str(e))

    async def handle_upload_chunk(self, websocket, chunk):
        """Handle an incoming chunk of video data"""
        if not self.current_upload:
            raise ValueError("No active upload session")
            
        self.current_upload['file_handle'].write(chunk)
        self.current_upload['bytes_received'] += len(chunk)
        
        # Calculate and send progress
        progress = (self.current_upload['bytes_received'] / self.current_upload['size']) * 100
        await self.send_response(websocket, {
            'type': 'upload_progress',
            'progress': round(progress, 2),
            'bytes_received': self.current_upload['bytes_received'],
            'total_bytes': self.current_upload['size']
        })
        
        self.logger.debug(f"Upload progress: {progress:.1f}%")

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
                self.logger.info(f"Upload completed successfully: {self.current_upload['filename']}")
                
                # Get updated file list
                files = []
                for f_path in self.uploads_path.glob('*.mp4'):
                    try:
                        stat = f_path.stat()
                        files.append({
                            'name': f_path.name,
                            'size': stat.st_size,
                            'modified': stat.st_mtime,
                            'path': str(f_path)
                        })
                    except OSError as e:
                        self.logger.error(f"Error accessing file {f_path}: {e}")
                        continue
                
                # Send both completion acknowledgment and updated file list
                await self.send_response(websocket, {
                    'type': 'upload_complete_ack',
                    'filename': self.current_upload['filename'],
                    'size': file_size
                })
                
                await self.send_response(websocket, {
                    'type': 'uploaded_files',
                    'files': sorted(files, key=lambda x: x['modified'], reverse=True)
                })
            else:
                raise FileNotFoundError(f"Uploaded file not found: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error finalizing upload: {str(e)}")
            await self.send_error(websocket, f"Failed to finalize upload: {str(e)}")
        finally:
            self.current_upload = None
