from .base_handler import BaseMessageHandler
from pathlib import Path

class FileListHandler(BaseMessageHandler):
    def __init__(self, uploads_path: Path):
        super().__init__()
        self.uploads_path = uploads_path
        
    async def handle(self, websocket, message_data):
        """Handle get_uploaded_files message type"""
        try:
            files = await self.get_uploaded_files()
            self.logger.info(f"Sending file list: {len(files)} files found")
            await self.send_response(websocket, {
                'type': 'uploaded_files',
                'files': files
            })
        except Exception as e:
            self.logger.error(f"Error getting file list: {e}")
            await self.send_error(websocket, f"Failed to get file list: {str(e)}")
            
    async def get_uploaded_files(self):
        """Get list of uploaded files with metadata"""
        if not self.uploads_path.exists():
            self.logger.warning(f"Creating uploads directory: {self.uploads_path}")
            self.uploads_path.mkdir(parents=True, exist_ok=True)
            return []
            
        files = []
        for file_path in self.uploads_path.glob('*.mp4'):
            try:
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'path': str(file_path)
                })
            except OSError as e:
                self.logger.error(f"Error accessing file {file_path}: {e}")
                continue
                
        return sorted(files, key=lambda x: x['modified'], reverse=True)
