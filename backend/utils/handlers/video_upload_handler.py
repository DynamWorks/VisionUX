import os
import logging
import time
from typing import Any, Dict, Optional
from pathlib import Path
from werkzeug.utils import secure_filename
from .base_handler import BaseHandler
from .progress_handler import ProgressHandler

class VideoUploadHandler(BaseHandler):
    """Handler for video file uploads"""
    
    def __init__(self, upload_path: str = "tmp_content/uploads", progress_handler: Optional[ProgressHandler] = None):
        super().__init__("video_upload", "file")
        self.upload_path = Path(upload_path)
        self.progress_handler = progress_handler
        self.allowed_extensions = {'.mp4', '.avi', '.mov', '.webm'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Initialized VideoUploadHandler with upload path: {self.upload_path}")

    def _handle_impl(self, data: Any) -> Dict:
        """Handle file upload"""
        try:
            if not isinstance(data, dict):
                raise ValueError("Invalid upload data format")

            file = data.get('file')
            if not file:
                raise ValueError("No file provided")

            # Create operation in progress handler
            operation_id = f"upload_{int(time.time())}"
            if self.progress_handler:
                self.progress_handler.start_operation(operation_id, "file_upload")

            # Validate file
            filename = secure_filename(file.filename)
            if not self._is_allowed_file(filename):
                raise ValueError(f"File type not allowed: {filename}")

            if file.content_length > self.max_file_size:
                raise ValueError(f"File too large: {file.content_length} bytes")

            # Save file with progress tracking
            file_path = self.upload_path / filename
            bytes_written = 0
            file_size = file.content_length

            with open(file_path, 'wb') as f:
                while True:
                    chunk = file.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
                    
                    if self.progress_handler:
                        progress = (bytes_written / file_size) * 100
                        self.progress_handler.update_operation(
                            operation_id,
                            progress,
                            f"Uploading {filename}: {bytes_written}/{file_size} bytes"
                        )

            # Complete operation
            if self.progress_handler:
                self.progress_handler.complete_operation(
                    operation_id,
                    f"Upload completed: {filename}"
                )

            return {
                'status': 'success',
                'filename': filename,
                'path': str(file_path.relative_to(self.upload_path)),
                'size': file_path.stat().st_size,
                'operation_id': operation_id,
                'timestamp': time.time()
            }

        except Exception as e:
            self._log_error(e, "Error handling file upload")
            if self.progress_handler:
                self.progress_handler.fail_operation(operation_id, str(e))
            raise

    def validate(self, data: Any) -> bool:
        """Validate upload data"""
        if not isinstance(data, dict):
            return False
        if 'file' not in data:
            return False
        return True

    def _is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return Path(filename).suffix.lower() in self.allowed_extensions

    def get_file_path(self, filename: str) -> Path:
        """Get full path for a file"""
        return self.upload_path / secure_filename(filename)

    def cleanup(self) -> None:
        """Cleanup handler resources"""
        super().cleanup()
        # Optionally clean up old uploads
        self._cleanup_old_files()

    def _cleanup_old_files(self, max_age_days: int = 7) -> None:
        """Clean up files older than max_age_days"""
        try:
            current_time = time.time()
            for file_path in self.upload_path.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > (max_age_days * 24 * 60 * 60):
                        file_path.unlink()
                        self.logger.info(f"Cleaned up old file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")

    def get_upload_stats(self) -> Dict:
        """Get upload statistics"""
        total_size = 0
        file_count = 0
        file_types = {}

        try:
            for file_path in self.upload_path.iterdir():
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
        except Exception as e:
            self.logger.error(f"Error getting upload stats: {e}")

        return {
            'total_size': total_size,
            'file_count': file_count,
            'file_types': file_types,
            'timestamp': time.time()
        }

    def check_upload_space(self) -> Dict:
        """Check available upload space"""
        try:
            total, used, free = os.statvfs(self.upload_path).f_blocks, \
                               os.statvfs(self.upload_path).f_bsize * os.statvfs(self.upload_path).f_bfree, \
                               os.statvfs(self.upload_path).f_bavail * os.statvfs(self.upload_path).f_bsize
            return {
                'total': total,
                'used': used,
                'free': free,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Error checking upload space: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }
