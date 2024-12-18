import os
import logging
from typing import Any, Dict, List
from pathlib import Path
from .base_handler import BaseHandler

class FileListHandler(BaseHandler):
    """Handler for file listing operations"""
    
    def __init__(self, base_path: str = "tmp_content/uploads"):
        super().__init__("file_list", "file")
        self.base_path = Path(base_path)
        self.allowed_extensions = {'.mp4', '.avi', '.mov', '.webm'}
        self.files_cache = {}
        self.logger.info(f"Initialized FileListHandler with base path: {self.base_path}")

    def _handle_impl(self, data: Any) -> Dict:
        """Handle file list request"""
        try:
            files = self._get_files()
            return {
                'status': 'success',
                'files': files,
                'timestamp': self._get_timestamp(),
                'count': len(files)
            }
        except Exception as e:
            self._log_error(e, "Error getting file list")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': self._get_timestamp()
            }

    def validate(self, data: Any) -> bool:
        """Validate request data"""
        return True  # No input data needed for file listing

    def _get_files(self) -> List[Dict]:
        """Get list of files with metadata"""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
            return []

        files = []
        for file_path in self.base_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.allowed_extensions:
                try:
                    stats = file_path.stat()
                    file_info = {
                        'name': file_path.name,
                        'path': str(file_path.relative_to(self.base_path)),
                        'size': stats.st_size,
                        'modified': stats.st_mtime,
                        'type': file_path.suffix.lower()[1:],
                        'url': f'/api/v1/files/{file_path.name}'
                    }
                    files.append(file_info)
                    self.files_cache[file_path.name] = file_info
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")

        return sorted(files, key=lambda x: x['modified'], reverse=True)

    def get_file_info(self, filename: str) -> Dict:
        """Get information about a specific file"""
        if filename in self.files_cache:
            return self.files_cache[filename]

        file_path = self.base_path / filename
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {filename}")

        if file_path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(f"Invalid file type: {file_path.suffix}")

        stats = file_path.stat()
        file_info = {
            'name': file_path.name,
            'path': str(file_path.relative_to(self.base_path)),
            'size': stats.st_size,
            'modified': stats.st_mtime,
            'type': file_path.suffix.lower()[1:],
            'url': f'/api/v1/files/{file_path.name}'
        }
        self.files_cache[filename] = file_info
        return file_info

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists"""
        try:
            self.get_file_info(filename)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def get_file_path(self, filename: str) -> Path:
        """Get the full path for a file"""
        file_info = self.get_file_info(filename)
        return self.base_path / file_info['path']

    def cleanup(self) -> None:
        """Cleanup handler resources"""
        super().cleanup()
        self.files_cache.clear()

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

    def refresh_cache(self) -> None:
        """Force refresh of file cache"""
        self.files_cache.clear()
        self._get_files()  # Rebuild cache
