import os
from pathlib import Path
import shutil
import json
import logging
from typing import Optional, Dict, List
import time

class ContentManager:
    """Manages temporary content storage for API outputs and user files"""
    
    def __init__(self, base_path: str = "tmp_content"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.uploads_dir = self.base_path / "uploads"
        self.analysis_dir = self.base_path / "analysis"
        self.chat_dir = self.base_path / "chat_history"
        
        for directory in [self.uploads_dir, self.analysis_dir, self.chat_dir]:
            directory.mkdir(exist_ok=True)
            
        self.logger = logging.getLogger(__name__)

    def save_upload(self, file_data: bytes, filename: str) -> str:
        """Save uploaded file and return path"""
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        file_path = self.uploads_dir / safe_filename
        
        with open(file_path, "wb") as f:
            f.write(file_data)
            
        return str(file_path)

    def save_analysis(self, analysis_data: Dict, source_id: str) -> str:
        """Save analysis results with reference to source"""
        timestamp = int(time.time())
        filename = f"analysis_{source_id}_{timestamp}.json"
        file_path = self.analysis_dir / filename
        
        with open(file_path, "w") as f:
            json.dump(analysis_data, f, indent=2)
            
        return str(file_path)

    def save_chat_history(self, chat_data: List[Dict], session_id: str):
        """Save chat history for a session"""
        file_path = self.chat_dir / f"chat_{session_id}.json"
        
        with open(file_path, "w") as f:
            json.dump(chat_data, f, indent=2)

    def get_recent_analysis(self, source_id: str, limit: int = 5) -> List[Dict]:
        """Get recent analysis results for a source"""
        analysis_files = list(self.analysis_dir.glob(f"analysis_{source_id}_*.json"))
        analysis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        results = []
        for file_path in analysis_files[:limit]:
            with open(file_path) as f:
                results.append(json.load(f))
                
        return results

    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session"""
        file_path = self.chat_dir / f"chat_{session_id}.json"
        if file_path.exists():
            with open(file_path) as f:
                return json.load(f)
        return []

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Remove files older than specified hours"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for directory in [self.uploads_dir, self.analysis_dir, self.chat_dir]:
            for file_path in directory.glob("*"):
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    try:
                        file_path.unlink()
                        self.logger.info(f"Removed old file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove {file_path}: {e}")