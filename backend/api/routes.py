import json
import time
import cv2
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, Response, send_file, current_app, send_from_directory
import shutil

from backend.services import SceneAnalysisService, ChatService
from backend.utils.config import Config
from backend.content_manager import ContentManager

# Initialize services and managers
content_manager = ContentManager()
config = Config()
scene_service = SceneAnalysisService()
chat_service = ChatService()

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

@api.route('/tmp_content/uploads/<path:filename>')
def serve_video(filename):
    """Serve uploaded video files"""
    try:
        # Use absolute path resolved from project root
        uploads_path = Path("tmp_content/uploads").resolve()
        if not uploads_path.exists():
            uploads_path.mkdir(parents=True, exist_ok=True)
            
        file_path = uploads_path / filename
        if not file_path.exists():
            return jsonify({'error': f'File not found: {filename}'}), 404
            
        return send_from_directory(uploads_path, filename, as_attachment=False)
    except Exception as e:
        logger.error(f"Error serving video file: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'service': 'video-analytics-api',
            'components': {
                'api': {'status': 'healthy'},
                'content': content_manager.get_status(),
                'websocket': current_app.socket_handler.get_status() if hasattr(current_app, 'socket_handler') else {'status': 'not_initialized'}
            }
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@api.route('/analyze_scene', methods=['POST'])
def analyze_scene():
    """Analyze scene and suggest computer vision applications"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({'error': 'Missing video_path'}), 400

        video_path = Path(data['video_path'])
        if not video_path.exists():
            return jsonify({'error': 'Video file not found'}), 404

        analysis = scene_service.analyze_scene(
            str(video_path),
            context=f"Analyzing video file: {video_path.name}"
        )

        content_manager.save_analysis(
            analysis,
            f"scene_analysis_{int(time.time())}"
        )

        return jsonify(analysis)

    except Exception as e:
        logger.error("Scene analysis failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/chat', methods=['POST'])
def chat_analysis():
    """Chat-based video analysis with RAG"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing video_path or prompt'}), 400

        response = chat_service.process_chat(
            data['prompt'],
            data['video_path'],
            use_swarm=data.get('use_swarm', False)
        )

        content_manager.save_chat_history(
            [{'role': 'user', 'content': data['prompt']},
             {'role': 'assistant', 'content': response.get('rag_response', '')}],
            f"chat_{int(time.time())}"
        )

        return jsonify(response)

    except Exception as e:
        logger.error("Chat analysis failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/upload', methods=['POST'])
def upload_file():
    """Upload a video file"""
    try:
        logger.info("Upload request received")
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if not file.filename:
            logger.error("No filename provided")
            return jsonify({'error': 'No filename provided'}), 400
            
        # Create uploads directory with absolute path
        uploads_path = Path("tmp_content/uploads").resolve()
        uploads_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Upload directory: {uploads_path}")
            
        # Save file with secure filename
        from werkzeug.utils import secure_filename
        safe_filename = secure_filename(file.filename)
        file_path = uploads_path / safe_filename
        logger.info(f"Saving file to: {file_path}")
        file.save(str(file_path))
        
        # Verify file was saved
        if file_path.exists():
            logger.info(f"File saved successfully: {file_path}")
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': file.filename,
                'path': str(file_path)
            })
        else:
            logger.error(f"File not saved at {file_path}")
            return jsonify({'error': 'Failed to save file'}), 500
    except Exception as e:
        logger.error("Upload failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

from flask_limiter.util import get_remote_address
from functools import wraps

def rate_limit(limit_string):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import current_app
            limiter = current_app.limiter
            limit = limiter.shared_limit(limit_string, scope=get_remote_address)
            return limit(f)(*args, **kwargs)
        return decorated_function
    return decorator

@api.route('/files/list', methods=['GET'])
@rate_limit("200 per hour")
def get_files_list():
    """Get list of uploaded video files"""
    try:
        uploads_path = Path("tmp_content/uploads")
        if not uploads_path.exists():
            uploads_path.mkdir(parents=True, exist_ok=True)
            return jsonify({"files": []})

        files = []
        for file_path in uploads_path.glob('*'):  # List all files
            if file_path.is_file():  # Only include files
                try:
                    stat = file_path.stat()
                    files.append({
                        'name': file_path.name,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'path': str(file_path)
                    })
                except OSError as e:
                    logger.error(f"Error accessing file {file_path}: {e}")
                    continue

        return jsonify({"files": sorted(files, key=lambda x: x['modified'], reverse=True)})

    except Exception as e:
        logger.error("Error listing files", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error("Unhandled error", exc_info=True)
    return jsonify({'error': str(error)}), 500
