import json
import time
import cv2
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, Response

from ..services.scene_service import SceneAnalysisService
from ..services.chat_service import ChatService
from ..utils.memory_manager import MemoryManager
from ..utils.config import Config
from ..content_manager import ContentManager

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize services
config = Config()
scene_service = SceneAnalysisService()
chat_service = ChatService()
frame_memory = MemoryManager(content_manager=None)


@api.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'video-analytics-api'
    })

@api.route('/analyze_scene', methods=['POST'])
def analyze_scene():
    """Analyze scene and suggest computer vision applications"""
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'error': 'Missing image_path in request'}), 400

        video_path = Path(data['image_path'])
        if not video_path.exists():
            return jsonify({'error': f"Video file not found: {data['image_path']}"}), 400

        # Extract first frame
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return jsonify({'error': "Failed to read video frame"}), 400

        # Analyze scene
        analysis = scene_service.analyze_scene(
            frame,
            context=f"Stream type: {data.get('stream_type', 'unknown')}. {data.get('context', '')}"
        )

        # Save analysis results
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
    """Endpoint for chat-based video analysis with RAG"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing video_path or prompt'}), 400

        response = chat_service.process_chat(
            data['prompt'],
            data['video_path'],
            use_swarm=data.get('use_swarm', False)
        )

        # Save chat history
        content_manager.save_chat_history(
            [{'role': 'user', 'content': data['prompt']},
             {'role': 'assistant', 'content': response.get('rag_response', '')}],
            f"chat_{int(time.time())}"
        )

        return jsonify(response)

    except Exception as e:
        logger.error("Chat analysis failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/files', methods=['GET'])
def get_files():
    """Get list of uploaded video files"""
    try:
        uploads_path = Path("tmp_content/uploads")
        if not uploads_path.exists():
            uploads_path.mkdir(parents=True, exist_ok=True)
            return jsonify({"files": []})

        files = []
        for file_path in uploads_path.glob('*.mp4'):
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

@api.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'video-analytics-api'
    })

@api.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error("Unhandled error", exc_info=True)
    return jsonify({'error': str(error)}), 500
import json
import time
import cv2
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, Response
from ..services.scene_service import SceneAnalysisService
from ..services.chat_service import ChatService
from ..utils.memory_manager import MemoryManager
from ..utils.config import Config
from ..content_manager import ContentManager

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize services
config = Config()
scene_service = SceneAnalysisService()
chat_service = ChatService()
content_manager = ContentManager()
