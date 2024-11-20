import json
import time
import cv2
from pathlib import Path
from flask import Blueprint, request, jsonify, Response
from backend.services.scene_service import SceneAnalysisService
from backend.services.chat_service import ChatService
from backend.utils.memory_manager import MemoryManager
from backend.content_manager import ContentManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize services and memory store with config
from ...utils.config import Config
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
    """
    Analyze scene and suggest computer vision applications
    
    Expected JSON payload:
    {
        "image_path": "path/to/video.mp4",
        "context": "Optional context about the video stream",
        "stream_type": "webcam|traffic|moving_camera"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Missing request data'}), 400
            
        if 'image_path' not in data:
            return jsonify({'error': 'Missing image_path in request'}), 400
            
        if not isinstance(data['image_path'], str):
            return jsonify({'error': 'image_path must be a string'}), 400
            
        # Validate video path exists
        from pathlib import Path
        video_path = Path(data['image_path'])
        if not video_path.exists():
            return jsonify({'error': f"Video file not found: {data['image_path']}"}), 400
            
        # Initialize scene analysis service
        from ..services.scene_service import SceneAnalysisService
        scene_service = SceneAnalysisService()
        
        # Extract first frame from video
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return jsonify({'error': f"Failed to read video frame: {data['image_path']}"}), 400
        
        # Analyze scene
        context = data.get('context', '')
        stream_type = data.get('stream_type', 'unknown')
        
        analysis = scene_service.analyze_scene(
            frame,  # Pass the first frame
            context=f"Stream type: {stream_type}. {context}"
        )
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Scene analysis failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"Analysis failed: {str(e)}"
        }), 500

@api.route('/chat', methods=['POST'])
def chat_analysis():
    """
    Endpoint for chat-based video analysis with RAG and swarm execution
    
    Expected JSON payload:
    {
        "video_path": "path/to/video.mp4",
        "prompt": "What's happening in this video?",
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'video_path' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing required parameters (video_path and prompt)'}), 400
            
        video_path = data['video_path']
        prompt = data['prompt']
        
        logger.info(f"Processing chat analysis with RAG: {prompt}")
        
        # Initialize chat service
        from ..services.chat_service import ChatService
        chat_service = ChatService()
        
        # Process chat query
        response = chat_service.process_chat(prompt, video_path)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing chat analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': str(error)
    }), 500
from flask import Blueprint, jsonify
from pathlib import Path
import os

api = Blueprint('api', __name__)

@api.route('/files', methods=['GET'])
def get_files():
    """Get list of uploaded video files"""
    uploads_path = Path("tmp_content/uploads")
    if not uploads_path.exists():
        uploads_path.mkdir(parents=True, exist_ok=True)
        return jsonify({"files": []})
        
    try:
        files = []
        for file_path in uploads_path.glob('*.mp4'):
            stat = file_path.stat()
            files.append({
                'name': file_path.name,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'path': str(file_path)
            })
        return jsonify({"files": sorted(files, key=lambda x: x['modified'], reverse=True)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
