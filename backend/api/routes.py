import json
import time
import cv2
import logging
import os
from pathlib import Path
from flask import Blueprint, request, jsonify, Response, send_file, current_app, send_from_directory
import json
import shutil
import logging
import numpy as np
from typing import Optional, Tuple

def setup_video_writer(video_file: str, cap: cv2.VideoCapture) -> Tuple[cv2.VideoWriter, Path]:
    """Setup video writer with common configuration"""
    vis_path = Path("tmp_content/visualizations")
    vis_path.mkdir(parents=True, exist_ok=True)
    output_video = vis_path / f"{video_file}.mp4"
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (width, height)
    )
    
    return writer, output_video

def validate_video_file(video_file: str) -> Path:
    """Validate video file exists and return path"""
    if not video_file:
        raise ValueError('No video file specified')
        
    video_path = Path("tmp_content/uploads") / video_file
    if not video_path.exists():
        raise FileNotFoundError(f'Video file not found: {video_file}')
        
    return video_path

def handle_frame_write(writer: cv2.VideoWriter, frame: np.ndarray, frame_count: int) -> bool:
    """Handle frame validation and writing with error checking"""
    try:
        if frame is None or frame.size == 0:
            logger.warning(f"Invalid frame at position {frame_count}")
            return False

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        writer.write(frame)
        return True
        
    except Exception as e:
        logger.error(f"Frame write error at position {frame_count}: {e}")
        return False

from backend.utils.video_streaming.stream_manager import StreamManager
from backend.services import SceneAnalysisService, ChatService, CVService
from backend.utils.config import Config
from backend.content_manager import ContentManager
from pathlib import Path

# Initialize services and managers
content_manager = ContentManager()
config = Config()
scene_service = SceneAnalysisService()
chat_service = ChatService()

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

@api.route('/files/clear', methods=['POST'])
def clear_files():
    """Clear all files from tmp_content directory"""
    try:
        tmp_content = Path("tmp_content")
        if tmp_content.exists():
            shutil.rmtree(tmp_content)
            tmp_content.mkdir(parents=True)
            # Recreate required subdirectories
            (tmp_content / "uploads").mkdir()
            (tmp_content / "analysis").mkdir()
            (tmp_content / "chat_history").mkdir()
            (tmp_content / "visualizations").mkdir()
        return jsonify({"status": "success", "message": "All files cleared"})
    except Exception as e:
        logger.error(f"Error clearing files: {e}")
        return jsonify({"error": str(e)}), 500
from backend.services import SceneAnalysisService, ChatService, CVService
from backend.utils.config import Config
from backend.content_manager import ContentManager

# Initialize services and managers
content_manager = ContentManager()
config = Config()
scene_service = SceneAnalysisService()
chat_service = ChatService()
cv_service = CVService()

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

@api.route('/tmp_content/<path:filepath>')
def serve_video(filepath):
    """Serve video files from any tmp_content subdirectory"""
    try:
        # Ensure path is within tmp_content
        base_path = Path("tmp_content").resolve()
        file_path = (base_path / filepath).resolve()
        
        # Security check - ensure file is within tmp_content
        if not str(file_path).startswith(str(base_path)):
            return jsonify({'error': 'Invalid file path'}), 403
            
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': f'File not found: {filepath}'}), 404
            
        logger.info(f"Serving file: {file_path}")
            
        # Get directory and filename
        directory = file_path.parent
        filename = file_path.name
            
        return send_from_directory(directory, filename, as_attachment=False)
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

@api.route('/detect_objects', methods=['POST'])
def detect_objects():
    """Detect objects in video file"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        try:
            video_path = validate_video_file(data.get('video_file'))
        except (ValueError, FileNotFoundError) as e:
            return jsonify({'error': str(e)}), 400

        # Call object detection tool
        from backend.core.analysis_tools import ObjectDetectionTool
        detection_tool = ObjectDetectionTool()
        result = detection_tool._run(video_path)
        
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result), 404
            
        return jsonify(result)

    except Exception as e:
        logger.error("Object detection failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/detect_edges', methods=['POST'])
def detect_edges():
    """Detect edges in video file"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
            
        # Get save_analysis parameter, default to True for backward compatibility
        save_analysis = data.get('save_analysis', True)
        
        try:
            video_path = validate_video_file(data.get('video_file'))
        except (ValueError, FileNotFoundError) as e:
            return jsonify({'error': str(e)}), 400
        # Call edge detection tool
        from backend.core.analysis_tools import EdgeDetectionTool
        edge_tool = EdgeDetectionTool()
        result = edge_tool._run(video_path, save_analysis=save_analysis)
        
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result), 404
            
        return jsonify(result)

    except Exception as e:
        logger.error("Edge detection failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/analyze_scene', methods=['POST'])
def analyze_scene():
    """Analyze scene from video file"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        video_file = data.get('video_file')
        if not video_file:
            return jsonify({'error': 'No video file specified'}), 400
            
        video_path = Path("tmp_content/uploads") / video_file
        if not video_path.exists():
            return jsonify({'error': f'Video file not found: {video_file}'}), 404

        # Use SceneAnalysisTool for analysis
        from backend.core.analysis_tools import SceneAnalysisTool
        scene_tool = SceneAnalysisTool()
        
        try:
            result = scene_tool._run(video_path)
            
            if isinstance(result, str):
                # Extract description from success message
                description = result.replace("Scene analysis completed: ", "")
                
                response_data = {
                    'scene_analysis': {
                        'description': description
                    },
                    'chat_messages': [
                        {
                            'role': 'system',
                            'content': 'Starting scene analysis...'
                        },
                        {
                            'role': 'assistant',
                            'content': description
                        },
                        {
                            'role': 'system',
                            'content': 'Analysis complete - results saved.'
                        }
                    ]
                }
                
                # Save chat messages to history
                content_manager.save_chat_history(
                    response_data['chat_messages'],
                    video_file
                )
                
                return jsonify(response_data)
            else:
                return jsonify({'error': 'Analysis failed'}), 500
                
        except Exception as e:
            logger.error(f"Scene analysis error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error("Scene analysis failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/chat', methods=['POST'])
def chat_analysis():
    """Chat-based video analysis with RAG and tool execution"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing video_path or prompt'}), 400

        # Process chat with tool handling
        response = chat_service.process_chat(
            query=data['prompt'],
            video_path=data['video_path'],
            confirmed=data.get('confirmed', False),
            tool_input=data.get('tool_input', {})
        )
        # Save chat history
        video_name = data.get('video_path')
        if video_name and response.get('chat_messages'):
            content_manager.save_chat_history(
                response['chat_messages'],
                video_name
            )
        
        rag_response = response.get("answer").content if hasattr(response.get("answer"), 'content') else str(response.get("answer"))

        return jsonify({"rag_response": rag_response})

    except Exception as e:
        logger.error("Chat analysis failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/chat/history/<video_name>', methods=['GET'])
def get_chat_history(video_name):
    """Get chat history for a specific video"""
    try:
        chat_history = content_manager.get_chat_history(video_name)
        return jsonify({"messages": chat_history})
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/chat/clear/<video_name>', methods=['POST']) 
def clear_chat_history(video_name):
    """Clear chat history for a specific video"""
    try:
        content_manager.clear_chat_history(video_name)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({"error": str(e)}), 500

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

        # Clear tmp_content directory first
        tmp_content = Path("tmp_content")
        if tmp_content.exists():
            shutil.rmtree(tmp_content)
        
        # Recreate required directories
        uploads_path = tmp_content / "uploads"
        (tmp_content / "analysis").mkdir(parents=True)
        (tmp_content / "chat_history").mkdir(parents=True)
        (tmp_content / "visualizations").mkdir(parents=True)
        (tmp_content / "knowledgebase").mkdir(parents=True)
        uploads_path.mkdir(parents=True)
        
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
            if not hasattr(current_app, 'limiter'):
                return f(*args, **kwargs)  # Fallback if limiter not configured
            limit = current_app.limiter.limit(limit_string)
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
