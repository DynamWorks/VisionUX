import json
import time
import cv2
import logging
import os
from pathlib import Path
from flask import Blueprint, request, jsonify, Response, send_file, current_app, send_from_directory
import shutil

from backend.utils.video_streaming.stream_manager import StreamManager
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
    """Analyze scene from video file"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        # Get stream type and video file information
        stream_type = data.get('stream_type', 'video')
        video_file = data.get('video_file')
        
        # For video files, verify existence
        if stream_type == 'video':
            if not video_file:
                return jsonify({'error': 'No video file specified'}), 400
            video_path = Path("tmp_content/uploads") / video_file
            if not video_path.exists():
                return jsonify({'error': f'Video file not found: {video_file}'}), 404
            
        num_frames = int(data.get('num_frames', 8))
        
        try:
            # Get frames based on source type
            if stream_type == 'video':
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return jsonify({'error': 'Failed to open video file'}), 500

                try:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0

                    if total_frames < num_frames:
                        num_frames = total_frames

                    interval = max(1, total_frames // num_frames)
                    frame_positions = [i * interval for i in range(num_frames)]

                    frames = []
                    frame_numbers = []
                    timestamps = []

                    for pos in frame_positions:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                        ret, frame = cap.read()
                        if ret:
                            timestamp = pos / fps if fps > 0 else 0
                            frames.append(frame)
                            frame_numbers.append(int(pos))
                            timestamps.append(timestamp)
                finally:
                    cap.release()
            else:
                # Get frames from stream manager
                stream_manager = StreamManager()
                captured_frames = stream_manager.capture_frames_for_analysis(num_frames)
                frames = []
                frame_numbers = []
                timestamps = []
                
                for frame, timestamp, frame_number in captured_frames:
                    frames.append(frame)
                    timestamps.append(timestamp)
                    frame_numbers.append(frame_number)
                    
                duration = max(timestamps) - min(timestamps) if timestamps else 0
                fps = len(frames) / duration if duration > 0 else 0

            if not frames:
                return jsonify({'error': 'Failed to capture frames'}), 500

            # Build context information
            context = {
                'video_file': video_file,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'frame_count': len(frames),
                'total_frames': total_frames,
                'duration': duration,
                'fps': fps,
                'frames_analyzed': frame_numbers
            }

            # Perform analysis
            analysis = scene_service.analyze_scene(
                frames,
                context=str(context),
                frame_numbers=frame_numbers,
                timestamps=timestamps
            )

            if 'error' in analysis:
                return jsonify({'error': analysis['error']}), 500

            # Prepare response
            response_data = {
                'scene_analysis': analysis['scene_analysis'],
                'technical_details': analysis['technical_details'],
                'metadata': {
                    'timestamp': time.time(),
                    'video_file': video_file,
                    'frame_count': len(frames),
                    'frame_numbers': frame_numbers,
                    'duration': duration,
                    'fps': fps
                }
            }

            # Save analysis results
            analysis_id = f"scene_analysis_{int(time.time())}"
            saved_path = content_manager.save_analysis(response_data, analysis_id)
            response_data['storage'] = {'path': str(saved_path), 'id': analysis_id}

            return jsonify(response_data)

        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

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
        uploads_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_content/uploads")).resolve()
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
