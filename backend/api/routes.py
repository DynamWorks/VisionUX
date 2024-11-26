import json
import time
import cv2
import logging
import asyncio
from pathlib import Path
from flask import Blueprint, request, jsonify, Response, send_file, current_app
import shutil

from backend.services.scene_service import SceneAnalysisService
from backend.services.chat_service import ChatService
from backend.utils.memory_manager import MemoryManager
from backend.utils.config import Config
from backend.content_manager import ContentManager

# Initialize services and managers
content_manager = ContentManager()
config = Config()
scene_service = SceneAnalysisService()
chat_service = ChatService()
frame_memory = MemoryManager(content_manager=content_manager)

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize services
config = Config()
scene_service = SceneAnalysisService()
chat_service = ChatService()
frame_memory = MemoryManager(content_manager=None)


@api.route('/ws/restart', methods=['POST'])
def restart_websockets():
    """Restart WebSocket servers"""
    try:
        # Get socket handler instance
        if hasattr(current_app, 'socket_handler'):
            # Close all existing connections
            for client in current_app.socket_handler.clients.copy():
                try:
                    client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting client: {e}")
            
            # Clear clients set
            current_app.socket_handler.clients.clear()
            
            # Reinitialize SocketIO
            current_app.socket_handler.socketio.init_app(
                current_app,
                cors_allowed_origins="*",
                async_mode='gevent'
            )
            
            return jsonify({
                'status': 'success',
                'message': 'WebSocket server restarted'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'WebSocket handler not initialized'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to restart WebSocket server: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/rerun/restart', methods=['POST'])
def restart_rerun():
    """Restart Rerun server"""
    try:
        # Get RerunManager instance
        from backend.utils.rerun_manager import RerunManager
        import asyncio
        
        rerun_manager = RerunManager()
        
        # Run cleanup in event loop
        asyncio.run(rerun_manager.cleanup())
        
        # Reinitialize with fresh state
        rerun_manager.initialize()#clear_existing=True)
        
        return jsonify({
            'status': 'success',
            'message': 'Rerun server restarted'
        })
        
    except Exception as e:
        logger.error(f"Failed to restart Rerun server: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed service status"""
    try:
        # Get RerunManager instance for its health check
        from backend.utils.rerun_manager import RerunManager
        rerun_manager = RerunManager()
        rerun_status = asyncio.run(rerun_manager.check_health())
        
        # Check content manager
        content_status = {
            'status': 'healthy',
            'uploads_dir': content_manager.uploads_dir.exists(),
            'analysis_dir': content_manager.analysis_dir.exists(),
            'chat_dir': content_manager.chat_dir.exists()
        }
        
        # Check WebSocket server status
        ws_status = {'status': 'healthy'}
        try:
            if hasattr(current_app, 'socket_handler') and current_app.socket_handler.socketio:
                ws_status['connections'] = len(current_app.socket_handler.clients)
            else:
                ws_status['status'] = 'not_initialized'
        except Exception as e:
            ws_status['status'] = 'error'
            ws_status['error'] = str(e)

        response = jsonify({
            'status': 'healthy',
            'service': 'video-analytics-api',
            'timestamp': time.time(),
            'components': {
                'api': {'status': 'healthy'},
                'rerun': rerun_status,
                'content': content_status,
                'websocket': ws_status
            }
        })
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', '*')
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

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

@api.route('/stream/start', methods=['POST'])
def start_stream():
    """Start video streaming"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        file_path = Path("tmp_content/uploads") / filename
        if not file_path.exists():
            return jsonify({'error': f"Video file not found: {filename}"}), 400
            
        # Initialize video stream
        from backend.utils.video_stream import VideoStream
        video_stream = VideoStream(str(file_path))
        video_stream.start()
        
        # Initialize RerunManager and reset viewer
        from backend.utils.rerun_manager import RerunManager
        rerun_manager = RerunManager()
        rerun_manager.reset()
        
        # Store video stream in app context
        current_app.video_stream = video_stream
        
        return jsonify({
            'status': 'success',
            'message': 'Stream started',
            'filename': filename
        })
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop video streaming"""
    try:
        # Stop video stream if exists
        if hasattr(current_app, 'video_stream'):
            current_app.video_stream.stop()
            delattr(current_app, 'video_stream')
            
        # Reset Rerun viewer
        from backend.utils.rerun_manager import RerunManager
        rerun_manager = RerunManager()
        rerun_manager.reset()
        
        return jsonify({
            'status': 'success',
            'message': 'Stream stopped'
        })
    except Exception as e:
        logger.error(f"Failed to stop stream: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/stream/pause', methods=['POST'])
def pause_stream():
    """Pause video streaming"""
    try:
        if not hasattr(current_app, 'video_stream'):
            return jsonify({'error': 'No active stream'}), 400
            
        current_app.video_stream.pause()
        return jsonify({
            'status': 'success',
            'message': 'Stream paused'
        })
    except Exception as e:
        logger.error(f"Failed to pause stream: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/stream/resume', methods=['POST'])
def resume_stream():
    """Resume video streaming"""
    try:
        if not hasattr(current_app, 'video_stream'):
            return jsonify({'error': 'No active stream'}), 400
            
        current_app.video_stream.resume()
        return jsonify({
            'status': 'success',
            'message': 'Stream resumed'
        })
    except Exception as e:
        logger.error(f"Failed to resume stream: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/upload', methods=['POST'])
def upload_file():
    """Upload a video file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        # Clear tmp_content directory
        tmp_content = Path("tmp_content")
        if tmp_content.exists():
            shutil.rmtree(tmp_content)
        tmp_content.mkdir(parents=True)
        
        # Create uploads directory
        uploads_path = tmp_content / "uploads"
        uploads_path.mkdir(parents=True)
        
        # Save file
        file_path = uploads_path / file.filename
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'path': str(file_path)
        })
    except Exception as e:
        logger.error("Upload failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/files/list', methods=['GET'])
def get_files_list():
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

@api.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error("Unhandled error", exc_info=True)
    return jsonify({'error': str(error)}), 500

