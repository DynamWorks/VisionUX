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
            self.logger.error(f"File not found: {file_path}")
            return jsonify({'error': f'File not found: {filepath}'}), 404
            
        self.logger.info(f"Serving file: {file_path}")
            
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
        
        video_file = data.get('video_file')
        if not video_file:
            return jsonify({'error': 'No video file specified'}), 400
            
        video_path = Path("tmp_content/uploads") / video_file
        if not video_path.exists():
            return jsonify({'error': f'Video file not found: {video_file}'}), 404

        # Initialize CV service
        from backend.services import CVService
        cv_service = CVService()

        # Process video frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video file'}), 500

        # Setup video writer for visualization
        vis_path = Path("tmp_content/visualizations")
        vis_path.mkdir(parents=True, exist_ok=True)
        output_video = vis_path / f"{video_file}_objects.mp4"
        
        # Get original video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use H264 codec for better compatibility and quality
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'avc1'),  # H264 codec
            fps,
            (width, height),
            True  # isColor=True
        )
        
        # Set video writer properties for better quality
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)  # Higher quality

        try:
            detections = []
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Run detection on frame
                result = cv_service.detect_objects(frame)
                if 'error' in result:
                    continue
                    
                # Add frame number and timestamp
                result['frame_number'] = frame_count
                result['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                detections.append(result)
                
                # Draw detections on frame
                for det in result.get('detections', []):
                    bbox = det['bbox']
                    cv2.rectangle(frame, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), 
                        (0, 255, 0), 2)
                    cv2.putText(frame, 
                        f"{det['class']}: {det['confidence']:.2f}", 
                        (int(bbox[0]), int(bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                try:
                    # Ensure frame is in correct format
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # Ensure frame is BGR
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        
                    # Write frame
                    if not writer.write(frame):
                        raise ValueError("Failed to write frame")
                    frame_count += 1
                    
                except Exception as e:
                    logger.error(f"Error writing frame {frame_count}: {e}")
                    continue

        finally:
            # Ensure proper cleanup
            if cap is not None:
                cap.release()
            if 'writer' in locals() and writer is not None:
                writer.release()
                # Verify output file was created
                if not output_video.exists():
                    raise ValueError("Failed to create output video file")
                # Check file size is non-zero
                if output_video.stat().st_size == 0:
                    raise ValueError("Output video file is empty")

        if not detections:
            return jsonify({'error': 'No objects detected'}), 404

        # Setup video writer for visualization
        vis_path = Path("tmp_content/visualizations")
        vis_path.mkdir(parents=True, exist_ok=True)
        output_video = vis_path / f"{video_file}_objects.mp4"
            
        # Get original video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
            
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        # Save results
        analysis_id = f"object_detection_{int(time.time())}"
        results = {
            'video_file': video_file,
            'frame_count': frame_count,
            'detections': detections,
            'visualization': str(output_video),
            'timestamp': time.time()
        }
            
        saved_path = content_manager.save_analysis(results, analysis_id)

        return jsonify({
            'analysis_id': analysis_id,
            'detections': detections,
            'frame_count': frame_count,
            'storage_path': str(saved_path),
            'visualization': str(output_video)
        })

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
        
        video_file = data.get('video_file')
        if not video_file:
            return jsonify({'error': 'No video file specified'}), 400
            
        video_path = Path("tmp_content/uploads") / video_file
        if not video_path.exists():
            return jsonify({'error': f'Video file not found: {video_file}'}), 404

        # Initialize CV service
        from backend.services import CVService
        cv_service = CVService()

        # Process video frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video file'}), 500

        # Setup video writer for visualization
        vis_path = Path("tmp_content/visualizations")
        vis_path.mkdir(parents=True, exist_ok=True)
        output_video = vis_path / f"{video_file}_edges.mp4"
        
        # Get original video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        try:
            edge_results = []
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Run edge detection on frame
                result = cv_service.detect_edges(frame)
                if 'error' in result:
                    continue
                    
                # Add frame number and timestamp
                result['frame_number'] = frame_count
                result['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                edge_results.append(result)
                
                # Write processed frame with edges
                if 'frame' in result:
                    writer.write(result['frame'])
                frame_count += 1

        finally:
            cap.release()

        if not edge_results:
            return jsonify({'error': 'Edge detection failed'}), 404

        # Setup video writer for visualization
        vis_path = Path("tmp_content/visualizations")
        vis_path.mkdir(parents=True, exist_ok=True)
        output_video = vis_path / f"{video_file}_edges.mp4"
            
        # Get original video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
            
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        response_data = {
            'frame_count': frame_count,
            'visualization': str(output_video)
        }

        # Save analysis if requested
        if save_analysis:
            analysis_id = f"edge_detection_{int(time.time())}"
            
            # Convert edge_results to compressed format
            import numpy as np
            compressed_results = []
            for result in edge_results:
                # Convert edges to sparse format - only store non-zero positions
                if 'edges' in result:
                    edges = np.array(result['edges'])
                    non_zero = np.nonzero(edges)
                    compressed_result = {
                        'frame_number': result['frame_number'],
                        'timestamp': result['timestamp'],
                        'shape': edges.shape,
                        'positions': list(zip(non_zero[0].tolist(), non_zero[1].tolist()))
                    }
                else:
                    compressed_result = {
                        'frame_number': result['frame_number'],
                        'timestamp': result['timestamp']
                    }
                compressed_results.append(compressed_result)
                
            results = {
                'video_file': video_file,
                'frame_count': frame_count,
                'edge_results': compressed_results,
                'visualization': str(output_video),
                'timestamp': time.time(),
                'format': 'sparse'  # Indicate compression format
            }
            
            saved_path = content_manager.save_analysis(results, analysis_id)
            response_data.update({
                'analysis_id': analysis_id,
                'storage_path': str(saved_path)
            })

        return jsonify(response_data)

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
            # Create a new video capture for analysis
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
                        frames.append(frame.copy())  # Make a copy of the frame
                        frame_numbers.append(int(pos))
                        timestamps.append(timestamp)
            finally:
                cap.release()

            if not frames:
                return jsonify({'error': 'Failed to capture frames'}), 500

            # Build context information
            context = {
                'video_file': video_file,
                'source_type': stream_type,
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

            # Format scene analysis description
            scene_description = analysis['scene_analysis']['description']
            formatted_description = f"""Scene Analysis Results:

{scene_description}

Analyzed Frames: {len(frames)}
Video Duration: {duration:.2f} seconds
Frame Rate: {fps:.2f} FPS"""

            # Add chat messages to response
            response_data['chat_messages'] = [
                {
                    'role': 'system',
                    'content': 'Starting scene analysis...'
                },
                {
                    'role': 'assistant',
                    'content': formatted_description
                }
            ]

            # Add technical details if available
            if response_data['technical_details']:
                tech_details = json.dumps(response_data['technical_details'], indent=2)
                response_data['chat_messages'].append({
                    'role': 'system',
                    'content': f"Technical Details:\n{tech_details}"
                })

            # Add completion message
            response_data['chat_messages'].append({
                'role': 'system',
                'content': 'Analysis complete - results saved.'
            })

            # Save chat messages to history
            from backend.content_manager import ContentManager
            content_manager = ContentManager()
            content_manager.save_chat_history(
                response_data['chat_messages'],
                video_file
            )

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
