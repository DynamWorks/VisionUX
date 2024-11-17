import json
import time
from pathlib import Path
from flask import Blueprint, request, jsonify, Response
from ..services.scene_service import SceneAnalysisService
from ..services.chat_service import ChatService
from ...utils.memory_manager import MemoryManager
from ..content_manager import ContentManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize processor and memory store with config
from ...utils.config import Config
config = Config()
processor = VideoProcessor(analyzer=ClipVideoAnalyzer(config=config.config))
frame_memory = MemoryManager(content_manager=None)

@api.route('/analyze', methods=['POST'])
def analyze_video():
    """
    Endpoint to analyze video with specified parameters
    
    Expected JSON payload:
    {
        "video_path": "path/to/video.mp4",
        "text_queries": ["car", "person", ...],
        "sample_rate": 1,
        "max_workers": 4
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'video_path' not in data:
            return jsonify({'error': 'Missing video_path parameter'}), 400
            
        # Get video data and save to tmp_content/uploads
        video_path = data['video_path']
        content_manager = ContentManager()
        
        # If video_path is a file object or base64 string, save it
        if 'video_data' in data:
            video_path = content_manager.save_upload(
                data['video_data'].encode() if isinstance(data['video_data'], str) else data['video_data'],
                Path(video_path).name
            )
            
        text_queries = data.get('text_queries', [])
        sample_rate = data.get('sample_rate', 1)
        max_workers = data.get('max_workers', 4)
        
        logger.info(f"Processing video from: {video_path}")
        
        # Create generator for streaming results
        def generate_results():
            def frame_callback(result):
                if not result:
                    logger.warning("Empty frame result received")
                    return
                    
                # Store in frame memory
                frame_memory.add_frame(result)
                
                # Format and yield result
                formatted_result = {
                    'frame_number': result.get('frame_number', 0),
                    'timestamp': result.get('timestamp', 0.0),
                    'detections': result.get('detections', {
                        'segments': [],
                        'lanes': [],
                        'text': [],
                        'signs': [],
                        'tracking': {}
                    }),
                    'memory_size': len(frame_memory.frames)
                }
                yield f"data: {json.dumps(formatted_result)}\n\n"

            return frame_callback

        # Create callback and start processing
        callback = generate_results()
        results = processor.process_video(
            video_path=video_path,
            text_queries=text_queries,
            sample_rate=sample_rate,
            max_workers=max_workers,
            callback=callback
        )
        
        # Format results and store in memory
        formatted_results = []
        for frame_result in results:
            if not isinstance(frame_result, dict):
                logger.warning(f"Unexpected result format: {type(frame_result)}")
                continue
                
            # Store in frame memory
            frame_memory.add_frame(frame_result)
                
            detections = frame_result.get('detections', {})
            if not isinstance(detections, dict):
                detections = {
                    'segments': detections if isinstance(detections, list) else [],
                    'lanes': [],
                    'text': [],
                    'signs': [],
                    'tracking': {}
                }
                
            formatted_results.append({
                'frame_number': frame_result.get('frame_number', 0),
                'timestamp': frame_result.get('timestamp', 0.0),
                'detections': {
                    'segments': detections.get('segments', []),
                    'lanes': detections.get('lanes', []),
                    'text': detections.get('text', []),
                    'signs': detections.get('signs', []),
                    'tracking': detections.get('tracking', {})
                }
            })

        return Response(
            generate_results(),
            mimetype='text/event-stream'
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
        "image_path": "path/to/image.jpg",
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
            
        # Validate image path exists
        from pathlib import Path
        image_path = Path(data['image_path'])
        if not image_path.exists():
            return jsonify({'error': f"Image file not found: {data['image_path']}"}), 400
            
        # Initialize scene analysis service
        from ..services.scene_service import SceneAnalysisService
        scene_service = SceneAnalysisService()
        
        # Analyze scene
        context = data.get('context', '')
        stream_type = data.get('stream_type', 'unknown')
        
        try:
            # Read image file
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                return jsonify({'error': f"Failed to read image: {data['image_path']}"}), 400
                
            analysis = scene_service.analyze_scene(
                image,  # Pass the actual image data
                context=f"Stream type: {stream_type}. {context}"
            )
            return jsonify(analysis)
        except Exception as e:
            return jsonify({
                'error': f"Scene analysis failed: {str(e)}"
            }), 500
        
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
