import json
from flask import Blueprint, request, jsonify, Response
from ..core.processor import VideoProcessor
from ..core.analyzer import ClipVideoAnalyzer
from ..utils.memory import FrameMemory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize processor and memory store with config
from ..utils.config import Config
config = Config()
processor = VideoProcessor(analyzer=ClipVideoAnalyzer(config=config.config))
frame_memory = FrameMemory()

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
            
        video_path = data['video_path']
        text_queries = data.get('text_queries', [])
        sample_rate = data.get('sample_rate', 1)
        max_workers = data.get('max_workers', 4)
        
        logger.info(f"Processing video: {video_path}")
        
        # Create generator for streaming results
        def generate_results():
            def frame_callback(result):
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
                yield f"data: {jsonify(formatted_result).get_data(as_text=True)}\n\n"

        # Create generator for streaming results
        def frame_callback(result):
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
            yield f"data: {jsonify(formatted_result).get_data(as_text=True)}\n\n"

        results = processor.process_video(
            video_path=video_path,
            text_queries=text_queries,
            sample_rate=sample_rate,
            max_workers=max_workers,
            callback=frame_callback
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
        
        if not data or 'image_path' not in data:
            return jsonify({'error': 'Missing image_path'}), 400
            
        # Initialize scene analysis service
        from ..services.scene_service import SceneAnalysisService
        scene_service = SceneAnalysisService()
        
        # Analyze scene
        context = data.get('context')
        stream_type = data.get('stream_type', 'unknown')
        
        analysis = scene_service.analyze_scene(
            data['image_path'],
            context=f"Stream type: {stream_type}. {context if context else ''}"
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
    Endpoint for chat-based video analysis
    
    Expected JSON payload:
    {
        "video_path": "path/to/video.mp4",
        "prompt": "What's happening in this video?",
        "sample_rate": 30,
        "max_workers": 4,
        "use_vila": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'video_path' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing required parameters (video_path and prompt)'}), 400
            
        video_path = data['video_path']
        prompt = data['prompt']
        sample_rate = data.get('sample_rate', 30)
        max_workers = data.get('max_workers', 4)
        
        logger.info(f"Processing video chat analysis: {video_path}")
        
        # Initialize VILA processor for chat
        from ..core.vila import VILAProcessor
        vila = VILAProcessor()
        
        def generate_chat_response():
            # Process video frames using the chat prompt as text query
            for result in processor.process_video(
                video_path=video_path,
                text_queries=[prompt],  # Use chat prompt as detection query
                sample_rate=sample_rate,
                max_workers=max_workers
            ):
                # Add VILA analysis
                # Use VILA service for analysis and response
                from ..services.vila_service import VILAService
                vila_service = VILAService()
                
                # Generate response using VILA service
                response = vila_service.analyze_frame(result, prompt)
                result['response'] = response
                
                yield f"data: {json.dumps(result)}\n\n"
        
        return Response(
            generate_chat_response(),
            mimetype='text/event-stream'
        )
        
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
