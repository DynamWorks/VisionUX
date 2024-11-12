from flask import Blueprint, request, jsonify
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
        
        results = processor.process_video(
            video_path=video_path,
            text_queries=text_queries,
            sample_rate=sample_rate,
            max_workers=max_workers
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

        return jsonify({
            'status': 'success',
            'total_frames': len(formatted_results),
            'results': formatted_results
        })
        
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

@api.route('/query', methods=['POST'])
def query_frames():
    """
    Query past video frame analysis results with advanced filtering
    
    Expected JSON payload:
    {
        "query": "What vehicles were seen?",
        "max_results": 5,
        "video_path": "path/to/video.mp4",
        "threshold": 0.2,
        "filters": {
            "time_range": [0, 100],
            "object_types": ["car", "truck"],
            "min_confidence": 0.5
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'video_path' not in data:
            return jsonify({'error': 'Missing required parameters (query and video_path)'}), 400
            
        query = data['query']
        max_results = data.get('max_results', 5)
        threshold = data.get('threshold', 0.2)
        filters = data.get('filters', {})
        
        # Search frame memory with filters
        results = frame_memory.search(
            query=query,
            max_results=max_results,
            threshold=threshold
        )
        
        # Apply additional filters
        if filters:
            filtered_results = []
            for result in results:
                # Time range filter
                if 'time_range' in filters:
                    t_start, t_end = filters['time_range']
                    if not (t_start <= result['timestamp'] <= t_end):
                        continue
                
                # Object type filter
                if 'object_types' in filters:
                    objects = [det['class'] for det in result['detections'].get('segments', [])]
                    if not any(obj in filters['object_types'] for obj in objects):
                        continue
                
                # Confidence filter
                if 'min_confidence' in filters:
                    min_conf = filters['min_confidence']
                    detections = result['detections'].get('segments', [])
                    if not any(det['confidence'] >= min_conf for det in detections):
                        continue
                
                filtered_results.append(result)
            
            results = filtered_results[:max_results]
        
        return jsonify({
            'status': 'success',
            'query': query,
            'filters_applied': bool(filters),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
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
