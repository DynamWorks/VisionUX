from flask import Blueprint, request, jsonify
from ..core.processor import VideoProcessor
from ..core.analyzer import ClipVideoAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize processor with config
from ..utils.config import Config
config = Config()
processor = VideoProcessor(analyzer=ClipVideoAnalyzer(config=config.config))

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
        
        return jsonify({
            'status': 'success',
            'results': results
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

@api.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': str(error)
    }), 500
