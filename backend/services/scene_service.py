from openai import OpenAI
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import numpy as np
import base64
import json
from ..content_manager import ContentManager

class SceneAnalysisService:
    """Service for scene analysis using GPT-4V"""
    
    def __init__(self):
        self.client = OpenAI()
        self.content_manager = ContentManager()
        self.logger = logging.getLogger(__name__)
        
        # Define analysis pipeline functions
        self.pipeline_functions = {
            'object_detection': {
                'description': 'Detect and locate objects in the scene',
                'confidence_threshold': 0.5
            },
            'motion_analysis': {
                'description': 'Analyze motion patterns and changes between frames',
                'min_area': 500
            },
            'scene_classification': {
                'description': 'Classify the overall scene type and setting',
                'confidence_threshold': 0.7
            }
        }

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string"""
        try:
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Failed to encode frame")
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Frame encoding error: {e}")
            raise

    def _prepare_vision_prompt(self, frames_info: List[Dict]) -> str:
        """Prepare detailed prompt for GPT-4V analysis"""
        return f"""Analyze these {len(frames_info)} frames from a video stream and provide:
1. Scene Description:
   - Setting and environment
   - Main objects and their positions
   - Activities or events occurring
   - Lighting and visual conditions

2. Changes and Motion:
   - Notable changes between frames
   - Movement patterns or trajectories
   - Stability and camera motion

3. Technical Analysis:
   - Image quality assessment
   - Key areas of interest
   - Potential processing challenges

4. Recommendations:
   - Suggested computer vision tasks
   - Optimal processing pipeline
   - Potential applications

Focus on providing detailed, actionable insights that would be useful for computer vision processing.
"""

    def analyze_scene(self, frames: List[np.ndarray], context: Optional[str] = None,
                     frame_numbers: Optional[List[int]] = None,
                     timestamps: Optional[List[float]] = None) -> Dict:
        """
        Analyze scene using GPT-4V and suggest processing pipeline
        """
        try:
            if not frames:
                raise ValueError("No frames provided for analysis")

            # Prepare frames info with metadata
            frames_info = []
            for i, frame in enumerate(frames):
                frame_info = {
                    'frame_number': frame_numbers[i] if frame_numbers else i,
                    'timestamp': timestamps[i] if timestamps else time.time(),
                    'shape': frame.shape
                }
                frames_info.append(frame_info)

            # Prepare vision API request
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self._prepare_vision_prompt(frames_info)
                        }
                    ]
                }
            ]

            # Add frames to user message
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": context or "Analyze these frames:"}]
            }

            # Add each frame as base64 image
            for frame in frames:
                encoded_frame = self._encode_frame(frame)
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_frame}"
                    }
                })

            messages.append(user_message)

            # Call GPT-4V API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000
            )

            # Process and structure the response
            analysis = response.choices[0].message.content
            
            # Parse response into structured format
            structured_response = {
                'scene_analysis': {
                    'description': analysis,
                    'timestamp': time.time(),
                    'frames_analyzed': len(frames),
                    'frame_numbers': frame_numbers if frame_numbers else list(range(len(frames))),
                    'context': context
                },
                'technical_details': {
                    'resolution': f"{frames[0].shape[1]}x{frames[0].shape[0]}",
                    'frame_count': len(frames),
                    'frame_timestamps': timestamps if timestamps else []
                }
            }

            # Save analysis to disk with video filename and timestamp
            timestamp = int(time.time())
            video_name = context.get('video_file', 'unknown') if isinstance(context, dict) else 'unknown'
            analysis_id = f"{video_name}_analysis_{timestamp}"
            analysis_path = self.content_manager.save_analysis(
                structured_response,
                analysis_id
            )
            structured_response['storage'] = {
                'path': str(analysis_path),
                'id': analysis_id,
                'video_file': video_name,
                'timestamp': timestamp
            }

            self.logger.info(f"Scene analysis completed and saved to {analysis_path}")
            return structured_response

        except Exception as e:
            self.logger.error(f"Scene analysis failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': time.time()
            }

    def get_recent_analyses(self, limit: int = 5) -> List[Dict]:
        """Get recent scene analyses"""
        try:
            analysis_dir = Path(self.content_manager.analysis_dir)
            analysis_files = sorted(
                analysis_dir.glob("scene_analysis_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:limit]

            results = []
            for file_path in analysis_files:
                try:
                    with open(file_path) as f:
                        results.append(json.load(f))
                except Exception as e:
                    self.logger.error(f"Error reading analysis file {file_path}: {e}")

            return results
        except Exception as e:
            self.logger.error(f"Error getting recent analyses: {e}")
            return []
