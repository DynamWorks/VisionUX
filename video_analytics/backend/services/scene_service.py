from openai import OpenAI
import logging
from typing import Dict, List, Optional
import cv2
import numpy as np
import base64
import io
from PIL import Image

class SceneAnalysisService:
    """Service for scene analysis using GPT-4V"""
    
    def __init__(self):
        self.client = OpenAI()
        self.available_functions = {
            'object_detection': {
                'function': 'yolov8',
                'inputs': ['image'],
                'outputs': ['bboxes', 'labels', 'confidences']
            },
            'segmentation': {
                'function': 'sam2',
                'inputs': ['image', 'prompt'],
                'outputs': ['masks', 'scores']
            },
            'face_analysis': {
                'function': 'mediapipe',
                'inputs': ['image'],
                'outputs': ['face_landmarks', 'face_mesh']
            },
            'traffic_analysis': {
                'function': 'traffic_analyzer',
                'inputs': ['video_stream', 'roi'],
                'outputs': ['vehicle_count', 'direction', 'speed']
            }
        }

    def _encode_image(self, image) -> str:
        """Convert image to base64 string"""
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Failed to encode image")
            return base64.b64encode(buffer).decode('utf-8')
        else:
            raise ValueError("Unsupported image format")

    def analyze_scene(self, frames: List[np.ndarray], context: Optional[str] = None,
                     frame_numbers: Optional[List[int]] = None,
                     timestamps: Optional[List[float]] = None) -> Dict:
        """
        Analyze multiple frames using GPT-4V and suggest use cases
        
        Args:
            frames: List of frames to analyze
            context: Optional context about the video stream
            frame_numbers: Optional list of frame numbers
            timestamps: Optional list of timestamps
            
        Returns:
            Dictionary containing scene analysis and suggested use cases
        """
        try:
            # Prepare content array with text and images
            content = [
                {
                    "type": "text",
                    "text": "Analyze these frames from a video and suggest relevant computer vision applications. Note any changes or patterns between frames."
                }
            ]
            
            # Validate and add frames
            if not frames:
                raise ValueError("No frames provided for analysis")
                
            # Add all frames at once in the content array
            for frame in frames[:8]:  # Still limit to 8 frames
                try:
                    if isinstance(frame, str):
                        # Frame is already base64 encoded
                        encoded_frame = frame
                    else:
                        # Frame needs encoding
                        encoded_frame = self._encode_image(frame)
                        
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_frame}"
                        }
                    })
                except Exception as e:
                    logging.error(f"Failed to encode frame: {str(e)}")
                    continue
                    
            if len(content) <= 1:  # Only has the text prompt
                raise ValueError("No valid frames could be encoded for analysis")
            
            # Prepare system message with available functions
            system_msg = f"""Analyze the video frames and identify:
1. Scene type and setting
2. Main objects and activities
3. Any changes or patterns between frames
4. Relevant use cases from: {list(self.available_functions.keys())}
5. Specific applications based on scene context

Additional context: {context if context else 'None provided'}"""

            # Call GPT-4V API with multiple frames
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=500
            )

            # Process and structure the response
            analysis = response.choices[0].message.content
            
            # Parse response into structured format
            structured_response = {
                'scene_analysis': {
                    'description': analysis,
                    'type': 'scene_analysis'
                },
                'available_functions': self.available_functions,
                'suggested_pipeline': self._suggest_pipeline(analysis)
            }
            
            return structured_response
            
        except Exception as e:
            logging.error(f"Scene analysis failed: {str(e)}")
            return {
                'error': str(e),
                'scene_analysis': None,
                'available_functions': self.available_functions
            }

    def _suggest_pipeline(self, analysis: str) -> List[str]:
        """Suggest processing pipeline based on scene analysis"""
        pipeline = []
        
        # Add relevant functions based on keywords in analysis
        if any(kw in analysis.lower() for kw in ['face', 'person', 'people']):
            pipeline.append('face_analysis')
        if any(kw in analysis.lower() for kw in ['car', 'vehicle', 'traffic']):
            pipeline.append('traffic_analysis')
        if any(kw in analysis.lower() for kw in ['object', 'detect']):
            pipeline.append('object_detection')
        
        return pipeline
