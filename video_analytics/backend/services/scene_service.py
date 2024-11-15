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

    def analyze_scene(self, frame, context: Optional[str] = None) -> Dict:
        """
        Analyze scene using GPT-4V and suggest use cases
        
        Args:
            frame: Image/video frame to analyze
            context: Optional context about the video stream
            
        Returns:
            Dictionary containing scene analysis and suggested use cases
        """
        try:
            # Encode image for API
            base64_image = self._encode_image(frame)
            
            # Prepare system message with available functions
            system_msg = f"""Analyze the scene and identify:
1. Scene type and setting
2. Main objects and activities
3. Relevant use cases from: {list(self.available_functions.keys())}
4. Specific applications based on scene context

Additional context: {context if context else 'None provided'}"""

            # Call GPT-4V API
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            },
                            {
                                "type": "text",
                                "text": "Analyze this scene and suggest relevant computer vision applications."
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            # Process and structure the response
            analysis = response.choices[0].message.content
            
            # Parse response into structured format
            # (You would implement more sophisticated parsing here)
            structured_response = {
                'scene_analysis': analysis,
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
