from transformers import AutoModel, AutoTokenizer
import torch
import logging
from typing import Optional, Dict, List

class VILAService:
    """Service for running VILA model inference"""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", use_cpu: bool = True):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_cpu = use_cpu
        self.device = "cpu" if use_cpu else "cuda"
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the VILA model"""
        try:
            logging.info(f"Loading VILA model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            
            self.processor = LlavaProcessor.from_pretrained(self.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if not self.use_cpu else torch.float32,
                device_map="auto" if not self.use_cpu else "cpu",
                trust_remote_code=True
            )
            logging.info("VILA model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load VILA model: {str(e)}")
            raise
            
    def analyze_frame(self, frame_data: Dict, prompt: str) -> str:
        """
        Analyze a video frame using VILA
        
        Args:
            frame_data: Dictionary containing frame analysis data
            prompt: User query about the frame
            
        Returns:
            VILA's response about the frame
        """
        try:
            # Convert frame data to PIL Image
            import numpy as np
            from PIL import Image
            frame = frame_data.get('frame', None)
            if frame is None:
                return "No frame data available for analysis."
                
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            elif isinstance(frame, Image.Image):
                image = frame
            else:
                return "Invalid frame format for analysis."

            # Process image and text inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1
            )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logging.error(f"VILA inference failed: {str(e)}")
            return "Sorry, I couldn't analyze this frame at the moment."
            
    def _build_frame_context(self, frame_data: Dict) -> str:
        """Build analysis context from frame data"""
        context = []
        
        # Add detections
        if "detections" in frame_data:
            dets = frame_data["detections"]
            if "segments" in dets:
                objects = [f"{d['class']} ({d['confidence']:.2f})" 
                          for d in dets["segments"]]
                context.append(f"Detected objects: {', '.join(objects)}")
                
        # Add VILA analysis if present
        if "vila_analysis" in frame_data:
            vila = frame_data["vila_analysis"]
            if "scene_type" in vila:
                context.append(f"Scene type: {vila['scene_type']}")
            if "activities" in vila:
                context.append(f"Activities: {', '.join(vila['activities'])}")
                
        return "\n".join(context)
