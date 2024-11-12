from vllm import LLM, SamplingParams
import logging
from typing import Optional, Dict, List

class VILAService:
    """Service for running VILA model inference"""
    
    def __init__(self, model_name: str = "Efficient-Large-Model/VILA1.5-3b"):
        self.model_name = model_name
        self.llm = None
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the VILA model"""
        try:
            logging.info(f"Loading VILA model: {self.model_name}")
            self.llm = LLM(model=self.model_name)
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
        if not self.llm:
            raise RuntimeError("VILA model not initialized")
            
        # Construct context from frame data
        context = self._build_frame_context(frame_data)
        
        # Build full prompt
        full_prompt = f"""Analyze this video frame:
{context}

User question: {prompt}

Provide a natural, conversational response about what's happening in this frame.
Focus on answering the user's specific question while incorporating relevant details from the frame analysis.
"""

        try:
            # Generate response
            outputs = self.llm.generate(full_prompt, self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            return response
            
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
