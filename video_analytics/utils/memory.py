import numpy as np
from typing import List, Dict
from collections import deque
import torch
from transformers import AutoProcessor, AutoModel
from vila import VILAModel, VILAProcessor

class FrameMemory:
    """Store and query video frame analysis results"""
    
    def __init__(self, max_frames: int = 1000):
        """Initialize frame memory store"""
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        
        # Initialize VILA for semantic search
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VILAModel.from_pretrained("microsoft/VILA-1.5-3b").to(self.device)
        self.processor = VILAProcessor.from_pretrained("microsoft/VILA-1.5-3b")
        
        # Store frame embeddings
        self.embeddings = []
        
    def add_frame(self, frame_result: Dict):
        """Add frame analysis result to memory"""
        self.frames.append(frame_result)
        
        # Create text description of frame content
        description = self._create_frame_description(frame_result)
        
        # Generate embedding
        inputs = self.processor(
            text=[description],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
            self.embeddings.append(embedding)
            
        # Trim embeddings if needed
        if len(self.embeddings) > self.max_frames:
            self.embeddings.pop(0)
            
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search frame memory using semantic similarity"""
        if not self.frames:
            return []
            
        # Generate query embedding
        # Process query with VILA
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model.get_text_embeddings(**inputs)
            
        # Calculate similarities
        similarities = []
        for emb in self.embeddings:
            sim = torch.nn.functional.cosine_similarity(query_embedding, emb)
            similarities.append(float(sim))
            
        # Get top matches
        top_indices = np.argsort(similarities)[-max_results:][::-1]
        
        results = []
        for idx in top_indices:
            frame_result = list(self.frames)[idx]
            results.append({
                'frame_number': frame_result.get('frame_number'),
                'timestamp': frame_result.get('timestamp'),
                'similarity': similarities[idx],
                'detections': frame_result.get('detections', {})
            })
            
        return results
        
    def _create_frame_description(self, frame_result: Dict) -> str:
        """Create detailed text description of frame content using VILA's understanding"""
        detections = frame_result.get('detections', {})
        
        # Gather scene elements
        objects = [d['class'] for d in detections.get('segments', [])]
        signs = [d['class'] for d in detections.get('signs', [])]
        texts = [d['text'] for d in detections.get('text', [])]
        has_lanes = bool(detections.get('lanes'))
        
        # Create structured description
        description = "In this frame, "
        
        # Object descriptions
        if objects:
            obj_counts = {}
            for obj in objects:
                obj_counts[obj] = obj_counts.get(obj, 0) + 1
            obj_desc = ", ".join([f"{count} {obj}{'s' if count > 1 else ''}" 
                                for obj, count in obj_counts.items()])
            description += f"I observe {obj_desc}. "
        
        # Traffic signs
        if signs:
            description += f"There are traffic signs including {', '.join(signs)}. "
        
        # Lane information
        if has_lanes:
            description += "The frame shows clear lane markings on the road. "
        
        # Text information
        if texts:
            description += f"Text is visible reading: {', '.join(texts)}. "
            
        # Add tracking information
        tracking = detections.get('tracking', {})
        if tracking:
            current = tracking.get('current', 0)
            total = tracking.get('total', 0)
            if current > 0:
                description += f"Currently tracking {current} objects "
                if total > current:
                    description += f"out of {total} total detected. "
                else:
                    description += ". "
                    
        if description == "In this frame, ":
            description = "This frame appears to be empty or contains no notable elements."
            
        return description.strip()
