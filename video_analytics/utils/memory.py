import numpy as np
from typing import List, Dict
from collections import deque
import torch
from transformers import CLIPProcessor, CLIPModel

class FrameMemory:
    """Store and query video frame analysis results"""
    
    def __init__(self, max_frames: int = 1000):
        """Initialize frame memory store"""
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        
        # Initialize CLIP for semantic search
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
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
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model.get_text_features(**inputs)
            
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
        """Create text description of frame content"""
        parts = []
        
        detections = frame_result.get('detections', {})
        
        # Add object detections
        objects = [d['class'] for d in detections.get('segments', [])]
        if objects:
            parts.append(f"Objects detected: {', '.join(objects)}")
            
        # Add traffic signs
        signs = [d['class'] for d in detections.get('signs', [])]
        if signs:
            parts.append(f"Traffic signs: {', '.join(signs)}")
            
        # Add text detections
        texts = [d['text'] for d in detections.get('text', [])]
        if texts:
            parts.append(f"Text detected: {', '.join(texts)}")
            
        # Add lane info
        if detections.get('lanes'):
            parts.append(f"Lane markings detected")
            
        if not parts:
            parts.append("Empty frame")
            
        return " ".join(parts)
