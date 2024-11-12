import numpy as np
import logging
from typing import List, Dict
from collections import deque
import torch
from transformers import AutoProcessor, AutoModel
from transformers import AutoModel, AutoProcessor

class FrameMemory:
    """Store and query video frame analysis results"""
    
    def __init__(self, max_frames: int = 1000):
        """Initialize frame memory store"""
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        
        # Initialize CLIP for semantic search
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
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
            
    def search(self, query: str, max_results: int = 5, threshold: float = 0.2) -> List[Dict]:
        """
        Search frame memory using semantic similarity with filtering
        
        Args:
            query: Text query to search for
            max_results: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching frame results
        """
        if not self.frames:
            return []
            
        # Generate query embedding
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model.get_text_features(**inputs)
            
        # Handle empty embeddings list
        if not self.embeddings:
            return []
            
        try:
            # Calculate similarities with batched processing
            all_embeddings = torch.stack(self.embeddings).to(self.device)
            query_emb = query_embedding.to(self.device)
            
            # Compute similarities in one batch operation
            similarities = torch.nn.functional.cosine_similarity(
                query_emb.expand(all_embeddings.shape[0], -1),
                all_embeddings
            )
            
            # Convert to numpy for filtering
            similarities = similarities.detach().cpu().numpy()
        except Exception as e:
            logging.error(f"Error computing similarities: {str(e)}")
            return []
        
        # Filter by threshold and get top matches
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            return []
            
        # Sort similarities and get top matches
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[-max_results:][::-1]]
        
        results = []
        frames_list = list(self.frames)
        
        for idx in sorted_indices:
            if 0 <= idx < len(frames_list):
                frame_result = frames_list[idx]
            
            # Extract relevant frame info
            frame_info = {
                'frame_number': frame_result.get('frame_number'),
                'timestamp': frame_result.get('timestamp'),
                'similarity': float(similarities[idx]),
                'detections': frame_result.get('detections', {}),
                'description': self._create_frame_description(frame_result)
            }
            
            # Add scene analysis if available
            if 'analysis' in frame_result:
                frame_info['analysis'] = frame_result['analysis']
                
            results.append(frame_info)
            
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
