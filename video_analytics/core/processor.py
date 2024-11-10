import cv2
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from ..core.analyzer import ClipVideoAnalyzer

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from .agents import AnalysisOrchestrator

@dataclass
class FrameConfig:
    """Configuration for frame processing"""
    sample_rate: int = 1
    batch_size: int = 4
    slice_height: int = 384
    slice_width: int = 384
    overlap_ratio: float = 0.2

class VideoProcessor:
    """Advanced video frame processor with configurable processing"""
    
    def __init__(self, analyzer: ClipVideoAnalyzer = None):
        self.analyzer = analyzer or ClipVideoAnalyzer()
        self.orchestrator = AnalysisOrchestrator()
        self.logger = logging.getLogger(__name__)
        self.frame_config = FrameConfig()

    def process_video(self, video_path: str, text_queries: List[str],
                     sample_rate: int = 1, max_workers: int = 4,
                     analysis_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Process a video file using the analyzer
        
        Args:
            video_path: Path to input video file
            text_queries: List of text descriptions to detect
            sample_rate: Process every nth frame
            max_workers: Number of parallel workers
            
        Returns:
            List of frame analysis results
        """
        if not video_path:
            raise ValueError("Video path cannot be empty")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(f"Processing video: {total_frames} frames at {fps} FPS")

        results = []
        processed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                if frame_count % sample_rate != 0:
                    continue

                # Create a copy of the frame to prevent modification during processing
                frame_copy = frame.copy()
                #from nose.tools import set_trace; set_trace()
                
                # Validate frame dimensions and content
                if frame_copy.size == 0 or len(frame_copy.shape) != 3:
                    self.logger.warning(f"Invalid frame at position {frame_count}")
                    continue

                future = executor.submit(
                    self.analyzer.analyze_frame,
                    frame_copy,
                    text_queries
                )
                futures.append((frame_count, future))

                # Process completed frames
                for frame_idx, future in futures[:]:
                    if future.done():
                        try:
                            frame_results = future.result()
                            timestamp = frame_idx / fps
                            
                            # Always ensure we have a valid result dictionary
                            if not isinstance(frame_results, dict):
                                frame_results = self.analyzer._get_default_result()
                                
                            frame_results['timestamp'] = timestamp
                            frame_results['frame_number'] = frame_idx
                            
                            # Validate detections structure
                            if not isinstance(frame_results.get('detections'), dict):
                                frame_results['detections'] = {
                                    'segments': [],
                                    'lanes': [],
                                    'text': [],
                                    'signs': [],
                                    'tracking': {'current': 0, 'total': 0}
                                }
                                
                            results.append(frame_results)
                            
                            processed_count += 1
                            futures.remove((frame_idx, future))
                            
                            self._log_progress(processed_count, total_frames)
                        except Exception as e:
                            self.logger.error(f"Error processing frame {frame_idx}: {e}")

            # Wait for remaining frames
            self.logger.info("Finalizing processing...")
            for frame_idx, future in futures:
                try:
                    frame_results = future.result()
                    if frame_results is not None:
                        timestamp = frame_idx / fps
                        
                        # Create default structure if None
                        if not isinstance(frame_results, dict):
                            frame_results = {
                                'detections': {
                                    'segments': [],
                                    'lanes': [],
                                    'text': [],
                                    'signs': [],
                                    'tracking': {'current': 0, 'total': 0}
                                }
                            }
                            
                        frame_results['timestamp'] = timestamp
                        frame_results['frame_number'] = frame_idx
                        results.append(frame_results)
                    else:
                        self.logger.warning(f"Skipping frame {frame_idx} - received None result")
                    
                    processed_count += 1
                    self._log_progress(processed_count, total_frames)
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx}: {e}")

        cap.release()
        return results

    def _log_progress(self, processed: int, total: int):
        """Log processing progress"""
        progress = (processed * 100) / total
        self.logger.info(f"Progress: {progress:.1f}% - Processed {processed} frames")


class ProcessingQueue:
    """Queue for managing frame processing"""
    def __init__(self, max_size: int = 10):
        self.frame_queue = Queue(maxsize=max_size)
        self.result_queue = Queue(maxsize=max_size)
        self.is_running = False
        
    def start(self):
        """Start processing queue"""
        self.is_running = True
        
    def stop(self):
        """Stop processing queue"""
        self.is_running = False
        
    def put_frame(self, frame):
        """Add frame to processing queue"""
        self.frame_queue.put(frame)
        
    def get_result(self):
        """Get processed result"""
        return self.result_queue.get()
        
    def put_result(self, result):
        """Add result to queue"""
        self.result_queue.put(result)
        
    def get_frame(self):
        """Get frame from queue"""
        return self.frame_queue.get()
