import cv2
import numpy as np

class EdgeDetectionService:
    def __init__(self, low_threshold=100, high_threshold=200):
        """
        Initialize edge detection service with Canny parameters
        
        Args:
            low_threshold (int): Lower threshold for Canny edge detection
            high_threshold (int): Higher threshold for Canny edge detection
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
    def detect_edges(self, frame):
        """
        Detect edges in the input frame using Canny edge detection
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            np.ndarray: Edge detection result
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 
                         self.low_threshold, 
                         self.high_threshold)
        
        # Convert back to RGB for visualization
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return edges_rgb
