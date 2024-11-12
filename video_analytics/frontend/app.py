import streamlit as st
import rerun as rr
import cv2
import numpy as np
from pathlib import Path
import requests
import json
from PIL import Image
import io
import time

def init_rerun():
    """Initialize and connect to Rerun"""
    try:
        rr.init("video_analytics/frontend", spawn=True)
        rr.connect()
    except Exception as e:
        st.error(f"Failed to connect to Rerun: {e}")
        st.info("Starting Rerun viewer...")
        time.sleep(2)  # Give time for viewer to start
        try:
            rr.connect()
        except Exception as e:
            st.error(f"Could not connect to Rerun after retry: {e}")
            return False
    return True

def main():
    # Initialize Rerun
    if not init_rerun():
        st.warning("Continuing without Rerun visualization...")
        
    st.title("Video Analytics Dashboard")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        video_path = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        
        if video_path:
            query = st.text_input("Enter query", "Show me cars and people")
            analyze_button = st.button("Analyze Video")
            
            if analyze_button:
                process_video(video_path, query)

def process_video(video_path, query):
    # Create processing columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Video Stream")
        video_placeholder = st.empty()
        
    with col2:
        st.header("Analysis Results")
        results_placeholder = st.empty()
        
    # Save uploaded video temporarily
    temp_path = Path("temp_video.mp4")
    temp_path.write_bytes(video_path.read())
    
    # Start video processing
    try:
        # Send analysis request
        response = requests.post(
            "http://localhost:5000/api/analyze",
            json={
                "video_path": str(temp_path),
                "text_queries": [query],
                "sample_rate": 30,
                "max_workers": 4
            },
            stream=True
        )
        
        # Process streaming results
        cap = cv2.VideoCapture(str(temp_path))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb)
            
            # Log frame to Rerun
            rr.log("video/frame", rr.Image(frame_rgb))
            
            # Get analysis results
            try:
                for line in response.iter_lines():
                    if line:
                        result = json.loads(line.decode().replace('data: ', ''))
                        
                        # Display results
                        results_placeholder.json(result)
                        
                        # Log detections to Rerun
                        for det in result.get('detections', {}).get('segments', []):
                            bbox = det.get('bbox', [0,0,0,0])
                            rr.log("detections", 
                                  rr.Boxes2D(
                                      boxes=[[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]],
                                      labels=[f"{det.get('class', '')}: {det.get('confidence', 0):.2f}"]
                                  ))
                            
                        time.sleep(0.03)  # Control display rate
                        
            except Exception as e:
                st.error(f"Error processing results: {str(e)}")
                break
                
        cap.release()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        # Cleanup
        temp_path.unlink()

if __name__ == "__main__":
    main()
