# Set page config as first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Video Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import rerun as rr
import cv2
import numpy as np
from pathlib import Path
import requests
import json
from PIL import Image
import io
import time
import logging
import threading
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Rerun
RERUN_PORT = 9000  # Port for Rerun web viewer

def check_server_status(url: str = "http://localhost:8001") -> bool:
    """Check if the API server is running"""
    try:
        response = requests.get(f"{url}/api/health")
        return response.status_code == 200
    except:
        return False

def process_video(video_path, query, chat_mode=False, use_swarm=False):
    """Process video with analysis and visualization"""
    # Create processing columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Video Stream")
        video_placeholder = st.empty()
        
    with col2:
        st.header("Analysis Results")
        results_placeholder = st.empty()

    # Check server status first
    if not check_server_status():
        st.error("API server is not running. Please start it with:")
        st.code("python video_analytics/main.py")
        return

    # Use the saved video path from tmp_content
    video_file_path = Path(st.session_state.current_video)
    if not video_file_path.exists():
        st.error("Video file not found in tmp_content")
        return
    
    try:
        # Start video processing
        with st.spinner("Analyzing video..."):
            # Send analysis request
            endpoint = "/api/v1/chat" if chat_mode else "/api/v1/analyze"
            response = requests.post(
                f"http://localhost:8001{endpoint}",
                json={
                    "video_path": str(video_file_path),
                    "prompt": query if chat_mode else None,
                    "text_queries": [query] if not chat_mode else None,
                    "sample_rate": 30,
                    "max_workers": 4,
                    "use_vila": chat_mode,
                    "use_swarm": use_swarm
                },
                stream=True
            )
            
            # Process streaming results
            cap = cv2.VideoCapture(str(video_file_path))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                try:
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb)
                    
                    # Log frame to Rerun
                    rr.log("video/frame", rr.Image(frame_rgb))
                    
                    # Process analysis results
                    for line in response.iter_lines():
                        if line:
                            result = json.loads(line.decode().replace('data: ', ''))
                            
                            # Handle chat mode
                            if chat_mode and "response" in result:
                                with st.chat_message("assistant"):
                                    st.markdown(result["response"])
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": result["response"]
                                })
                            else:
                                # Display analysis results
                                results_placeholder.json(result)
                            
                            # Draw pipeline results
                            if "agent_results" in result:
                                for agent_result in result["agent_results"]:
                                    if agent_result.pipeline_name == "object_detection":
                                        boxes = agent_result.result.get("boxes", [])
                                        names = agent_result.result.get("names", {})
                                        rr.log("detections",
                                              rr.Boxes2D(
                                                  boxes=[[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
                                                  labels=[f"{names[int(b[5])]}: {b[4]:.2f}" for b in boxes]
                                              ))
                                    elif agent_result.pipeline_name == "face_analysis":
                                        landmarks = agent_result.result.get("landmarks", [])
                                        rr.log("faces",
                                              rr.Points2D(
                                                  points=[[l[0], l[1]] for l in landmarks],
                                                  labels=["face"] * len(landmarks)
                                              ))
                            
                            # Log additional detections
                            for det in result.get('detections', {}).get('segments', []):
                                bbox = det.get('bbox', [0,0,0,0])
                                rr.log("detections", 
                                      rr.Boxes2D(
                                          boxes=[[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]],
                                          labels=[f"{det.get('class', '')}: {det.get('confidence', 0):.2f}"]
                                      ))
                                
                            time.sleep(0.03)  # Control display rate
                            
                except Exception as e:
                    st.warning(f"Frame processing error: {str(e)}")
                    continue
                    
            cap.release()
            
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
    finally:
        # No cleanup needed - file is managed by content_manager
        pass

def init_rerun():
    """Initialize and connect to Rerun"""
    try:
        # Initialize Rerun with application name and spawn viewer
        rr.init("video_analytics/frontend", spawn=True)
        
        # Wait briefly for viewer to start
        import time
        time.sleep(1)
        
        # Connect to the viewer
        rr.connect()
        
        # Create viewer container
        viewer_container = st.empty()
        
        # Configure recording settings
        rr.connect()  # Connect to the running viewer
        
        # Store viewer container for later use
        st.session_state.viewer_container = viewer_container
        return True
    except Exception as e:
        st.error(f"Failed to initialize Rerun viewer: {e}")
        return False

def is_ready():
    """Check if frontend is ready"""
    return True

def main():
    """Main application entry point"""
    # Add title
    st.title("Video Analytics Dashboard")
    
    # Create two columns for controls and chat
    controls_col, chat_col = st.columns([1, 1])
    
    # Controls column
    with controls_col:
        st.header("Controls")
        
        # Video upload
        video_path = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        if video_path:
            # Save uploaded video
            tmp_content = Path('tmp_content')
            tmp_content.mkdir(parents=True, exist_ok=True)
            
            uploads_dir = tmp_content / 'uploads'
            uploads_dir.mkdir(exist_ok=True)
            
            video_filename = f"uploaded_video_{int(time.time())}.mp4"
            saved_video_path = uploads_dir / video_filename
            
            with open(saved_video_path, 'wb') as f:
                f.write(video_path.getvalue())
                
            st.session_state.current_video = str(saved_video_path)
            
            # Analysis settings
            st.subheader("Analysis Settings")
            sample_rate = st.slider("Sample Rate (frames)", 1, 60, 30)
            max_workers = st.slider("Max Workers", 1, 8, 4)
            
            # Processing options
            st.subheader("Processing Options")
            enable_object_detection = st.checkbox("Object Detection", value=True)
            enable_tracking = st.checkbox("Object Tracking", value=True)
            enable_scene_analysis = st.checkbox("Scene Analysis", value=True)

    # Chat column
    with chat_col:
        st.header("Analysis Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Chat input
        if video_path:
            if prompt := st.chat_input("Ask about the video..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                # Process video analysis
                with st.spinner("Analyzing video..."):
                    try:
                        response = requests.post(
                            "http://localhost:8001/api/v1/chat",
                            json={
                                "video_path": st.session_state.current_video,
                                "prompt": prompt,
                                "sample_rate": sample_rate,
                                "max_workers": max_workers,
                                "enable_detection": enable_object_detection,
                                "enable_tracking": enable_tracking,
                                "enable_scene": enable_scene_analysis
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            # Add assistant response
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result.get("response", "No response generated")
                            })
                            with st.chat_message("assistant"):
                                st.markdown(result.get("response", "No response generated"))
                        else:
                            st.error(f"Analysis failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Upload a video to start the analysis chat")


def start():
    """Start the frontend server"""
    main()

if __name__ == "__main__":
    start()
