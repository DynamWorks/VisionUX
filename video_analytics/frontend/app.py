
import streamlit as st
# Set page config as first Streamlit command
st.set_page_config(
    page_title="Video Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import rerun as rr
import cv2
import numpy as np
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

def process_video(video_path, query, sample_rate: int = 30, max_workers: int = 4, 
                 chat_mode=False, use_swarm=False):
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
            # Initialize video stream
            from ..utils.video_stream import VideoStream
            stream = VideoStream(str(video_file_path), loop=True)
            stream.start()
            
            # Send chat request with UI settings
            response = requests.post(
                "http://localhost:8001/api/v1/chat",
                json={
                    "video_path": str(video_file_path),
                    "prompt": query,
                    "sample_rate": sample_rate,
                    "max_workers": max_workers,
                    "use_vila": True,
                    "use_swarm": use_swarm,
                    "stream_mode": True
                },
                stream=True
            )
            
            # Process streaming results
            while True:
                frame_data = stream.read()
                if not frame_data:
                    continue
                    
                frame = frame_data['frame']
                    
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
                    
            stream.stop()
            
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
    finally:
        if 'stream' in locals():
            stream.stop()

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
        
        # Create tmp_content directory path
        from pathlib import Path
        tmp_content_dir = Path('tmp_content')
        
        # Video source selection with session state
        if 'source_type' not in st.session_state:
            st.session_state.source_type = "Upload Video"
        source_type = st.radio("Select Video Source", ["Upload Video", "Use Camera"], 
                             key='source_type')
        
        if source_type == "Upload Video":
            video_path = st.file_uploader("Upload Video", type=['mp4', 'avi'])
            if video_path:
                # Save uploaded video
                tmp_content_dir.mkdir(parents=True, exist_ok=True)
                
                uploads_dir = tmp_content_dir / 'uploads'
                uploads_dir.mkdir(exist_ok=True)
                
                video_filename = f"uploaded_video_{int(time.time())}.mp4"
                saved_video_path = uploads_dir / video_filename
                
                with open(saved_video_path, 'wb') as f:
                    f.write(video_path.getvalue())
                    
                st.session_state.current_video = str(saved_video_path)
                st.session_state.video_source = "file"
                
                # Initialize rerun for video visualization
                import rerun as rr
                try:
                    rr.init("video_analytics", spawn=True)
                    rr.connect()
                except Exception as e:
                    st.warning(f"Failed to initialize rerun: {e}")

                # Video controls in container
                with st.container():
                    st.subheader("Video Controls")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Start Video", type="primary"):
                            st.session_state.video_active = True
                            # Initialize video stream
                            from ..utils.video_stream import VideoStream
                            stream = VideoStream(str(saved_video_path), loop=True)
                            st.session_state.video_stream = stream
                            stream.start()
                    
                    with col2:
                        if st.button("Stop Video", type="secondary", 
                                   disabled=not getattr(st.session_state, 'video_active', False)):
                            if hasattr(st.session_state, 'video_stream'):
                                st.session_state.video_stream.stop()
                                st.session_state.video_active = False
                                del st.session_state.video_stream
                                st.experimental_rerun()

                # Display video feed if active
                if hasattr(st.session_state, 'video_active') and st.session_state.video_active:
                    frame_placeholder = st.empty()
                    while st.session_state.video_active:
                        frame_data = st.session_state.video_stream.read()
                        if frame_data:
                            frame = frame_data['frame']
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb)
                            # Log to rerun
                            if hasattr(st.session_state, '_rerun_initialized'):
                                rr.log("video/frame", rr.Image(frame_rgb))
                        time.sleep(0.03)  # Control frame rate
                    
        else:  # Use Camera
            import sys
            from pathlib import Path
            # Add project root to Python path
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.append(project_root)
            from video_analytics.utils.camera import CameraManager
            camera_mgr = CameraManager()
            available_cameras = camera_mgr.get_available_cameras()
            
            if not available_cameras:
                st.error("No cameras detected on your device")
            else:
                camera_options = {f"{cam['name']} ({cam['system']})": cam['id'] 
                                for cam in available_cameras}
                selected_camera = st.selectbox("Select Camera", list(camera_options.keys()))
                
                # Camera controls in container
                with st.container():
                    st.subheader("Camera Controls")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Start Camera", type="primary"):
                            camera_id = camera_options[selected_camera]
                            cap = camera_mgr.open_camera(camera_id)
                            
                            if cap:
                                # Store camera state in session
                                st.session_state.current_video = f"camera:{camera_id}"
                                st.session_state.video_source = "camera"
                                st.session_state.camera_active = True
                                st.session_state.camera_cap = cap
                                
                                # Initialize rerun for camera visualization
                                import rerun as rr
                                try:
                                    if not hasattr(st.session_state, '_rerun_initialized'):
                                        rr.init("video_analytics", spawn=True)
                                        rr.connect()
                                        st.session_state._rerun_initialized = True
                                except Exception as e:
                                    st.warning(f"Failed to initialize rerun: {e}")
                    
                    with col2:
                        if st.button("Stop Camera", type="secondary", 
                                   disabled=not getattr(st.session_state, 'camera_active', False)):
                            if hasattr(st.session_state, 'camera_cap'):
                                st.session_state.camera_cap.release()
                                st.session_state.camera_active = False
                                del st.session_state.camera_cap
                                st.experimental_rerun()

                # Display camera feed if active
                if hasattr(st.session_state, 'camera_active') and st.session_state.camera_active:
                    frame_placeholder = st.empty()
                    while st.session_state.camera_active:
                        ret, frame = st.session_state.camera_cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb)
                            # Log to rerun
                            if hasattr(st.session_state, '_rerun_initialized'):
                                rr.log("camera/frame", rr.Image(frame_rgb))
                        time.sleep(0.03)  # Control frame rate
            
            # Analysis settings
            st.subheader("Analysis Settings")
            sample_rate = st.slider("Sample Rate (frames)", 1, 60, 30)
            max_workers = st.slider("Max Workers", 1, 8, 4)
            
            # Processing options
            st.subheader("Processing Options")
            enable_object_detection = st.checkbox("Object Detection", value=True)
            enable_tracking = st.checkbox("Object Tracking", value=True)
            enable_scene_analysis = st.checkbox("Scene Analysis", value=True)


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat input - outside columns
    if source_type == "Upload Video" and video_path:
        # Chat header
        st.header("Analysis Chat")
        if prompt := st.chat_input("Ask about the video..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                # Process video analysis with UI settings
                with st.spinner("Analyzing video..."):
                    try:
                        response = requests.post(
                            "http://localhost:8001/api/v1/chat",
                            json={
                                "video_path": str(st.session_state.current_video) if isinstance(st.session_state.current_video, Path) else st.session_state.current_video,
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
