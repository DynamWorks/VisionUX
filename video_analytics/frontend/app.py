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
import logging
import threading
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Rerun
RERUN_PORT = 9000  # Port for Rerun web viewer

# Set page config as first Streamlit command
st.set_page_config(
    page_title="Video Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    # Create main container with custom CSS
    st.markdown("""
        <style>
        .main-container {
            padding: 1.5rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            width: calc(100vw - 3rem);
            max-width: calc(100vw - 3rem);
            margin: 1.5rem;
            box-sizing: border-box;
        }
        /* Adjust Streamlit container padding */
        .block-container {
            padding: 1rem !important;
            max-width: 100% !important;
        }
        /* Add padding to all Streamlit elements */
        .element-container {
            padding: 0.5rem 0;
        }
        /* Add spacing between widgets */
        .stButton, .stSelectbox, .stTextInput, .stTextArea, 
        .stNumberInput, .stFileUploader {
            margin: 0.5rem 0;
        }
        /* Remove title padding */
        .stTitle {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        .stApp {
            margin: 0 auto;
        }
        /* Panel styling removed as it was unused */
        /* Make columns responsive */
        @media (max-width: 768px) {
            .stColumns {
                flex-direction: column;
            }
            .control-panel, .viewer-panel, .chat-panel {
                height: auto;
                min-height: 300px;
                margin-bottom: 1rem;
            }
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add title with minimal padding
    st.markdown("<h1 style='margin:0;padding:0.5rem;'>Video Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Create responsive columns
    left_col, center_col, right_col = st.columns([1, 2, 1])
    
    # Left column - Controls and Upload
    with left_col:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.header("Controls")
        
        # Clear tmp_content when new video is uploaded
        video_path = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        if video_path:
            # Clear tmp_content directory
            import shutil
            tmp_content = Path('tmp_content')
            if tmp_content.exists():
                shutil.rmtree(tmp_content)
            tmp_content.mkdir(parents=True)
            
            # Create uploads directory and save video
            uploads_dir = tmp_content / 'uploads'
            uploads_dir.mkdir(exist_ok=True)
            
            video_filename = f"uploaded_video_{int(time.time())}.mp4"
            saved_video_path = uploads_dir / video_filename
            
            with open(saved_video_path, 'wb') as f:
                f.write(video_path.getvalue())
                
            st.session_state.current_video = str(saved_video_path)
            
            st.subheader("Analysis Settings")
            with st.expander("Basic Settings", expanded=True):
                sample_rate = st.slider("Sample Rate (frames)", 1, 60, 30)
                max_workers = st.slider("Max Workers", 1, 8, 4)
            
            with st.expander("Processing Options", expanded=True):
                enable_object_detection = st.checkbox("Object Detection", value=True)
                enable_tracking = st.checkbox("Object Tracking", value=True)
                enable_scene_analysis = st.checkbox("Scene Analysis", value=False)
                enable_swarm_analysis = st.checkbox("Enable Swarm Analysis", value=False,
                    help="Use swarm agents for advanced scene analysis when querying")
        st.markdown('</div>', unsafe_allow_html=True)

    # Center column - Empty for now
    with center_col:
        st.markdown('<div class="viewer-panel">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Perform initial scene analysis if video is uploaded
        if video_path:
            with st.spinner("Analyzing scene..."):
                # Save uploaded video with timestamp in tmp_content
                timestamp = int(time.time())
                video_filename = f"uploaded_video_{timestamp}.mp4"
                video_path_obj = Path("tmp_content") / video_filename
                video_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                with open(video_path_obj, "wb") as f:
                    f.write(video_path.getvalue())
                
                # Store video path in session state
                st.session_state.current_video = str(video_path_obj)
                
                # Read frames starting at 10 seconds
                frames = []
                cap = cv2.VideoCapture(str(video_path_obj))
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                start_time = 10.0  # Start at 10 seconds
                start_frame = int(start_time * fps)
                
                try:
                    # Set frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    
                    # Read 8 frames
                    for _ in range(8):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                finally:
                    cap.release()
            
                if not frames:
                    raise ValueError("Could not read video file")

                try:
                    # Send scene analysis request
                    response = requests.post(
                        "http://localhost:8001/api/v1/analyze_scene",
                        json={
                            "video_path": str(video_path_obj),
                            "context": "Video upload analysis",
                            "stream_type": "uploaded_video",
                            "start_time": 1.0,  # Start at 1 seconds
                            "num_frames": 8      # Analyze 8 frames
                        }
                    )

                    if response.status_code != 200:
                        raise ValueError(f"Scene analysis failed with status {response.status_code}: {response.text}")

                    scene_analysis = response.json()
                    
                    # Display scene analysis results in chat
                    with st.chat_message("assistant"):
                        st.markdown("**Scene Analysis:**")
                        description = scene_analysis.get('scene_analysis', {}).get('description', '')
                        st.markdown(description)
                        
                        st.markdown("\n**Suggested Processing Pipeline:**")
                        for step in scene_analysis.get('suggested_pipeline', []):
                            st.markdown(f"- {step}")

                except Exception as e:
                    st.warning("Scene analysis failed. Continuing with basic processing.")
                    logging.warning(f"Scene analysis failed: {str(e)}")

    # Right column - Chat History
    with right_col:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat container with scrolling
        chat_container = st.container()
        with chat_container:
            st.header("Analysis Chat")
            if not st.session_state.messages:
                st.info("Start analyzing a video to begin the chat interaction")
            else:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Scene Analysis Section
        if video_path:
            st.markdown('<div class="scene-panel">', unsafe_allow_html=True)
            st.header("Scene Analysis")
            # Initialize Rerun viewer
            if not init_rerun():
                st.warning("Rerun visualization unavailable")
            else:
                # Log video to Rerun
                file_bytes = video_path.read()
                temp_video = "temp_video.mp4"
                with open(temp_video, "wb") as f:
                    f.write(file_bytes)
                
                # Reset file pointer for later use
                video_path.seek(0)
                
                # Log video frames to Rerun
                cap = cv2.VideoCapture(temp_video)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rr.log("video", rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                cap.release()
                
                # Clean up temp file
                Path(temp_video).unlink()
            st.markdown('</div>', unsafe_allow_html=True)

    # Chat input below columns
    if video_path:  # Only show chat input if video is uploaded
        if prompt := st.chat_input("Ask about the video...", key="chat_input"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process video with chat prompt
            process_video(video_path, prompt, chat_mode=True, 
                        use_swarm=st.session_state.get('enable_swarm_analysis', False))


def start():
    """Start the frontend server"""
    main()

if __name__ == "__main__":
    start()
