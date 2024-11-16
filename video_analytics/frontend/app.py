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

def process_video(video_path, query, chat_mode=False):
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

    # Save uploaded video temporarily
    temp_path = Path("temp_video.mp4")
    temp_path.write_bytes(video_path.read())
    
    try:
        # Start video processing
        with st.spinner("Analyzing video..."):
            # Send analysis request
            endpoint = "/api/v1/chat" if chat_mode else "/api/v1/analyze"
            response = requests.post(
                f"http://localhost:8001{endpoint}",
                json={
                    "video_path": str(temp_path),
                    "prompt": query if chat_mode else None,
                    "text_queries": [query] if not chat_mode else None,
                    "sample_rate": 30,
                    "max_workers": 4,
                    "use_vila": chat_mode
                },
                stream=True
            )
            
            # Process streaming results
            cap = cv2.VideoCapture(str(temp_path))
            
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
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

def init_rerun():
    """Initialize and connect to Rerun"""
    try:
        # Initialize Rerun with web viewer
        rr.init("video_analytics/frontend", spawn=True)
        rr.connect()
        
        # Configure Rerun viewer URL
        RERUN_VERSION = rr.__version__  # Get current Rerun version
        RRD_URL = "ws://localhost:9000"  # WebSocket URL for Rerun data
        
        # Create viewer iframe URL
        viewer_url = f"https://app.rerun.io/version/{RERUN_VERSION}/index.html?url={RRD_URL}"
        
        # Add iframe to display Rerun viewer
        st.components.v1.html(
            f"""
            <iframe 
                src="{viewer_url}"
                width="100%"
                height="600px"
                frameborder="0"
                style="border: 1px solid #ddd; border-radius: 4px;"
                allow="accelerometer; camera; gyroscope; microphone"
            ></iframe>
            """,
            height=620  # Slightly larger than iframe to account for borders
        )
        return True
    except Exception as e:
        st.error(f"Failed to initialize Rerun viewer: {e}")
        return False

def is_ready():
    """Check if frontend is ready"""
    return True

def main():
    """Main application entry point"""
    st.title("Video Analytics Dashboard")
    
    # Create main container with custom CSS
    st.markdown("""
        <style>
        .main-container {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            width: calc(100vw - 2rem);
            max-width: calc(100vw - 2rem);
            margin: 0 1rem;
            box-sizing: border-box;
        }
        /* Adjust Streamlit container padding */
        .block-container {
            padding: 1rem !important;
            max-width: calc(100vw - 2rem) !important;
        }
        .stApp {
            margin: 0 auto;
        }
        .control-panel {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: calc(100vh - 100px);
            margin: 0.5rem;
            overflow-y: auto;
        }
        .viewer-panel {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: calc(100vh - 100px);
            overflow-y: auto;
        }
        .chat-panel {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: calc(100vh - 100px);
            overflow-y: auto;
        }
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

    # Create responsive columns
    left_col, center_col, right_col = st.columns([1, 2, 1])
    
    # Left column - Controls and Upload
    with left_col:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.header("Controls")
        video_path = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        
        if video_path:
            st.subheader("Analysis Settings")
            with st.expander("Basic Settings", expanded=True):
                sample_rate = st.slider("Sample Rate (frames)", 1, 60, 30)
                max_workers = st.slider("Max Workers", 1, 8, 4)
            
            with st.expander("Processing Options", expanded=True):
                enable_object_detection = st.checkbox("Object Detection", value=True)
                enable_tracking = st.checkbox("Object Tracking", value=True)
                enable_scene_analysis = st.checkbox("Scene Analysis", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Center column - Rerun Visualizer
    with center_col:
        st.markdown('<div class="viewer-panel">', unsafe_allow_html=True)
        st.header("Live Analysis")
        
        if video_path:
            # Initialize Rerun viewer
            if not init_rerun():
                st.warning("Rerun visualization unavailable")
            
            # Add progress indicators
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Frames Processed", "0")
            with col2:
                st.metric("Processing Speed", "0 fps")
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Perform initial scene analysis if video is uploaded
        if video_path:
            with st.spinner("Analyzing scene..."):
                try:
                    # Convert uploaded file to bytes and read with OpenCV
                    file_bytes = video_path.read()
                    temp_video = "temp_video.mp4"
                
                    with open(temp_video, "wb") as f:
                        f.write(file_bytes)
                    
                    # Read first 8 frames
                    frames = []
                    cap = cv2.VideoCapture(temp_video)
                    try:
                        for _ in range(8):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frames.append(frame)
                    finally:
                        cap.release()
                
                    if not frames:
                        raise ValueError("Could not read video file")
                finally:
                    # Clean up temp video file
                    Path(temp_video).unlink(missing_ok=True)

                # Convert frames to base64 for API request
                encoded_frames = []
                for frame in frames:
                    success, buffer = cv2.imencode('.jpg', frame)
                    if not success:
                        raise ValueError("Failed to encode frame")
                    encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))

                # Save first frame as temporary image for scene analysis
                first_frame = frames[0]
                temp_image = "temp_scene_frame.jpg"
                cv2.imwrite(temp_image, first_frame)

                try:
                    response = requests.post(
                        "http://localhost:8001/api/v1/analyze_scene",
                        json={
                            "image_path": temp_image,
                            "context": "Video upload analysis",
                            "stream_type": "uploaded_video"
                        }
                    )

                    if response.status_code != 200:
                        raise ValueError(f"Scene analysis failed with status {response.status_code}: {response.text}")

                    scene_analysis = response.json()
                    
                    # Display scene analysis results
                    st.subheader("Scene Analysis")
                    st.json(scene_analysis.get('scene_analysis', {}))
                    
                    # Display suggested pipeline
                    st.subheader("Suggested Processing Pipeline")
                    for step in scene_analysis.get('suggested_pipeline', []):
                        st.write(f"- {step}")

                except Exception as e:
                    st.warning("Scene analysis failed. Continuing with basic processing.")
                    logging.warning(f"Scene analysis failed: {str(e)}")
                finally:
                    # Clean up temporary image file
                    Path(temp_image).unlink(missing_ok=True)

    # Right column - Chat History
    with right_col:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        st.header("Analysis Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat container with scrolling
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.info("Start analyzing a video to begin the chat interaction")
            else:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input below columns
    if video_path:  # Only show chat input if video is uploaded
        if prompt := st.chat_input("Ask about the video...", key="chat_input"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process video with chat prompt
            process_video(video_path, prompt, chat_mode=True)


def start():
    """Start the frontend server"""
    main()

if __name__ == "__main__":
    start()
