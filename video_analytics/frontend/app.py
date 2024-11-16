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

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Rerun
RERUN_PORT = 9000  # Port for Rerun web viewer

# Set page config as first Streamlit command
st.set_page_config(page_title="Video Analytics Dashboard")

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

def main():
    """Main application entry point"""
    # Initialize Rerun
    if not init_rerun():
        st.warning("Continuing without Rerun visualization...")

def init_rerun():
    """Initialize and connect to Rerun"""
    try:
        # Initialize Rerun with web viewer
        rr.init("video_analytics/frontend", spawn=True)
        rr.connect()
        
        # Get the viewer URL from Rerun
        viewer_url = "http://localhost:6006"  # Default Rerun web viewer port
        
        # Add iframe to display Rerun viewer
        st.components.v1.iframe(
            viewer_url,
            height=600,
            scrolling=True
        )
        return True
    except Exception as e:
        st.error(f"Failed to initialize Rerun viewer: {e}")
        return False

def is_ready():
    """Check if frontend is ready"""
    return True

def start():
    """Start the frontend server"""
    main()

if __name__ == "__main__":
    start()
    st.title("Video Analytics Dashboard")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        video_path = st.file_uploader("Upload Video", type=['mp4', 'avi'])

    if video_path:
        # Perform initial scene analysis
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
                for _ in range(8):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                
                if not frames:
                    raise ValueError("Could not read video file")
                
                # Clean up temp video file
                Path(temp_video).unlink()

                # Convert frames to base64 for API request
                encoded_frames = []
                for frame in frames:
                    success, buffer = cv2.imencode('.jpg', frame)
                    if not success:
                        raise ValueError("Failed to encode frame")
                    encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))

                response = requests.post(
                    "http://localhost:8001/api/v1/analyze_scene",
                    json={
                        "frames": encoded_frames,
                        "context": "Video upload analysis",
                        "stream_type": "uploaded_video",
                        "frame_count": len(frames)
                    }
                )

                if response.status_code != 200:
                    raise ValueError(f"Scene analysis failed with status {response.status_code}: {response.text}")
                if response.status_code == 200:
                    scene_analysis = response.json()
                    
                    # Display scene analysis results
                    st.subheader("Scene Analysis")
                    st.json(scene_analysis.get('scene_analysis', {}))
                    
                    # Display suggested pipeline
                    st.subheader("Suggested Processing Pipeline")
                    for step in scene_analysis.get('suggested_pipeline', []):
                        st.write(f"- {step}")
                else:
                    st.warning("Scene analysis failed. Continuing with basic processing.")
            except Exception as e:
                st.error(f"Scene analysis error: {str(e)}")
                logging.error(f"Scene analysis failed: {str(e)}", exc_info=True)
                
                # Cleanup temp file
                try:
                    Path(temp_video).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    logging.error(f"Failed to cleanup temp file: {cleanup_error}")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        st.header("Chat Analysis")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input (outside sidebar)
        if prompt := st.chat_input("Ask about the video..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process video with chat prompt
            process_video(video_path, prompt, chat_mode=True)


if __name__ == "__main__":
    main()
