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

def main():
    """Main application entry point"""
    st.set_page_config(page_title="Video Analytics Dashboard")
    
    # Initialize Rerun
    if not init_rerun():
        st.warning("Continuing without Rerun visualization...")

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

def is_ready():
    """Check if frontend is ready"""
    return True

def start():
    """Start the frontend server"""
    main()

if __name__ == "__main__":
    start()
    # Initialize Rerun
    if not init_rerun():
        st.warning("Continuing without Rerun visualization...")
        
    st.title("Video Analytics Dashboard")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        video_path = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        
        if video_path:
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            st.header("Chat Analysis")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask about the video..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Process video with chat prompt
                process_video(video_path, prompt, chat_mode=True)

def check_server_status(url: str = "http://localhost:8001") -> bool:
    """Check if the API server is running"""
    try:
        response = requests.get(f"{url}/api/health")
        return response.status_code == 200
    except:
        return False

def process_video(video_path, query, chat_mode=False):
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
    
    # Start video processing
    try:
        with st.spinner("Analyzing video..."):
            # Send analysis request
            endpoint = "/api/chat" if chat_mode else "/api/analyze"
            response = requests.post(
                f"http://localhost:8001{endpoint}",
                json={
                    "video_path": str(temp_path),
                    "prompt": query if chat_mode else None,
                    "text_queries": [query] if not chat_mode else None,
                    "sample_rate": 30,
                    "max_workers": 4,
                    "use_vila": chat_mode  # Enable VILA for chat mode
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
            except Exception as e:
                st.warning(f"Could not display frame: {str(e)}")
                continue
            
            # Get analysis results
            try:
                for line in response.iter_lines():
                    if line:
                        result = json.loads(line.decode().replace('data: ', ''))
                        
                        if chat_mode:
                            # Add assistant response to chat
                            if "response" in result:
                                with st.chat_message("assistant"):
                                    st.markdown(result["response"])
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": result["response"]
                                })
                        else:
                            # Display analysis results
                            results_placeholder.json(result)
                        
                        # Log detections to Rerun
                        try:
                            for det in result.get('detections', {}).get('segments', []):
                                bbox = det.get('bbox', [0,0,0,0])
                                rr.log("detections", 
                                      rr.Boxes2D(
                                          boxes=[[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]],
                                          labels=[f"{det.get('class', '')}: {det.get('confidence', 0):.2f}"]
                                      ))
                        except Exception as e:
                            st.warning(f"Could not log detection: {str(e)}")
                            continue
                            
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
