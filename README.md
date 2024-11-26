# Video Analytics Platform

A comprehensive real-time video analytics platform with AI-powered analysis capabilities.

## Project Structure

```
video-analytics/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API endpoint definitions
│   ├── config/
│   │   ├── __init__.py
│   │   ├── aws.py                 # AWS configuration
│   │   └── urls.py                # URL configuration
│   ├── core/
│   │   ├── __init__.py
│   │   └── swarm_agents.py        # Multi-agent analysis system
│   ├── services/
│   │   ├── chat_service.py        # Chat processing service
│   │   ├── edge_detection_service.py
│   │   ├── rag_service.py         # Retrieval Augmented Generation
│   │   ├── scene_service.py       # Scene analysis service
│   ├── utils/
│   │   ├── handlers/
│   │   │   ├── aws_websocket_handler.py
│   │   │   ├── base_handler.py
│   │   │   ├── camera_stream_handler.py
│   │   │   ├── file_list_handler.py
│   │   │   ├── handler_interface.py
│   │   │   ├── message_router.py
│   │   │   ├── progress_handler.py
│   │   │   ├── video_upload_handler.py
│   │   ├── video_streaming/
│   │   │   ├── __init__.py
│   │   │   ├── cv_subscribers.py
│   │   │   ├── stream_manager.py
│   │   │   ├── stream_publisher.py
│   │   │   ├── stream_subscriber.py
│   │   │   └── websocket_publisher.py
│   │   ├── custom_viewer.py
│   │   ├── edge_detection_subscriber.py
│   │   ├── frame_processor.py
│   │   ├── memory_manager.py
│   │   ├── rerun_manager.py
│   │   ├── rerun_processor.py
│   │   ├── socket_handler.py
│   │   ├── video_stream.py
│   │   └── viewer_factory.py
│   ├── __init__.py
│   ├── app.py                     # Main Flask application
│   ├── config.yaml                # Configuration file
│   ├── content_manager.py         # Content/file management
│   ├── requirements.txt           # Python dependencies
│   └── run.py                     # Application entry point
├── frontend/
│   ├── public/
│   │   ├── favicon.ico
│   │   └── index.html
│   ├── src/
│   │   ├── assets/
│   │   │   └── logo.png
│   │   ├── components/
│   │   │   ├── AnalysisControls.js
│   │   │   ├── CameraSelector.js
│   │   │   ├── Chat.js
│   │   │   ├── CustomViewer.js
│   │   │   ├── ErrorBoundary.js
│   │   │   ├── FileList.js
│   │   │   ├── Header.js
│   │   │   ├── RestartControls.js
│   │   │   └── VideoUpload.js
│   │   ├── services/
│   │   │   └── websocket.js       # WebSocket service
│   │   ├── stores/
│   │   │   ├── index.js
│   │   │   └── videoStore.js      # Video state management
│   │   ├── App.js                 # Main React component
│   │   ├── index.js               # Application entry point
│   │   └── theme.js               # Material-UI theme
│   └── package.json               # Node.js dependencies
└── README.md
```

## Features

- Real-time video processing
- Camera stream support
- Video file upload/management
- Edge detection
- Object detection
- Scene analysis
- Chat interface with RAG
- WebSocket communication
- Custom video viewer

## Technology Stack

### Backend
- Python 3.8+
- Flask
- OpenCV
- PyTorch
- Socket.IO
- GPT-4V
- RAG (Retrieval Augmented Generation)

### Frontend
- React.js
- Material-UI
- Socket.IO Client
- WebSocket
- Zustand (State Management)

## Setup

1. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Configure environment:
- Copy `backend/.env.example` to `backend/.env`
- Update configuration in `backend/config.yaml`

4. Start backend server:
```bash
cd backend
python run.py
```

5. Start frontend development server:
```bash
cd frontend
npm start
```

## Architecture

### Video Processing Pipeline
1. Input Sources:
   - Camera streams
   - Uploaded video files
2. Frame Processing:
   - Edge detection
   - Object detection
   - Scene analysis
3. Real-time Visualization:
   - Custom viewer
   - WebSocket streaming
   - Analysis overlays

### Analysis Pipeline
1. Scene Understanding:
   - GPT-4V integration
   - Multi-agent swarm analysis
2. Chat System:
   - RAG-based retrieval
   - Context-aware responses
3. Results Storage:
   - Temporary storage
   - Analysis persistence
   - Automatic cleanup

## Configuration

Key configuration files:
- `backend/config.yaml`: Main configuration
- `backend/.env`: Environment variables
- `frontend/.env`: Frontend configuration

## Development

### Adding New Features
1. Backend:
   - Add routes in `api/routes.py`
   - Create services in `services/`
   - Add handlers in `utils/handlers/`

2. Frontend:
   - Add components in `components/`
   - Update state in `stores/`
   - Configure WebSocket in `services/`

### Testing
- Backend tests in `backend/tests/`
- Frontend tests with Jest/React Testing Library

## License

MIT License
