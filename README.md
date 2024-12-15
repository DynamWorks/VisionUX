# Video Analytics Platform

A comprehensive real-time video analytics platform with AI-powered analysis capabilities.

## Project Structure

```
vision-llm/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API endpoint definitions
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent_framework.py     # Agent orchestration
│   │   └── analysis_tools.py      # Analysis tools
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py        # Chat processing service
│   │   ├── cv_service.py          # Computer vision service
│   │   ├── rag_service.py         # Retrieval Augmented Generation
│   │   └── scene_service.py       # Scene analysis service
│   ├── utils/
│   │   ├── handlers/
│   │   │   ├── __init__.py
│   │   │   ├── base_handler.py
│   │   │   ├── camera_stream_handler.py
│   │   │   ├── file_list_handler.py
│   │   │   ├── handler_interface.py
│   │   │   ├── message_router.py
│   │   │   ├── progress_handler.py
│   │   │   └── video_upload_handler.py
│   │   ├── video_streaming/
│   │   │   ├── __init__.py
│   │   │   ├── stream_manager.py
│   │   │   ├── stream_publisher.py
│   │   │   ├── stream_subscriber.py
│   │   │   └── websocket_publisher.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── env_validator.py
│   │   └── socket_handler.py
│   ├── __init__.py
│   ├── app.py                     # Main Flask application
│   ├── config.yaml                # Configuration file
│   ├── content_manager.py         # Content/file management
│   └── run.py                     # Application entry point
├── frontend/
│   ├── public/
│   │   ├── favicon.ico
│   │   ├── favicon.png
│   │   └── index.html
│   ├── src/
│   │   ├── assets/
│   │   │   ├── favicon.ico
│   │   │   ├── favicon.png
│   │   │   └── logo.png
│   │   ├── components/
│   │   │   ├── AnalysisControls.js
│   │   │   ├── CameraSelector.js
│   │   │   ├── Chat.js
│   │   │   ├── CustomViewer.js
│   │   │   ├── ErrorBoundary.js
│   │   │   ├── FileList.js
│   │   │   ├── Footer.js
│   │   │   ├── Header.js
│   │   │   ├── InputSelector.js
│   │   │   ├── StreamRenderer.js
│   │   │   ├── VideoPlayer.js
│   │   │   └── VideoUpload.js
│   │   ├── hooks/
│   │   │   ├── useChat.js
│   │   │   └── useVideo.js
│   │   ├── services/
│   │   │   └── websocket.js       # WebSocket service
│   │   ├── store/
│   │   │   └── index.js           # Zustand store
│   │   ├── App.js                 # Main React component
│   │   ├── index.css              # Global styles
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

### Environment Variables

The following environment variables need to be configured in `.env`:

Required:
- `API_HOST`: API server bind address
- `API_PORT`: API server port
- `OPENAI_API_KEY`: Your OpenAI API key
- `GEMINI_API_KEY`: Your Gemini API key

Optional:
- `API_DEBUG`: Enable API debug mode (default: false)
- `API_CORS_ORIGINS`: CORS allowed origins (default: *)
- `WS_HOST`: WebSocket server host (default: localhost)
- `WS_PORT`: WebSocket server port (default: 8000)
- `WS_DEBUG`: Enable WebSocket debug mode (default: false)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4o-mini)
- `GEMINI_MODEL`: Gemini model to use (default: gemini-1.5-flash)
- `LOG_LEVEL`: Logging level (default: INFO)

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
