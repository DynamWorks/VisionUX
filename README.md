# VisionUX

VisionUX is a video analytics platform powered by an Agentic RAG framework incorporating OpenAI GPT-4o-mini and Google Gemini-1.5-Flash models. This proof-of-concept application combines computer vision, AI analysis, and interactive visualization for video understanding through an agent-based approach to content processing and user interaction.

[![YouTube](http://i.ytimg.com/vi/jx_z7j4_lFQ/hqdefault.jpg)](https://www.youtube.com/watch?v=jx_z7j4_lFQ)

## Prerequisites

- Python 3.11
- Node.js 20.17.0
- npm 10.8.2
- Conda (for environment management)
- OpenAI API key
- Google Gemini API key

## Installation

### Backend Setup

1. Create and activate a Python virtual environment:

```bash
conda create --name visionux python=3.11
conda activate visionux
cd backend
pip install -r requirements.txt
```

2. Copy and configure environment variables:

```bash
cp .env.example .env
```

Required environment variables:

```bash
API_HOST=localhost
API_PORT=8000
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### Frontend Setup

1. In a new terminal (without Python virtual environment):

```bash
conda deactivate
cd frontend
npm install
```

2. Copy and configure environment variables:

```bash
cp .env.example .env
```

Configure with these settings:

```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=http://localhost:8000
REACT_APP_API_VERSION=/api/v1
REACT_APP_WS_PORT=8000
REACT_APP_WS_HOST=localhost
REACT_APP_STREAM_PORT=8001
REACT_APP_STREAM_WS_URL=http://localhost:8001
```

## Running the Application

1. Start the backend server:

```bash
cd backend
python run.py --config config.yaml
```

2. In a separate terminal, start the frontend:

```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## Usage Guidelines

### Video Requirements

- Recommended video length: 10-20 seconds for optimal performance
- Primary supported format: MP4
- Other formats (AVI, MOV, WEBM) may work but might have compatibility issues
- Maximum file size: 100MB

### Current Limitations

- This is a proof-of-concept implementation
- Camera functionality is disabled (planned for future releases)
- Video processing is optimized for short clips
- Users should test thoroughly and report any issues

## Features

- Video file upload and management
- AI-powered scene analysis
- Object detection
- Edge detection
- Chat interface with RAG (Retrieval Augmented Generation)
- Analysis results visualization

## Support

For support questions, please open an issue in the repository.

## License

Licensing Terms for VisionUX

Commercial License

If you intend to use VisionUX to develop commercial sites, themes, projects, or applications, the Commercial License is the appropriate option. This license allows you to keep your source code proprietary.

For inquiries regarding the Commercial License, please contact us at contact@dynamworks.com.

Open Source License

VisionUX is also available under the GNU Affero General Public License v3 (GNU AGPL v3).
If you are developing an open-source application under a license compatible with the GNU AGPL v3, you may use VisionUX under the terms of this license.

For more information on the GNU AGPL v3, please visit [GNU AGPL v3 Overview](https://www.gnu.org/licenses/agpl-3.0.en.html).
