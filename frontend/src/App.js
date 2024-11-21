import React, { useState, useCallback, useEffect } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import Header from './components/Header';
import Footer from './components/Footer';
import CameraSelector from './components/CameraSelector';
import RerunViewer from './components/RerunViewer';
import VideoDisplay from './components/VideoDisplay';
import VideoUpload from './components/VideoUpload';
import InputSelector from './components/InputSelector';
import FileList from './components/FileList';

function App() {
    const [inputType, setInputType] = useState('camera');
    const [devices, setDevices] = useState([]);
    const [selectedDevice, setSelectedDevice] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [stream, setStream] = useState(null);
    const [videoFile, setVideoFile] = useState(null);
    const [uploadedFiles, setUploadedFiles] = useState([]);

    const [ws, setWs] = useState(null);

    useEffect(() => {
        const wsPort = process.env.REACT_APP_WS_PORT || '8001';
        const wsHost = process.env.REACT_APP_WS_HOST || 'localhost';
        const wsUrl = `ws://${wsHost}:${wsPort}`;

        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        let reconnectTimeout;
        let isFirstConnection = true;

        const connectWebSocket = () => {
            try {
                console.log('Connecting to WebSocket:', wsUrl);
                const websocket = new WebSocket(wsUrl);

                websocket.onopen = () => {
                    console.log('WebSocket Connected');
                    reconnectAttempts = 0; // Reset attempts on successful connection
                    setWs(websocket); // Only set ws when connection is established

                    // Send initial connection message
                    websocket.send(JSON.stringify({ type: 'connection_established' }));
                    
                    // Immediately request file list
                    console.log('Requesting initial file list...');
                    websocket.send(JSON.stringify({ type: 'get_uploaded_files' }));
                    
                    // Request file list on initial connection and reconnects
                    websocket.send(JSON.stringify({ type: 'get_uploaded_files' }));
                    console.log('Requested initial file list');
                    
                    if (isFirstConnection) {
                        isFirstConnection = false;
                    }
                };

                websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        console.log('WebSocket message received:', data);
                        
                        if (data.type === 'ping') {
                            websocket.send('pong');
                        } else if (data.type === 'uploaded_files') {
                            console.log('Received file list:', data.files);
                            setUploadedFiles(data.files || []);
                        }
                    } catch (error) {
                        console.error('Error processing WebSocket message:', error);
                    }
                };

                websocket.onclose = (event) => {
                    console.log(`WebSocket Closed: ${event.code} - ${event.reason}`);
                    if (!event.wasClean) {
                        if (reconnectAttempts < maxReconnectAttempts) {
                            reconnectAttempts++;
                            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                            console.log(`Reconnecting... Attempt ${reconnectAttempts} in ${delay}ms`);
                            reconnectTimeout = setTimeout(connectWebSocket, delay);
                        } else {
                            console.error('Max reconnection attempts reached');
                        }
                    }
                };

                websocket.onerror = (error) => {
                    console.error('WebSocket Error:', error);
                    // Attempt to reconnect on error
                    if (reconnectAttempts < maxReconnectAttempts) {
                        console.log('Attempting to reconnect due to error...');
                        setTimeout(connectWebSocket, 3000);
                    }
                };

                setWs(websocket);
            } catch (error) {
                console.error('WebSocket connection error:', error);
            }
        };

        connectWebSocket();

        return () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            clearTimeout(reconnectTimeout);
        };
    }, []);

    const startCamera = async (deviceId) => {
        try {
            // Stop any playing video first
            if (videoFile) {
                setVideoFile(null);
                setIsStreaming(false);
            }

            // // Reset Rerun video topic before starting camera
            // if (ws && ws.readyState === WebSocket.OPEN) {
            //     ws.send(JSON.stringify({
            //         type: 'reset_rerun'
            //     }));
            // }

            const constraints = {
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            setStream(mediaStream);
            setIsStreaming(true);

            // Set up video frame capture and sending
            const video = document.createElement('video');
            video.srcObject = mediaStream;
            video.play();

            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 1280;
            canvas.height = 720;

            const sendFrame = () => {
                if (isStreaming && ws && ws.readyState === WebSocket.OPEN) {
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob(async (blob) => {
                        if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;
                        
                        try {
                            // Convert blob to array buffer for more efficient transfer
                            const arrayBuffer = await blob.arrayBuffer();
                            
                            // Send frame type indicator first
                            ws.send(JSON.stringify({ 
                                type: 'camera_frame',
                                width: canvas.width,
                                height: canvas.height,
                                timestamp: Date.now()
                            }));
                            
                            // Send the frame data as binary
                            ws.send(arrayBuffer);
                        } catch (error) {
                            console.error('Error sending frame:', error);
                            setIsStreaming(false);
                        }
                    }, 'image/jpeg', 0.85);
                }
                if (isStreaming) {
                    requestAnimationFrame(sendFrame);
                }
            };

            video.onloadedmetadata = () => {
                sendFrame();
            };
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    };

    const [isPaused, setIsPaused] = useState(false);

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
            setIsStreaming(false);
            setIsPaused(false);
            // Reset frame counter
            window.framesSent = false;
            // Reset Rerun when stopping camera
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'reset_rerun' }));
            }
        }
    }, [stream]);

    const pauseCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => {
                track.enabled = false;
            });
            setIsPaused(true);
        }
    }, [stream]);

    const resumeCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => {
                track.enabled = true;
            });
            setIsPaused(false);
        }
    }, [stream]);

    const refreshDevices = useCallback(async () => {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            setDevices(videoDevices);
        } catch (error) {
            console.error('Error enumerating devices:', error);
        }
    }, []);

    // Initialize devices on mount
    const fetchUploadedFiles = useCallback(async () => {
        try {
            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/files/list`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log('Received file list:', data.files);
            setUploadedFiles(data.files || []);
        } catch (error) {
            console.error('Error fetching files:', error);
        }
    }, []);

    useEffect(() => {
        refreshDevices();
        fetchUploadedFiles(); // Initial fetch

        // Handle WebSocket messages for file list updates
        const handleWebSocketMessages = (event) => {
            if (!event.data) return;
            try {
                const data = JSON.parse(event.data);
                console.log('Received WebSocket message:', data);
                
                if (data.type === 'uploaded_files') {
                    console.log('Processing uploaded files:', data.files);
                    if (Array.isArray(data.files)) {
                        // Sort files by last modified date
                        const sortedFiles = data.files.sort((a, b) => b.modified - a.modified);
                        const files = sortedFiles.map(file => ({
                            name: file.name,
                            size: file.size,
                            lastModified: file.modified * 1000,
                            type: 'video/mp4',
                            path: file.path
                        }));
                        console.log('Updating file list with:', files);
                        setUploadedFiles(prev => {
                            // Only update if the file list has changed
                            const currentPaths = prev.map(f => f.path);
                            const newPaths = files.map(f => f.path);
                            if (JSON.stringify(currentPaths) !== JSON.stringify(newPaths)) {
                                return files;
                            }
                            return prev;
                        });
                    } else {
                        console.warn('Received invalid file list format:', data.files);
                    }
                } else if (data.type === 'error') {
                    console.error('Server error:', data.error);
                } else if (data.type === 'upload_complete_ack') {
                    console.log('Upload complete, refreshing file list');
                    fetchUploadedFiles();
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };

        const handleFileListMessages = (event) => {
            if (!event.data) return;
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'uploaded_files') {
                    console.log('Processing uploaded files:', data.files);
                    if (Array.isArray(data.files)) {
                        const sortedFiles = data.files.sort((a, b) => b.modified - a.modified);
                        setUploadedFiles(sortedFiles);
                    }
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };

        if (ws) {
            ws.addEventListener('message', handleFileListMessages);
            
            // Set up connection and file list request handlers
            const handleConnection = () => {
                if (ws.readyState === WebSocket.OPEN) {
                    console.log('WebSocket ready, fetching files');
                    fetchUploadedFiles();
                }
            };

            // Request immediately if already open
            handleConnection();

            // Set up listener for when connection opens
            ws.addEventListener('open', handleConnection);

            // Also request files after successful connection message
            ws.addEventListener('message', (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'connection_established') {
                        console.log('Connection established, fetching files');
                        fetchUploadedFiles();
                    }
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            });
        }

        return () => {
            if (ws) {
                ws.removeEventListener('message', handleFileListMessages);
            }
        };
    }, [refreshDevices, ws, fetchUploadedFiles]);

    return (
        <Router>
            <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
                <Header />
                <Container maxWidth="xl" sx={{ flex: 1, py: 3 }}>
                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <Box sx={{ width: '30%' }}>
                            <InputSelector
                                inputType={inputType}
                                setInputType={setInputType}
                            />
                            {inputType === 'camera' ? (
                                <CameraSelector
                                    devices={devices}
                                    selectedDevice={selectedDevice}
                                    setSelectedDevice={setSelectedDevice}
                                    isStreaming={isStreaming}
                                    startCamera={startCamera}
                                    stopCamera={stopCamera}
                                    refreshDevices={refreshDevices}
                                    ws={ws}
                                />
                            ) : (
                                <VideoUpload
                                    onUpload={(file) => {
                                        if (isStreaming) {
                                            stopCamera();
                                        }
                                        // Clear previous video file and update with new one
                                        if (videoFile) {
                                            URL.revokeObjectURL(URL.createObjectURL(videoFile));
                                        }
                                        setVideoFile(file);
                                        // Update uploaded files list, checking for duplicates
                                        setUploadedFiles(prev => {
                                            // Check if file already exists
                                            const fileExists = prev.some(existingFile =>
                                                existingFile.name === file.name
                                            );

                                            if (fileExists) {
                                                console.log(`File ${file.name} already exists in the list`);
                                                return prev;
                                            }

                                            // Add new file and keep only 5 most recent
                                            const newFiles = [file, ...prev.slice(0, 4)];
                                            // Clean up old URLs
                                            prev.slice(5).forEach(oldFile => {
                                                URL.revokeObjectURL(URL.createObjectURL(oldFile));
                                            });
                                            return newFiles;
                                        });

                                        // Upload file via API
                                        console.log('Starting file upload via API...');
                                        const formData = new FormData();
                                        formData.append('file', file);

                                        // Wrap upload in async function
                                        (async () => {
                                            try {
                                                const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/upload`, {
                                                    method: 'POST',
                                                    body: formData
                                                });

                                                if (!response.ok) {
                                                    throw new Error(`Upload failed: ${response.statusText}`);
                                                }

                                                const result = await response.json();
                                                console.log('Upload successful:', result);
                                            } catch (error) {
                                                console.error('Upload failed:', error);
                                                alert(`Upload failed: ${error.message}`);
                                            }
                                        })();

                                        // Notify WebSocket about upload
                                        if (ws && ws.readyState === WebSocket.OPEN) {
                                            ws.send(JSON.stringify({
                                                type: 'video_upload_complete',
                                                filename: file.name
                                            }));

                                            // Create FileReader instance
                                            const reader = new FileReader();
                                            reader.onload = async (event) => {
                                                try {
                                                    if (!ws || ws.readyState !== WebSocket.OPEN) {
                                                        throw new Error('WebSocket connection lost');
                                                    }

                                                    // Set binary type to arraybuffer for consistent handling
                                                    ws.binaryType = 'arraybuffer';

                                                    // Split file into chunks
                                                    const chunkSize = 1024 * 1024; // 1MB chunks
                                                    const fileData = event.target.result;
                                                    const chunks = [];

                                                    for (let i = 0; i < fileData.byteLength; i += chunkSize) {
                                                        chunks.push(fileData.slice(i, i + chunkSize));
                                                    }

                                                    // Create promise to wait for upload completion
                                                    const uploadComplete = new Promise((resolve, reject) => {
                                                        let timeoutId;
                                                        let progressTimeout;
                                                        let lastProgressTime = Date.now();

                                                        const messageHandler = (event) => {
                                                            try {
                                                                const response = JSON.parse(event.data);
                                                                console.log('Received upload response:', response);

                                                                if (response.type === 'upload_complete_ack') {
                                                                    clearTimeout(timeoutId);
                                                                    clearTimeout(progressTimeout);
                                                                    ws.removeEventListener('message', messageHandler);
                                                                    resolve(response);
                                                                } else if (response.type === 'upload_error') {
                                                                    clearTimeout(timeoutId);
                                                                    clearTimeout(progressTimeout);
                                                                    ws.removeEventListener('message', messageHandler);
                                                                    reject(new Error(response.error));
                                                                } else if (response.type === 'upload_progress') {
                                                                    // Reset progress timeout on any progress update
                                                                    lastProgressTime = Date.now();
                                                                }
                                                            } catch (error) {
                                                                console.warn('Non-JSON message received:', event.data);
                                                            }
                                                        };

                                                        // Set timeout for overall upload
                                                        timeoutId = setTimeout(() => {
                                                            ws.removeEventListener('message', messageHandler);
                                                            reject(new Error('Upload timed out after 2 minutes'));
                                                        }, 120000); // 2 minute total timeout

                                                        // Set timeout for progress updates
                                                        const checkProgress = () => {
                                                            const timeSinceLastProgress = Date.now() - lastProgressTime;
                                                            if (timeSinceLastProgress > 10000) { // 10 seconds
                                                                clearTimeout(timeoutId);
                                                                ws.removeEventListener('message', messageHandler);
                                                                reject(new Error('Upload stalled - no progress updates'));
                                                            } else {
                                                                progressTimeout = setTimeout(checkProgress, 2000); // Check every 2 seconds
                                                            }
                                                        };
                                                        progressTimeout = setTimeout(checkProgress, 2000);

                                                        ws.addEventListener('message', messageHandler);
                                                    });

                                                    // Send chunks with progress tracking
                                                    for (let i = 0; i < chunks.length; i++) {
                                                        if (ws.readyState !== WebSocket.OPEN) {
                                                            throw new Error('WebSocket connection lost during upload');
                                                        }

                                                        // Send chunk
                                                        ws.send(chunks[i]);

                                                        // Send progress update
                                                        ws.send(JSON.stringify({
                                                            type: 'upload_progress',
                                                            chunk: i + 1,
                                                            totalChunks: chunks.length,
                                                            progress: Math.round(((i + 1) / chunks.length) * 100)
                                                        }));

                                                        // Small delay between chunks
                                                        await new Promise(resolve => setTimeout(resolve, 100));
                                                    }

                                                    // Send upload complete notification
                                                    ws.send(JSON.stringify({
                                                        type: 'video_upload_complete',
                                                        filename: file.name,
                                                        totalSize: file.size
                                                    }));

                                                    console.log('Waiting for upload confirmation...');
                                                    await uploadComplete;
                                                    console.log('Upload completed successfully');

                                                    // Reset Rerun viewer and request file list
                                                    const rerunViewer = document.querySelector('iframe');
                                                    if (rerunViewer) {
                                                        rerunViewer.src = rerunViewer.src;
                                                    }
                                    
                                                    // Request updated file list
                                                    fetchUploadedFiles();
                                    
                                                    // Set up retry mechanism for file list
                                                    let retryCount = 0;
                                                    const maxRetries = 3;
                                                    const retryInterval = setInterval(() => {
                                                        if (retryCount < maxRetries) {
                                                            fetchUploadedFiles();
                                                            retryCount++;
                                                        } else {
                                                            clearInterval(retryInterval);
                                                        }
                                                    }, 1000);
                                                    // Reset Rerun after successful upload
                                                    // ws.send(JSON.stringify({
                                                    //     type: 'reset_rerun'
                                                    // }));
                                                } catch (error) {
                                                    console.error('Upload failed:', error);
                                                    alert(`Upload failed: ${error.message}`);
                                                }
                                            };

                                            reader.onerror = (error) => {
                                                console.error('Error reading file:', error);
                                                alert(`Failed to read file: ${error.target.error.message || 'Unknown error'}`);
                                            };

                                            // Read the entire file as ArrayBuffer
                                            reader.readAsArrayBuffer(file);

                                            // Handle WebSocket messages
                                            const messageHandler = (event) => {
                                                try {
                                                    const response = JSON.parse(event.data);
                                                    console.log('Received WebSocket response:', response);

                                                    if (response.type === 'upload_error') {
                                                        throw new Error(response.error);
                                                    }
                                                } catch (error) {
                                                    console.error('Error processing WebSocket message:', error);
                                                    alert(`Upload error: ${error.message}`);
                                                }
                                            };

                                            ws.addEventListener('message', messageHandler);

                                            // Cleanup
                                            return () => {
                                                ws.removeEventListener('message', messageHandler);
                                            };
                                        }
                                    }}
                                />
                            )}
                            <FileList
                                files={uploadedFiles}
                                activeFile={videoFile}
                                isPlaying={isStreaming}
                                onFileSelect={(file) => {
                                    setVideoFile(file);
                                }}
                                onPlayPause={(file, shouldPlay) => {
                                    if (shouldPlay && !isStreaming) {
                                        // Start streaming the selected file
                                        if (ws && ws.readyState === WebSocket.OPEN) {
                                            ws.send(JSON.stringify({
                                                type: 'start_video_stream',
                                                filename: file.name
                                            }));
                                        }
                                        setIsStreaming(true);
                                    } else if (!shouldPlay && isStreaming) {
                                        // Pause streaming
                                        if (ws && ws.readyState === WebSocket.OPEN) {
                                            ws.send(JSON.stringify({
                                                type: 'pause_video_stream'
                                            }));
                                        }
                                        setIsStreaming(false);
                                    }
                                }}
                                onStop={(file) => {
                                    if (ws && ws.readyState === WebSocket.OPEN) {
                                        ws.send(JSON.stringify({
                                            type: 'stop_video_stream'
                                        }));
                                    }
                                    setIsStreaming(false);
                                    setVideoFile(null);
                                }}
                            />
                        </Box>
                        <Box sx={{ width: '70%', display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <RerunViewer />
                        </Box>
                    </Box>
                </Container>
                <Footer />
            </Box>
        </Router>
    );
}

export default App;
