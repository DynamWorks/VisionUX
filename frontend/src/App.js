import React, { useState, useCallback, useEffect } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import Header from './components/Header';
import Footer from './components/Footer';
import CameraSelector from './components/CameraSelector';
import RerunViewer from './components/RerunViewer';
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

    // Initialize fetch function
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

    // Fetch files on component mount
    useEffect(() => {
        fetchUploadedFiles();
    }, [fetchUploadedFiles]);

    useEffect(() => {
        if (ws) return; // Prevent recreating if ws exists
        
        // Get WebSocket URL from environment with fallbacks
        // Get WebSocket URL with fallbacks
        const wsPort = process.env.REACT_APP_WS_PORT || '8001';
        const wsHost = process.env.REACT_APP_WS_HOST || window.location.hostname;
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = process.env.REACT_APP_WS_URL || `${wsProtocol}//${wsHost}:${wsPort}`;

        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10; // Increased max attempts
        let reconnectTimeout;

        const connectWebSocket = () => {
            if (ws?.readyState === WebSocket.OPEN) {
                console.log('WebSocket already connected');
                return;
            }

            try {
                console.log('Connecting to WebSocket:', wsUrl);
                const websocket = new WebSocket(wsUrl);
                let connectionTimeout;

                // Set connection timeout with jitter
                const baseDelay = 1000;
                const maxDelay = 10000; // Reduced max delay to 10 seconds
                const jitter = Math.random() * 1000; // Add random jitter
                const timeoutDuration = Math.min(
                    baseDelay * Math.pow(1.5, reconnectAttempts) + jitter,
                    maxDelay
                );
                
                connectionTimeout = setTimeout(() => {
                    if (websocket.readyState !== WebSocket.OPEN) {
                        console.log(`Connection attempt timed out after ${timeoutDuration}ms`);
                        websocket.close();
                        // Trigger immediate reconnect
                        clearTimeout(reconnectTimeout);
                        reconnectTimeout = setTimeout(connectWebSocket, 100);
                    }
                }, timeoutDuration);

                // Add connection error handler
                websocket.addEventListener('error', (error) => {
                    console.error('WebSocket connection error:', error);
                    clearTimeout(connectionTimeout);
                });

                websocket.onopen = () => {
                    clearTimeout(connectionTimeout);
                    console.log('WebSocket Connected');
                    reconnectAttempts = 0; // Reset attempts on successful connection
                    setWs(websocket); // Only set ws when connection is established

                    // Send initial connection message and request file list
                    websocket.send(JSON.stringify({ type: 'connection_established' }));
                    websocket.send(JSON.stringify({ type: 'get_uploaded_files' }));
                    console.log('Requested initial file list');
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
                    clearTimeout(connectionTimeout);
                    console.log(`WebSocket Closed: ${event.code} - ${event.reason}`);
                    setWs(null);

                    // Try reconnecting with improved backoff strategy
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        const jitter = Math.random() * 1000;
                        const delay = Math.min(
                            1000 * Math.pow(1.5, reconnectAttempts) + jitter,
                            10000 // Cap at 10 seconds
                        );
                        console.log(`Reconnecting... Attempt ${reconnectAttempts} in ${delay}ms`);
                        reconnectTimeout = setTimeout(connectWebSocket, delay);
                    } else {
                        console.error('Max reconnection attempts reached');
                        // Reset attempts after a longer delay
                        setTimeout(() => {
                            reconnectAttempts = 0;
                            connectWebSocket();
                        }, 30000); // Wait 30 seconds before resetting
                    }
                };

                websocket.onerror = (error) => {
                    console.error('WebSocket Error:', error);
                    // Only close if not already closing/closed
                    if (websocket.readyState !== WebSocket.CLOSING && websocket.readyState !== WebSocket.CLOSED) {
                        websocket.close();
                    }
                };

            } catch (error) {
                console.error('WebSocket connection error:', error);
                // Schedule reconnect on connection error
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                    console.log(`Reconnecting... Attempt ${reconnectAttempts} in ${delay}ms`);
                    reconnectTimeout = setTimeout(connectWebSocket, delay);
                }
            }
        };

        connectWebSocket();

        return () => {
            if (ws) {
                // Set a flag to prevent reconnection attempts
                ws.intentionalClose = true;
                if (ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
            }
            clearTimeout(reconnectTimeout);
            reconnectAttempts = 0;
        };
    }, []);

    const startCamera = async (deviceId) => {
        try {
            // Stop any playing video first
            if (videoFile) {
                setVideoFile(null);
                setIsStreaming(false);
            }
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

            let animationFrameId;
            const sendFrame = () => {
                if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) {
                    cancelAnimationFrame(animationFrameId);
                    return;
                }

                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob(async (blob) => {
                    try {
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
                        
                        // Schedule next frame only after successful send
                        animationFrameId = requestAnimationFrame(sendFrame);
                    } catch (error) {
                        console.error('Error sending frame:', error);
                        setIsStreaming(false);
                        cancelAnimationFrame(animationFrameId);
                    }
                }, 'image/jpeg', 0.85);
            };

            video.onloadedmetadata = () => {
                video.play();
                animationFrameId = requestAnimationFrame(sendFrame);
            };

            // Cleanup function
            return () => {
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                }
                if (video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                }
            };
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    };


    const stopCamera = useCallback(() => {
        if (stream) {
            // Stop all tracks
            stream.getTracks().forEach(track => {
                track.stop();
            });
            setStream(null);
            setIsStreaming(false);
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


    useEffect(() => {
        refreshDevices();
        
        // Only fetch files on initial load
        if (ws && ws.readyState === WebSocket.OPEN) {
            fetchUploadedFiles();
        }

        // Handle WebSocket messages for file list updates
        const handleMessages = (event) => {
            if (!event.data) return;
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'uploaded_files') {
                    console.log('Processing uploaded files:', data.files);
                    if (Array.isArray(data.files)) {
                        const sortedFiles = data.files.sort((a, b) => b.modified - a.modified);
                        setUploadedFiles(sortedFiles);
                    }
                } else if (data.type === 'connection_established') {
                    console.log('Connection established, fetching files');
                    fetchUploadedFiles();
                } else if (data.type === 'upload_complete_ack') {
                    console.log('Upload complete, fetching updated file list');
                    fetchUploadedFiles();
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };

        if (ws) {
            ws.addEventListener('message', handleMessages);
            
            // Request files when WebSocket first connects
            ws.addEventListener('open', () => {
                console.log('WebSocket connected, fetching files');
                fetchUploadedFiles();
            });
        }

        return () => {
            if (ws) {
                ws.removeEventListener('message', handleMessages);
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
                                            URL.revokeObjectURL(videoFile.url);
                                        }
                                        setVideoFile({
                                            file: file,
                                            url: URL.createObjectURL(file)
                                        });
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

                                            // Clean up old URLs
                                            prev.slice(4).forEach(oldFile => {
                                                if (oldFile.url) {
                                                    URL.revokeObjectURL(oldFile.url);
                                                }
                                            });

                                            // Add new file and keep only 5 most recent
                                            const newFile = {
                                                ...file,
                                                url: URL.createObjectURL(file)
                                            };
                                            return [newFile, ...prev.slice(0, 4)];
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
                                                        try {
                                                            // Send reset command to WebSocket
                                                            if (ws && ws.readyState === WebSocket.OPEN) {
                                                                ws.send(JSON.stringify({
                                                                    type: 'reset_rerun'
                                                                }));
                                                            }
                                                            // Reload iframe with delay
                                                            setTimeout(() => {
                                                                const currentSrc = rerunViewer.src;
                                                                rerunViewer.src = '';
                                                                rerunViewer.src = currentSrc;
                                                            }, 1000);
                                                        } catch (error) {
                                                            console.error('Error resetting Rerun viewer:', error);
                                                        }
                                                    }
                                    
                                                    // Close existing WebSocket connection
                                                    if (ws) {
                                                        ws.close();
                                                    }

                                                    // Request updated file list immediately
                                                    fetchUploadedFiles();

                                                    // Reconnect WebSocket after a short delay
                                                    setTimeout(() => {
                                                        const wsPort = process.env.REACT_APP_WS_PORT || '8001';
                                                        const wsHost = process.env.REACT_APP_WS_HOST || 'localhost';
                                                        const wsUrl = `ws://${wsHost}:${wsPort}`;
                                                        
                                                        const newWs = new WebSocket(wsUrl);
                                                        
                                                        newWs.onopen = () => {
                                                            console.log('WebSocket reconnected after upload');
                                                            setWs(newWs);
                                                            // Request fresh file list
                                                            newWs.send(JSON.stringify({
                                                                type: 'get_uploaded_files'
                                                            }));
                                                        };
                                                    }, 1000); // Wait 1 second before reconnecting

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
                                        // Start streaming via API
                                        fetch(`${process.env.REACT_APP_API_URL}/api/v1/stream/start`, {
                                            method: 'POST',
                                            headers: {
                                                'Content-Type': 'application/json',
                                            },
                                            body: JSON.stringify({
                                                filename: file.name
                                            })
                                        })
                                        .then(response => response.json())
                                        .then(data => {
                                            if (data.error) {
                                                throw new Error(data.error);
                                            }
                                            setIsStreaming(true);
                                        })
                                        .catch(error => {
                                            console.error('Error starting stream:', error);
                                            alert(`Failed to start stream: ${error.message}`);
                                        });
                                    } else if (!shouldPlay && isStreaming) {
                                        // Pause streaming
                                        // Pause streaming via API
                                        fetch(`${process.env.REACT_APP_API_URL}/api/v1/stream/pause`, {
                                            method: 'POST',
                                            headers: {
                                                'Content-Type': 'application/json'
                                            }
                                        })
                                        .then(response => response.json())
                                        .then(data => {
                                            if (data.error) {
                                                throw new Error(data.error);
                                            }
                                            setIsStreaming(false);
                                        })
                                        .catch(error => {
                                            console.error('Error pausing stream:', error);
                                            alert(`Failed to pause stream: ${error.message}`);
                                        });
                                    }
                                }}
                                onStop={(file) => {
                                    // Stop streaming via API
                                    fetch(`${process.env.REACT_APP_API_URL}/api/v1/stream/stop`, {
                                        method: 'POST',
                                        headers: {
                                            'Content-Type': 'application/json'
                                        }
                                    })
                                    .then(response => response.json())
                                    .then(data => {
                                        if (data.error) {
                                            throw new Error(data.error);
                                        }
                                        setIsStreaming(false);
                                        setVideoFile(null);
                                    })
                                    .catch(error => {
                                        console.error('Error stopping stream:', error);
                                        alert(`Failed to stop stream: ${error.message}`);
                                    });
                                }}
                            />
                        </Box>
                        <Box sx={{ width: '70%', display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <Box sx={{ 
                                flex: 1,
                                minHeight: '600px',
                                bgcolor: '#1a1a1a',
                                borderRadius: '8px',
                                overflow: 'hidden'
                            }}>
                                <RerunViewer />
                            </Box>
                        </Box>
                    </Box>
                </Container>
                <Footer />
            </Box>
        </Router>
    );
}

export default App;
