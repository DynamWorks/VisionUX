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
        if (!process.env.REACT_APP_WS_URL) {
            console.error('WebSocket URL is not defined in environment variables');
            return;
        }

        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        let reconnectTimeout;

        const connectWebSocket = () => {
            try {
                const wsUrl = process.env.REACT_APP_WS_URL;
                console.log('Connecting to WebSocket:', wsUrl);
                const websocket = new WebSocket(wsUrl);

                websocket.onopen = () => {
                    console.log('WebSocket Connected');
                    reconnectAttempts = 0; // Reset attempts on successful connection
                    setWs(websocket); // Only set ws when connection is established

                    // Send initial connection message
                    websocket.send(JSON.stringify({ type: 'connection_established' }));
                };

                websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'ping') {
                            websocket.send('pong');
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
                    canvas.toBlob(
                        (blob) => {
                            try {
                                // Reset Rerun if this is the first frame
                                if (!window.framesSent) {
                                    ws.send(JSON.stringify({ type: 'reset_rerun' }));
                                    window.framesSent = true;
                                }
                                // Send frame type indicator first
                                ws.send(JSON.stringify({ type: 'camera_frame' }));
                                // Then send the actual frame data
                                ws.send(blob);
                            } catch (error) {
                                console.error('Error sending frame:', error);
                            }
                        },
                        'image/jpeg',
                        0.8
                    );
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
    useEffect(() => {
        refreshDevices();
        
        // Fetch uploaded files on mount
        const fetchUploadedFiles = () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const handleMessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'uploaded_files') {
                            const files = data.files.map(file => new File(
                                [new Blob()], // Empty blob as placeholder
                                file.name,
                                {
                                    type: 'video/mp4',
                                    lastModified: file.modified * 1000
                                }
                            ));
                            setUploadedFiles(files);
                        }
                    } catch (error) {
                        console.error('Error processing uploaded files:', error);
                    }
                };
                
                // Add message listener
                ws.addEventListener('message', handleMessage);
                
                // Request file list
                ws.send(JSON.stringify({
                    type: 'get_uploaded_files'
                }));
                
                // Cleanup listener on unmount
                return () => ws.removeEventListener('message', handleMessage);
            }
        };

        // Call fetchUploadedFiles when WebSocket is ready
        if (ws && ws.readyState === WebSocket.OPEN) {
            fetchUploadedFiles();
        }
        
        // Log mount for debugging
        console.log('App mounted');
        return () => console.log('App unmounted');
    }, [refreshDevices, ws]);

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

                                        // Send video file through WebSocket
                                        if (ws && ws.readyState === WebSocket.OPEN) {
                                            const reader = new FileReader();

                                            // Send metadata first
                                            ws.send(JSON.stringify({
                                                type: 'video_upload_start',
                                                filename: file.name,
                                                size: file.size,
                                                contentType: file.type
                                            }));

                                            reader.onload = async () => {
                                                try {
                                                    if (!ws || ws.readyState !== WebSocket.OPEN) {
                                                        throw new Error('WebSocket connection lost');
                                                    }

                                                    // Set binary type to blob for better memory handling
                                                    ws.binaryType = 'blob';

                                                    // Split file into chunks
                                                    const chunkSize = 1024 * 1024; // 1MB chunks
                                                    const fileData = reader.result;
                                                    const chunks = [];

                                                    for (let i = 0; i < fileData.byteLength; i += chunkSize) {
                                                        chunks.push(fileData.slice(i, i + chunkSize));
                                                    }

                                                    // Create promise to wait for upload completion
                                                    const uploadComplete = new Promise((resolve, reject) => {
                                                        let timeoutId;

                                                        const messageHandler = (event) => {
                                                            try {
                                                                const response = JSON.parse(event.data);
                                                                console.log('Received upload response:', response);

                                                                if (response.type === 'upload_complete_ack') {
                                                                    clearTimeout(timeoutId);
                                                                    ws.removeEventListener('message', messageHandler);
                                                                    resolve(response);
                                                                } else if (response.type === 'upload_error') {
                                                                    clearTimeout(timeoutId);
                                                                    ws.removeEventListener('message', messageHandler);
                                                                    reject(new Error(response.error));
                                                                }
                                                            } catch (error) {
                                                                console.warn('Non-JSON message received:', event.data);
                                                            }
                                                        };

                                                        // Set timeout for upload confirmation
                                                        timeoutId = setTimeout(() => {
                                                            ws.removeEventListener('message', messageHandler);
                                                            reject(new Error('Upload confirmation timeout'));
                                                        }, 30000); // 30 second timeout

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

                                                    // Reset Rerun viewer
                                                    const rerunViewer = document.querySelector('iframe');
                                                    if (rerunViewer) {
                                                        rerunViewer.src = rerunViewer.src;
                                                    }
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
                                                alert('Failed to read file');
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
