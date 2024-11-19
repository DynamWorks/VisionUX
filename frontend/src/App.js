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

function App() {
    const [inputType, setInputType] = useState('camera');
    const [devices, setDevices] = useState([]);
    const [selectedDevice, setSelectedDevice] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [stream, setStream] = useState(null);
    const [videoFile, setVideoFile] = useState(null);

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
                const websocket = new WebSocket(process.env.REACT_APP_WS_URL);
                
                websocket.onopen = () => {
                    console.log('WebSocket Connected');
                    reconnectAttempts = 0; // Reset attempts on successful connection
                };
                
                websocket.onclose = () => {
                    console.log('WebSocket Closed');
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        console.log(`Reconnecting... Attempt ${reconnectAttempts}`);
                        reconnectTimeout = setTimeout(connectWebSocket, 3000);
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
                            // Send frame type indicator first
                            ws.send(JSON.stringify({ type: 'camera_frame' }));
                            // Then send the actual frame data
                            ws.send(blob);
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
        // Log mount for debugging
        console.log('App mounted');
        return () => console.log('App unmounted');
    }, [refreshDevices]);

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
                                        setVideoFile(file);
                                    
                                        // Send video file through WebSocket with chunking
                                        if (ws && ws.readyState === WebSocket.OPEN) {
                                            const CHUNK_SIZE = 1024 * 1024; // 1MB chunks
                                            const reader = new FileReader();
                                            let offset = 0;
                                            
                                            // Send metadata first
                                            ws.send(JSON.stringify({ 
                                                type: 'video_upload_start',
                                                filename: file.name,
                                                size: file.size,
                                                contentType: file.type,
                                                chunkSize: CHUNK_SIZE
                                            }));
                                            
                                            const sendChunk = () => {
                                                return new Promise((resolve, reject) => {
                                                    const chunk = file.slice(offset, offset + CHUNK_SIZE);
                                                    console.log(`Preparing chunk: offset=${offset}, size=${chunk.size}`);
                                                    reader.onload = async () => {
                                                        try {
                                                            // Send chunk metadata
                                                            ws.send(JSON.stringify({
                                                                type: 'video_upload_chunk',
                                                                offset: offset,
                                                                size: chunk.size,
                                                                progress: Math.round((offset / file.size) * 100)
                                                            }));
                                                            console.log(`Uploading chunk: ${Math.round((offset / file.size) * 100)}%`);
                                                            
                                                            // Send chunk data
                                                            ws.send(reader.result);
                                                            
                                                            offset += chunk.size;
                                                            
                                                            // Add small delay between chunks
                                                            await new Promise(r => setTimeout(r, 100));
                                                            
                                                            if (offset < file.size) {
                                                                await sendChunk();
                                                            } else {
                                                                ws.send(JSON.stringify({
                                                                    type: 'video_upload_complete',
                                                                    filename: file.name,
                                                                    totalSize: file.size,
                                                                    chunks: Math.ceil(file.size / CHUNK_SIZE)
                                                                }));
                                                                console.log('Upload completed, waiting for server confirmation...');
                                                            }
                                                            resolve();
                                                        } catch (error) {
                                                            reject(error);
                                                        }
                                                    };
                                                    reader.onerror = reject;
                                                    reader.readAsArrayBuffer(chunk);
                                                });
                                            };
                                            
                                            // Handle upload start acknowledgment
                                            ws.onmessage = async (event) => {
                                                try {
                                                    const response = JSON.parse(event.data);
                                                    console.log('Received WebSocket response:', response);
                                                    if (response.type === 'upload_start_ack') {
                                                        try {
                                                            await sendChunk();
                                                            console.log('Upload completed successfully');
                                                        } catch (error) {
                                                            console.error('Upload failed:', error);
                                                        }
                                                    }
                                                } catch (error) {
                                                    console.error('Error processing WebSocket message:', error);
                                                }
                                            };
                                        }
                                    }}
                                />
                            )}
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
