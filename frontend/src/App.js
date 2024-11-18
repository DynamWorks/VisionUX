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
        const websocket = new WebSocket(process.env.REACT_APP_WS_URL);
        setWs(websocket);
        return () => {
            if (websocket) {
                websocket.close();
            }
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
                                        
                                        // Send video file through WebSocket
                                        if (ws && ws.readyState === WebSocket.OPEN) {
                                            ws.send(JSON.stringify({ type: 'video_upload' }));
                                            ws.send(file);
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
