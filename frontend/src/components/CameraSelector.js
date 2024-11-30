import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
    Box,
    IconButton,
    Typography,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    CircularProgress
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import VideocamIcon from '@mui/icons-material/Videocam';
import useStore from '../store';
import { websocketService } from '../services/websocket';

const CameraSelector = () => {
    const { isStreaming, setIsStreaming } = useStore();
    const [devices, setDevices] = useState([]);
    const [selectedDevice, setSelectedDevice] = useState('');
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const streamRef = useRef(null);
    const canvasRef = useRef(null);
    const animationFrameRef = useRef(null);

    // Initialize camera devices
    useEffect(() => {
        const initDevices = async () => {
            try {
                const mediaDevices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = mediaDevices.filter(device => device.kind === 'videoinput');
                setDevices(videoDevices);

                if (videoDevices.length > 0 && !selectedDevice) {
                    setSelectedDevice(videoDevices[0].deviceId);
                }
            } catch (err) {
                console.error('Error getting camera devices:', err);
                setError('Failed to get camera devices');
            }
        };

        navigator.mediaDevices.addEventListener('devicechange', initDevices);
        initDevices();

        return () => {
            navigator.mediaDevices.removeEventListener('devicechange', initDevices);
        };
    }, []);

    const startCameraStream = useCallback(async () => {
        if (!selectedDevice) return;

        try {
            setLoading(true);
            setError(null);

            // Get camera stream
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: { exact: selectedDevice },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });

            // Store stream reference first
            streamRef.current = stream;

            // Create and setup video element
            const video = document.createElement('video');
            video.srcObject = stream;
            video.playsInline = true;

            // Wait for video to be ready
            await new Promise((resolve) => {
                video.onloadedmetadata = async () => {
                    try {
                        await video.play();
                        resolve();
                    } catch (e) {
                        console.error('Video play error:', e);
                    }
                };
            });

            // Create canvas matching video dimensions
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
            const ctx = canvas.getContext('2d');

            // Ensure WebSocket is connected first
            if (!websocketService.isConnected()) {
                await new Promise((resolve) => {
                    websocketService.connect();
                    websocketService.on('connect', resolve);
                });
            }

            // Signal stream start
            websocketService.emit('start_stream');

            // Frame capture loop
            const captureFrame = () => {
                if (!streamRef.current?.active) {
                    console.log('Stream not active, stopping capture');
                    return;
                }

                try {
                    // Draw current video frame
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    // Convert to blob and send
                    canvas.toBlob(blob => {
                        if (blob && websocketService.isConnected()) {
                            // Convert blob to array buffer
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                try {
                                    const arrayBuffer = reader.result;
                                    websocketService.emit('frame', arrayBuffer);
                                } catch (e) {
                                    console.error('Frame send error:', e);
                                }
                            };
                            reader.readAsArrayBuffer(blob);
                        }
                    }, 'image/jpeg', 0.85);

                    // Continue capture loop
                    if (streamRef.current?.active) {
                        animationFrameRef.current = requestAnimationFrame(captureFrame);
                    }
                } catch (err) {
                    console.error('Frame capture error:', err);
                }
            };

            // Start capture loop
            animationFrameRef.current = requestAnimationFrame(captureFrame);
            setIsStreaming(true);
            setLoading(false);

        } catch (err) {
            console.error('Camera start error:', err);
            setError(err.message);
            setLoading(false);
            stopCameraStream();
        }
    }, [selectedDevice, setIsStreaming]);

    // Update stop function to ensure proper cleanup
    const stopCameraStream = useCallback(() => {
        console.log('Stopping camera stream...');

        // Stop animation frame
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        // Stop media tracks
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        // Signal stream stop to server
        if (websocketService.isConnected()) {
            websocketService.emit('stop_stream');
        }

        setIsStreaming(false);
        setError(null);
    }, [setIsStreaming]);

    // Handle device change
    const handleDeviceChange = (event) => {
        const newDevice = event.target.value;
        setSelectedDevice(newDevice);

        if (isStreaming) {
            stopCameraStream();
            // Short delay to ensure previous stream is fully stopped
            setTimeout(() => {
                if (newDevice) {
                    startCameraStream();
                }
            }, 100);
        }
    };

    // Handle start/stop
    const handleStartStop = () => {
        if (isStreaming) {
            stopCameraStream();
        } else {
            startCameraStream();
        }
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopCameraStream();
        };
    }, [stopCameraStream]);

    return (
        <Box sx={{ mb: 2 }}>
            <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                gap: 2
            }}>
                <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1
                }}>
                    <VideocamIcon
                        sx={{
                            color: isStreaming ? 'success.main' : 'text.secondary',
                            animation: loading ? 'pulse 1.5s infinite' : 'none',
                            '@keyframes pulse': {
                                '0%': { opacity: 1 },
                                '50%': { opacity: 0.4 },
                                '100%': { opacity: 1 }
                            }
                        }}
                    />
                    <Typography
                        variant="body2"
                        color={isStreaming ? 'primary' : 'text.secondary'}
                    >
                        {loading ? 'Connecting...' :
                            isStreaming ? 'Camera Active' : 'Camera Inactive'}
                    </Typography>
                </Box>

                <FormControl
                    fullWidth
                    variant="outlined"
                    error={!!error}
                >
                    <InputLabel>Camera Device</InputLabel>
                    <Select
                        value={selectedDevice}
                        onChange={handleDeviceChange}
                        label="Camera Device"
                        disabled={loading || isStreaming}
                    >
                        {devices.map((device) => (
                            <MenuItem key={device.deviceId} value={device.deviceId}>
                                {device.label || `Camera ${device.deviceId.slice(0, 5)}...`}
                            </MenuItem>
                        ))}
                    </Select>
                    {error && (
                        <Typography
                            variant="caption"
                            color="error"
                            sx={{ mt: 0.5 }}
                        >
                            {error}
                        </Typography>
                    )}
                </FormControl>

                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                    <IconButton
                        onClick={handleStartStop}
                        disabled={!selectedDevice || loading}
                        sx={{
                            width: 48,
                            height: 48,
                            bgcolor: isStreaming ? 'error.main' : 'success.main',
                            color: 'white',
                            '&:hover': {
                                bgcolor: isStreaming ? 'error.dark' : 'success.dark'
                            },
                            '&.Mui-disabled': {
                                bgcolor: 'action.disabledBackground'
                            }
                        }}
                    >
                        {loading ? (
                            <CircularProgress size={24} color="inherit" />
                        ) : isStreaming ? (
                            <StopIcon />
                        ) : (
                            <PlayArrowIcon />
                        )}
                    </IconButton>
                </Box>
            </Box>
        </Box>
    );
};

export default CameraSelector;
