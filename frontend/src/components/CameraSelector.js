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

            // Connect WebSocket if needed
            if (!websocketService.isConnected()) {
                websocketService.connect();
            }

            // Initialize stream
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: { exact: selectedDevice },
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                }
            });

            streamRef.current = stream;
            
            // Setup video and canvas
            const video = document.createElement('video');
            video.srcObject = stream;
            video.muted = true;
            await video.play();

            canvasRef.current = document.createElement('canvas');
            canvasRef.current.width = 1280;
            canvasRef.current.height = 720;
            const ctx = canvasRef.current.getContext('2d');

            // Start frame capture
            websocketService.emit('start_stream');
            
            const captureFrame = () => {
                if (!streamRef.current) return;

                ctx.drawImage(video, 0, 0, canvasRef.current.width, canvasRef.current.height);
                canvasRef.current.toBlob(
                    (blob) => websocketService.emit('frame', blob),
                    'image/jpeg',
                    0.85
                );

                animationFrameRef.current = requestAnimationFrame(captureFrame);
            };

            animationFrameRef.current = requestAnimationFrame(captureFrame);
            setIsStreaming(true);

        } catch (err) {
            console.error('Camera start error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [selectedDevice, setIsStreaming]);

    const stopCameraStream = useCallback(() => {
        // Cleanup animation frame
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        // Stop media tracks
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        // Stop WebSocket stream
        websocketService.emit('stop_stream');
        setIsStreaming(false);
    }, [setIsStreaming]);

    // Setup WebSocket listeners
    useEffect(() => {
        websocketService.on('stream_started', () => {
            console.log('Stream started on server');
        });

        websocketService.on('stream_stopped', () => {
            console.log('Stream stopped on server');
        });

        websocketService.on('error', (error) => {
            console.error('Stream error:', error);
            setError(error.message);
            stopCameraStream();
        });

        return () => {
            websocketService.off('stream_started');
            websocketService.off('stream_stopped');
            websocketService.off('error');
            stopCameraStream();
        };
    }, [stopCameraStream]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopCameraStream();
        };
    }, [stopCameraStream]);

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

    const handleStartStop = () => {
        if (isStreaming) {
            stopCameraStream();
        } else {
            startCameraStream();
        }
    };

    // WebSocket connection status monitoring
    useEffect(() => {
        const handleDisconnect = () => {
            if (isStreaming) {
                stopCameraStream();
                setError('WebSocket connection lost');
            }
        };

        websocketService.on('disconnect', handleDisconnect);

        return () => {
            websocketService.off('disconnect', handleDisconnect);
        };
    }, [isStreaming, stopCameraStream]);

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
