import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Box, FormControl, Select, MenuItem, IconButton, useTheme, useMediaQuery } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import useStore from '../store';
import { websocketService } from '../services/websocket';

const CameraSelector = () => {
    const { setIsStreaming, setStreamMetrics } = useStore();
    const [devices, setDevices] = useState([]);
    const [selectedDevice, setSelectedDevice] = useState('');
    const [isStreaming, setLocalStreaming] = useState(false);
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const streamRef = useRef(null);
    const frameRequestRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    // Get available cameras on mount and after permissions granted
    useEffect(() => {
        const getDevices = async () => {
            try {
                // Request initial camera permission in Safari
                if (navigator.userAgent.includes('Safari') && !navigator.userAgent.includes('Chrome')) {
                    await navigator.mediaDevices.getUserMedia({ video: true })
                        .then(stream => {
                            // Stop the stream immediately after getting permission
                            stream.getTracks().forEach(track => track.stop());
                        });
                }
                
                // Now enumerate devices
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                
                // Filter out devices with empty labels (not permitted)
                const availableDevices = videoDevices.filter(device => device.label);
                
                setDevices(availableDevices);
                if (availableDevices.length > 0 && !selectedDevice) {
                    setSelectedDevice(availableDevices[0].deviceId);
                }
            } catch (error) {
                console.error('Error getting camera devices:', error);
                // Show user-friendly error for permission denial
                if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                    alert('Camera permission is required to list available devices. Please allow camera access and try again.');
                }
            }
        };
        
        getDevices();

        // Add device change listener
        navigator.mediaDevices.addEventListener('devicechange', getDevices);
        
        return () => {
            navigator.mediaDevices.removeEventListener('devicechange', getDevices);
        };
    }, [selectedDevice]);

    // Handle WebSocket connection
    useEffect(() => {
        const handleStreamMetrics = (metrics) => {
            setStreamMetrics(metrics);
        };

        websocketService.on('stream_metrics', handleStreamMetrics);
        return () => {
            websocketService.off('stream_metrics', handleStreamMetrics);
        };
    }, [setStreamMetrics]);

    const startCamera = useCallback(async () => {
        if (!selectedDevice) {
            alert('Please select a camera device first');
            return;
        }

        try {
            // Request camera access
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: { exact: selectedDevice },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });

            // Start stream via WebSocket
            websocketService.emit('start_stream', {
                deviceId: selectedDevice,
                width: 1280,
                height: 720
            });

            setIsStreaming(true);
            setLocalStreaming(true);

            // Clean up stream when component unmounts
            return () => {
                stream.getTracks().forEach(track => track.stop());
            };
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Failed to access camera: ' + error.message);
        }
    }, [selectedDevice, setIsStreaming]);

    const stopCamera = useCallback(() => {
        websocketService.emit('stop_stream');
        setIsStreaming(false);
        setLocalStreaming(false);
    }, [setIsStreaming]);

    const refreshDevices = useCallback(async () => {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            setDevices(videoDevices);
        } catch (error) {
            console.error('Error refreshing devices:', error);
        }
    }, []);

    const handleStartStop = useCallback(() => {
        if (isStreaming) {
            stopCamera();
        } else {
            startCamera();
        }
    }, [isStreaming, startCamera, stopCamera]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    const buttonSize = isMobile ? 36 : 42;
    const iconSize = isMobile ? 20 : 24;

    return (
        <Box sx={{ mb: 2 }}>
            <FormControl 
                fullWidth 
                sx={{ 
                    mb: 2,
                    '& .MuiOutlinedInput-root': {
                        height: buttonSize,
                        bgcolor: '#1a1a1a',
                        color: 'white',
                        '& fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                    }
                }}
            >
                <Select
                    value={selectedDevice || ''}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    disabled={isStreaming}
                    MenuProps={{
                        PaperProps: {
                            sx: {
                                bgcolor: '#1a1a1a',
                                color: 'white',
                                maxHeight: 300
                            }
                        }
                    }}
                >
                    {devices.map((device) => (
                        <MenuItem 
                            key={device.deviceId} 
                            value={device.deviceId}
                            sx={{
                                height: buttonSize,
                                whiteSpace: 'normal',
                                wordBreak: 'break-word'
                            }}
                        >
                            {device.label || `Camera ${device.deviceId}`}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton
                    onClick={handleStartStop}
                    sx={{
                        width: buttonSize,
                        height: buttonSize,
                        bgcolor: isStreaming ? '#d32f2f' : '#2e7d32',
                        color: 'white',
                        '&:hover': {
                            bgcolor: isStreaming ? '#9a0007' : '#1b5e20'
                        }
                    }}
                >
                    {isStreaming ? 
                        <StopIcon sx={{ fontSize: iconSize }} /> : 
                        <PlayArrowIcon sx={{ fontSize: iconSize }} />
                    }
                </IconButton>
                <IconButton
                    onClick={refreshDevices}
                    disabled={isStreaming}
                    sx={{
                        width: buttonSize,
                        height: buttonSize,
                        bgcolor: '#1976d2',
                        color: 'white',
                        '&:hover': {
                            bgcolor: '#1565c0'
                        },
                        '&.Mui-disabled': {
                            bgcolor: 'rgba(25, 118, 210, 0.3)',
                            color: 'rgba(255, 255, 255, 0.3)'
                        }
                    }}
                >
                    <RefreshIcon sx={{ fontSize: iconSize }} />
                </IconButton>
            </Box>
        </Box>
    );
};

export default CameraSelector;
