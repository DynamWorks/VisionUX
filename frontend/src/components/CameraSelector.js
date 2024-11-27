import React, { useCallback, useRef, useEffect } from 'react';
import { Box, FormControl, Select, MenuItem, IconButton, useTheme, useMediaQuery } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import useStore from '../store';
import { websocketService } from '../services/websocket';

const CameraSelector = ({ 
    devices, 
    selectedDevice, 
    setSelectedDevice, 
    isStreaming,
    refreshDevices
}) => {
    const { setIsStreaming, setStreamMetrics } = useStore();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const streamRef = useRef(null);
    const frameRequestRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    const cleanup = useCallback(() => {
        if (frameRequestRef.current) {
            cancelAnimationFrame(frameRequestRef.current);
            frameRequestRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => {
                track.stop();
                track.enabled = false;
            });
            streamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
            videoRef.current = null;
        }

        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            canvasRef.current = null;
        }
    }, []);

    const startCamera = useCallback(async () => {
        if (!selectedDevice) {
            alert('Please select a camera device first');
            return;
        }

        try {
            // Clean up any existing resources
            cleanup();

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            streamRef.current = stream;

            // Initialize video element
            videoRef.current = document.createElement('video');
            videoRef.current.srcObject = stream;
            await videoRef.current.play();

            // Initialize canvas
            canvasRef.current = document.createElement('canvas');
            canvasRef.current.width = 1280;
            canvasRef.current.height = 720;
            const ctx = canvasRef.current.getContext('2d');

            // Start stream on server
            websocketService.emit('start_stream', {
                deviceId: selectedDevice,
                width: 1280,
                height: 720
            });
            setIsStreaming(true);
            
            // Set initial metrics
            setStreamMetrics({
                fps: 0,
                frameCount: 0,
                resolution: '1280x720',
                timestamp: Date.now()
            });

            // Start frame capture loop
            const captureFrame = async () => {
                if (!isStreaming || !websocketService.isConnected()) {
                    cleanup();
                    setIsStreaming(false);
                    return;
                }

                try {
                    // Draw frame to canvas
                    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
                    
                    // Convert to blob and emit
                    const blob = await new Promise(resolve => {
                        canvasRef.current.toBlob(resolve, 'image/jpeg', 0.85);
                    });

                    // Convert blob to buffer
                    const arrayBuffer = await blob.arrayBuffer();
                    
                    // Emit frame metadata
                    websocketService.emit('frame_metadata', {
                        timestamp: Date.now(),
                        width: canvasRef.current.width,
                        height: canvasRef.current.height
                    });
                    
                    // Emit binary frame data
                    websocketService.emit('frame', arrayBuffer);

                    // Wait for next frame
                    frameRequestRef.current = requestAnimationFrame(captureFrame);
                } catch (error) {
                    console.error('Error capturing frame:', error);
                    if (error.message !== 'WebSocket not connected') {
                        frameRequestRef.current = requestAnimationFrame(captureFrame);
                    }
                }
            };

            frameRequestRef.current = requestAnimationFrame(captureFrame);

        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Failed to access camera: ' + error.message);
            cleanup();
            setIsStreaming(false);
        }
    }, [selectedDevice, isStreaming, setIsStreaming, cleanup]);

    const stopCamera = useCallback(async () => {
        try {
            // Stop stream on server first
            websocketService.emit('stop_stream');
            
            // Clean up resources
            cleanup();
            
            // Update state
            setIsStreaming(false);
        } catch (error) {
            console.error('Error stopping camera:', error);
            // Force cleanup even if there's an error
            cleanup();
            setIsStreaming(false);
        }
    }, [cleanup, setIsStreaming]);

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
            cleanup();
        };
    }, [cleanup]);

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
