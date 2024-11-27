import React, { useCallback, useRef, useEffect } from 'react';
import { Box, IconButton, useTheme, useMediaQuery } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import useStore from '../store';
import { websocketService } from '../services/websocket';

const CameraSelector = () => {
    const { isStreaming, setIsStreaming } = useStore();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const streamRef = useRef(null);
    const frameRequestRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    const startCamera = useCallback(async () => {
        try {
            // Stop any existing stream
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
            if (frameRequestRef.current) {
                cancelAnimationFrame(frameRequestRef.current);
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            streamRef.current = stream;

            // Initialize video element
            videoRef.current = document.createElement('video');
            videoRef.current.srcObject = stream;
            await videoRef.current.play();

            // Initialize canvas for frame capture
            canvasRef.current = document.createElement('canvas');
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;

            setIsStreaming(true);

            // Start frame capture loop
            const captureFrame = async () => {
                if (!isStreaming || !websocketService.isConnected()) {
                    cleanup();
                    return;
                }

                try {
                    const ctx = canvasRef.current.getContext('2d');
                    ctx.drawImage(videoRef.current, 0, 0);
                    
                    // Convert to blob and emit
                    const blob = await new Promise(resolve => {
                        canvasRef.current.toBlob(resolve, 'image/jpeg', 0.85);
                    });

                    // Convert blob to buffer
                    const arrayBuffer = await blob.arrayBuffer();
                    
                    // Emit frame
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
        }
    }, [isStreaming, setIsStreaming]);

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

        setIsStreaming(false);
    }, [setIsStreaming]);

    const handleStartStop = useCallback(() => {
        if (isStreaming) {
            cleanup();
        } else {
            startCamera();
        }
    }, [isStreaming, startCamera, cleanup]);

    // Cleanup on unmount
    useEffect(() => {
        return cleanup;
    }, [cleanup]);

    const buttonSize = isMobile ? 36 : 42;
    const iconSize = isMobile ? 20 : 24;

    return (
        <Box sx={{ mb: 2 }}>
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
            </Box>
        </Box>
    );
};

export default CameraSelector;
