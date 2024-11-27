import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import useStore from '../store';
import { websocketService } from '../services/websocket';

const CustomViewer = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const { currentVideo, inputMode, isStreaming } = useStore();

    // Dynamic sizing
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current && canvasRef.current) {
                const container = containerRef.current;
                const width = container.offsetWidth;
                const height = container.offsetHeight;

                canvasRef.current.width = width;
                canvasRef.current.height = height;

                // Clear canvas with black background
                const ctx = canvasRef.current.getContext('2d');
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, width, height);
            }
        };

        const resizeObserver = new ResizeObserver(updateDimensions);
        if (containerRef.current) {
            resizeObserver.observe(containerRef.current);
        }

        return () => {
            resizeObserver.disconnect();
        };
    }, []);

    // Handle video playback
    useEffect(() => {
        if (inputMode === 'upload' && currentVideo && videoRef.current) {
            const videoPath = `${process.env.REACT_APP_API_URL}/api/v1/tmp_content/uploads/${currentVideo.name}`;
            videoRef.current.src = videoPath;
            videoRef.current.load();
            setError(null);
            setLoading(true);

            // Cleanup function
            return () => {
                if (videoRef.current) {
                    videoRef.current.pause();
                    videoRef.current.src = '';
                    videoRef.current.load();
                }
            };
        }
    }, [inputMode, currentVideo]);

    // Handle camera stream display
    useEffect(() => {
        let frameHandler;
        let tempVideoElement;

        if (inputMode === 'camera' && canvasRef.current && isStreaming) {
            frameHandler = (frameData) => {
                const canvas = canvasRef.current;
                if (!canvas) return;

                const ctx = canvas.getContext('2d');
                
                // Reuse video element if possible
                if (!tempVideoElement) {
                    tempVideoElement = document.createElement('video');
                    tempVideoElement.autoplay = true;
                    tempVideoElement.playsInline = true;
                }

                // Create object URL from frame data
                const blob = new Blob([frameData], { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                tempVideoElement.src = url;

                tempVideoElement.onloadedmetadata = () => {
                    // Set canvas size to match video
                    canvas.width = tempVideoElement.videoWidth;
                    canvas.height = tempVideoElement.videoHeight;

                    // Draw frame
                    ctx.drawImage(tempVideoElement, 0, 0);

                    // Clean up URL only
                    URL.revokeObjectURL(url);
                };

                tempVideoElement.onerror = () => {
                    console.error('Error loading video frame');
                    URL.revokeObjectURL(url);
                };
            };

            websocketService.on('frame', frameHandler);
        }

        // Cleanup function
        return () => {
            if (frameHandler) {
                websocketService.off('frame', frameHandler);
            }
            if (tempVideoElement) {
                tempVideoElement.pause();
                tempVideoElement.src = '';
                tempVideoElement.remove();
                tempVideoElement = null;
            }
        };
    }, [inputMode, isStreaming]);

    return (
        <Box 
            ref={containerRef}
            sx={{
                width: '100%',
                height: '100%',
                bgcolor: '#1a1a1a',
                borderRadius: '8px',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative'
            }}
        >
            <Box sx={{
                flex: 1,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                position: 'relative'
            }}>
                {loading && (
                    <CircularProgress 
                        sx={{ 
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            zIndex: 2
                        }}
                    />
                )}
                
                {error && (
                    <Typography 
                        variant="body1" 
                        sx={{ 
                            color: 'error.main',
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            zIndex: 2,
                            textAlign: 'center',
                            width: '80%'
                        }}
                    >
                        {error}
                    </Typography>
                )}

                {!currentVideo && !isStreaming && !error && (
                    <Typography 
                        variant="body1" 
                        sx={{ 
                            color: 'text.secondary',
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            zIndex: 2,
                            textAlign: 'center',
                            width: '80%'
                        }}
                    >
                        {inputMode === 'upload' ? 'No video selected' : 'Camera not started'}
                    </Typography>
                )}

                {/* Video Player */}
                {inputMode === 'upload' && (
                    <video
                        ref={videoRef}
                        style={{
                            width: '100%',
                            height: '100%',
                            display: currentVideo && !loading ? 'block' : 'none',
                            backgroundColor: '#000',
                            objectFit: 'contain'
                        }}
                        controls
                        autoPlay
                        onLoadedData={() => setLoading(false)}
                        onError={(e) => {
                            console.error('Video error:', e);
                            setError('Error loading video');
                            setLoading(false);
                        }}
                    />
                )}

                {/* Camera Stream */}
                <canvas
                    ref={canvasRef}
                    style={{
                        width: '100%',
                        height: '100%',
                        display: inputMode === 'camera' ? 'block' : 'none',
                        backgroundColor: '#000',
                        objectFit: 'contain'
                    }}
                />
            </Box>
        </Box>
    );
};

export default CustomViewer;
