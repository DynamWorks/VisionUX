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
        let animationFrameId;
        
        if (inputMode === 'camera' && canvasRef.current && isStreaming && window.activeStream) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            
            // Create video element to receive camera stream
            const video = document.createElement('video');
            video.srcObject = window.activeStream;
            video.play();

            // Wait for video to be ready
            video.onloadedmetadata = () => {
                const containerWidth = containerRef.current.offsetWidth;
                const containerHeight = containerRef.current.offsetHeight;
                canvas.width = containerWidth;
                canvas.height = containerHeight;
                setLoading(false);
            };

            // Draw frames from video to canvas
            const drawFrame = () => {
                if (ctx && video.readyState === video.HAVE_ENOUGH_DATA) {
                    // Get container dimensions
                    const containerWidth = containerRef.current.offsetWidth;
                    const containerHeight = containerRef.current.offsetHeight;
                    
                    // Calculate aspect ratio
                    const videoAspect = video.videoWidth / video.videoHeight;
                    const containerAspect = containerWidth / containerHeight;
                    
                    let drawWidth = containerWidth;
                    let drawHeight = containerHeight;
                    
                    // Maintain aspect ratio while fitting in container
                    if (containerAspect > videoAspect) {
                        drawWidth = containerHeight * videoAspect;
                    } else {
                        drawHeight = containerWidth / videoAspect;
                    }
                    
                    // Center the video
                    const x = (containerWidth - drawWidth) / 2;
                    const y = (containerHeight - drawHeight) / 2;
                    
                    // Clear canvas and draw frame
                    ctx.fillStyle = '#000';
                    ctx.fillRect(0, 0, containerWidth, containerHeight);
                    ctx.drawImage(video, x, y, drawWidth, drawHeight);
                }
                animationFrameId = requestAnimationFrame(drawFrame);
            };
            
            drawFrame();
        }

        // Cleanup function
        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
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
                        display: inputMode === 'camera' && isStreaming ? 'block' : 'none',
                        backgroundColor: '#000',
                        objectFit: 'contain',
                        position: 'absolute',
                        top: 0,
                        left: 0
                    }}
                />
            </Box>
        </Box>
    );
};

export default CustomViewer;
