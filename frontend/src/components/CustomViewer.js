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
        }
    }, [inputMode, currentVideo]);

    // Handle camera stream display
    useEffect(() => {
        if (inputMode === 'camera' && canvasRef.current) {
            const handleFrame = (frameData) => {
                const canvas = canvasRef.current;
                if (!canvas) return;

                const ctx = canvas.getContext('2d');
                const blob = new Blob([frameData], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                const img = new Image();

                img.onload = () => {
                    // Calculate dimensions to maintain aspect ratio
                    const imgAspect = img.width / img.height;
                    const canvasAspect = canvas.width / canvas.height;
                    let drawWidth = canvas.width;
                    let drawHeight = canvas.height;
                    let offsetX = 0;
                    let offsetY = 0;

                    if (imgAspect > canvasAspect) {
                        // Image is wider than canvas
                        drawHeight = canvas.width / imgAspect;
                        offsetY = (canvas.height - drawHeight) / 2;
                    } else {
                        // Image is taller than canvas
                        drawWidth = canvas.height * imgAspect;
                        offsetX = (canvas.width - drawWidth) / 2;
                    }

                    // Clear canvas
                    ctx.fillStyle = '#000';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw frame
                    ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
                    URL.revokeObjectURL(url);
                };

                img.onerror = () => {
                    URL.revokeObjectURL(url);
                    console.error('Failed to load frame');
                };

                img.src = url;
            };

            websocketService.on('frame', handleFrame);
            return () => websocketService.off('frame', handleFrame);
        }
    }, [inputMode]);

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
