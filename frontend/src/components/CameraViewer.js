import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { websocketService } from '../services/websocket';

const CameraViewer = ({ isStreaming }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [frameData, setFrameData] = useState(null);

    // Handle WebSocket frame data
    useEffect(() => {
        if (!isStreaming) return;

        const handleFrame = async (data) => {
            try {
                // Convert binary data to image
                const blob = new Blob([data], { type: 'image/jpeg' });
                const imageUrl = URL.createObjectURL(blob);
                setFrameData(imageUrl);
                
                // Load and draw the image on canvas
                const img = new Image();
                img.onload = () => {
                    if (canvasRef.current) {
                        const ctx = canvasRef.current.getContext('2d');
                        const canvas = canvasRef.current;
                        
                        // Calculate aspect ratio preserving dimensions
                        const containerWidth = containerRef.current.offsetWidth;
                        const containerHeight = containerRef.current.offsetHeight;
                        const scale = Math.min(
                            containerWidth / img.width,
                            containerHeight / img.height
                        );
                        
                        canvas.width = img.width * scale;
                        canvas.height = img.height * scale;
                        
                        // Center the image
                        const x = (containerWidth - canvas.width) / 2;
                        const y = (containerHeight - canvas.height) / 2;
                        
                        // Clear and draw
                        ctx.fillStyle = '#000';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    }
                    URL.revokeObjectURL(imageUrl);
                };
                img.src = imageUrl;
            } catch (err) {
                console.error('Error processing frame:', err);
                setError('Error displaying camera feed');
            }
        };

        // Subscribe to frame events
        websocketService.on('frame', handleFrame);

        return () => {
            websocketService.off('frame', handleFrame);
            if (frameData) {
                URL.revokeObjectURL(frameData);
            }
        };
    }, [isStreaming, frameData]);

    // Handle container resizing
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current && canvasRef.current) {
                const container = containerRef.current;
                const canvas = canvasRef.current;
                
                // Set initial canvas size
                canvas.width = container.offsetWidth;
                canvas.height = container.offsetHeight;
                
                // Clear canvas with black background
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
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

                {!isStreaming && !error && (
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
                        Camera not started
                    </Typography>
                )}

                <canvas
                    ref={canvasRef}
                    style={{
                        width: '100%',
                        height: '100%',
                        display: isStreaming ? 'block' : 'none',
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

export default CameraViewer;
