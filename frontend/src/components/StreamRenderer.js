import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { websocketService } from '../services/websocket';

const StreamRenderer = ({ source, isStreaming }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    // Handle frame rendering
    useEffect(() => {
        if (!isStreaming) return;

        const handleFrame = async (data) => {
            try {
                let imageUrl;
                if (data instanceof Blob) {
                    imageUrl = URL.createObjectURL(data);
                } else if (data instanceof ArrayBuffer) {
                    const blob = new Blob([data], { type: 'image/jpeg' });
                    imageUrl = URL.createObjectURL(blob);
                } else if (typeof data === 'string' && data.startsWith('data:')) {
                    imageUrl = data;
                } else if (data instanceof HTMLVideoElement) {
                    // Handle video element directly
                    if (canvasRef.current) {
                        const ctx = canvasRef.current.getContext('2d');
                        const canvas = canvasRef.current;
                        
                        // Set canvas size to match video
                        canvas.width = data.videoWidth;
                        canvas.height = data.videoHeight;
                        
                        // Draw video frame
                        ctx.drawImage(data, 0, 0, canvas.width, canvas.height);
                        return;
                    }
                }

                if (!imageUrl) {
                    throw new Error('Invalid frame data');
                }

                // Load and draw the image
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
                setError('Error displaying stream');
            }
        };

        let animationFrameId;
        
        if (source === 'camera') {
            websocketService.on('frame', handleFrame);
        } else if (source === 'video') {
            // For video files, extract frames using requestAnimationFrame
            const extractFrame = () => {
                if (!isStreaming) return;
                const video = document.querySelector('video');
                if (video && !video.paused && !video.ended) {
                    handleFrame(video);
                }
                animationFrameId = requestAnimationFrame(extractFrame);
            };
            extractFrame();
        }

        return () => {
            if (source === 'camera') {
                websocketService.off('frame', handleFrame);
            }
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    }, [isStreaming, source]);

    // Handle container resizing
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current && canvasRef.current) {
                const container = containerRef.current;
                const canvas = canvasRef.current;
                
                canvas.width = container.offsetWidth;
                canvas.height = container.offsetHeight;
                
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
                        Stream not started
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

export default StreamRenderer;
