import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { websocketService } from '../services/websocket';
import useStore from '../store';

const StreamRenderer = ({ source, isStreaming }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const videoRef = useRef(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const { setIsStreaming } = useStore();

    const handleFrame = useCallback(async (data) => {
        if (!canvasRef.current) return;
        
        try {
            const ctx = canvasRef.current.getContext('2d');
            const canvas = canvasRef.current;

            let imageData;
            if (data instanceof Blob) {
                imageData = await createImageBitmap(data);
            } else if (data instanceof ArrayBuffer) {
                const blob = new Blob([data], { type: 'image/jpeg' });
                imageData = await createImageBitmap(blob);
            } else if (typeof data === 'string' && data.startsWith('data:')) {
                const response = await fetch(data);
                const blob = await response.blob();
                imageData = await createImageBitmap(blob);
            } else if (data instanceof HTMLVideoElement) {
                imageData = data;
            } else {
                throw new Error('Invalid frame data');
            }

            // Calculate dimensions preserving aspect ratio
            const containerWidth = containerRef.current?.offsetWidth || canvas.width;
            const containerHeight = containerRef.current?.offsetHeight || canvas.height;
            const scale = Math.min(
                containerWidth / imageData.width,
                containerHeight / imageData.height
            );

            canvas.width = imageData.width * scale;
            canvas.height = imageData.height * scale;

            // Clear and draw
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(imageData, 0, 0, canvas.width, canvas.height);

            if (imageData instanceof ImageBitmap) {
                imageData.close();
            }
            } catch (err) {
                console.error('Error processing frame:', err);
                setError('Error displaying stream');
            }
        };

    }, []);

    // Handle frame rendering
    useEffect(() => {
        if (!isStreaming) return;

        let animationFrameId;
        const frameHandler = source === 'camera' ? 
            (data) => handleFrame(data) :
            () => {
                if (videoRef.current && !videoRef.current.paused && !videoRef.current.ended) {
                    handleFrame(videoRef.current);
                }
                animationFrameId = requestAnimationFrame(frameHandler);
            };

        if (source === 'camera') {
            websocketService.on('frame', frameHandler);
            setIsStreaming(true);
        } else if (source === 'video' && videoRef.current) {
            frameHandler();
        }

        return () => {
            if (source === 'camera') {
                websocketService.off('frame', frameHandler);
            }
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
            setIsStreaming(false);
        };
    }, [isStreaming, source, handleFrame, setIsStreaming]);

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
