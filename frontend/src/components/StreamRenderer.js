import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { websocketService } from '../services/websocket';
import useStore from '../store';

const StreamRenderer = ({ source, isStreaming }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const { setStreamMetrics } = useStore();
    const isFirstRender = useRef(true);


    // Add FPS counter ref near the top
    const fpsCounterRef = useRef({ frames: 0, lastUpdate: Date.now() });

    // Update the drawFrame function to include FPS counting and canvas sizing
    const drawFrame = useCallback(async (frameData) => {
        if (!canvasRef.current || !containerRef.current) return;

        try {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            // Convert frame data to blob
            let blob;
            if (frameData instanceof ArrayBuffer) {
                blob = new Blob([frameData], { type: 'image/jpeg' });
            } else if (frameData instanceof Blob) {
                blob = frameData;
            } else {
                throw new Error('Invalid frame data type');
            }

            // Create and load image
            const img = new Image();
            const url = URL.createObjectURL(blob);

            await new Promise((resolve, reject) => {
                img.onload = () => {
                    // Set canvas size to match first frame if needed
                    if (canvas.width !== img.width || canvas.height !== img.height) {
                        canvas.width = img.width;
                        canvas.height = img.height;

                        // Adjust canvas style to maintain aspect ratio
                        const containerWidth = containerRef.current.offsetWidth;
                        const containerHeight = containerRef.current.offsetHeight;
                        const scale = Math.min(
                            containerWidth / img.width,
                            containerHeight / img.height
                        );

                        canvas.style.width = `${img.width * scale}px`;
                        canvas.style.height = `${img.height * scale}px`;
                    }

                    // Draw frame
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);

                    URL.revokeObjectURL(url);
                    resolve();
                };
                img.onerror = () => {
                    URL.revokeObjectURL(url);
                    reject(new Error('Failed to load image'));
                };
                img.src = url;
            });

            setLoading(false);
            setError(null);

        } catch (err) {
            console.error('Error drawing frame:', err);
            setError('Error displaying frame');
        }
    }, []);


    // Setup WebSocket event listeners
    useEffect(() => {
        // Skip first render cleanup
        if (isFirstRender.current) {
            isFirstRender.current = false;
            return;
        }

        if (!isStreaming || source !== 'camera') {
            console.log('Stream not active or source not camera');
            return;
        }

        console.log('Setting up WebSocket listeners for', source);

        const handleFrame = async (frameData) => {
            console.log('Received frame data');
            await drawFrame(frameData);
        };

        const handleStreamStarted = () => {
            console.log('Stream started event received');
            setLoading(false);
        };

        const handleStreamError = (error) => {
            console.error('Stream error:', error);
            setError(error.message || 'Stream error occurred');
            setLoading(false);
        };

        // Add event listeners
        websocketService.on('frame', handleFrame);
        websocketService.on('stream_started', handleStreamStarted);
        websocketService.on('error', handleStreamError);

        setLoading(true);
        console.log('Stream listeners setup complete');

        // Cleanup function
        return () => {
            // Only cleanup if we're actually stopping the stream
            if (!isStreaming) {
                console.log('Cleaning up WebSocket listeners');
                websocketService.off('frame', handleFrame);
                websocketService.off('stream_started', handleStreamStarted);
                websocketService.off('error', handleStreamError);
            }
        };
    }, [isStreaming, source, drawFrame]);

    // Handle container resizing
    useEffect(() => {
        if (!containerRef.current || !canvasRef.current) return;

        const resizeObserver = new ResizeObserver(() => {
            if (canvasRef.current) {
                const canvas = canvasRef.current;
                const container = containerRef.current;
                canvas.style.width = '100%';
                canvas.style.height = '100%';
            }
        });

        resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    return (
        <Box
            ref={containerRef}
            sx={{
                width: '100%',
                height: '100%',
                minHeight: '400px',
                bgcolor: '#1a1a1a',
                borderRadius: '8px',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                justifyContent: 'center', // Center vertically
                alignItems: 'center'      // Center horizontally
            }}
        >
            <canvas
                ref={canvasRef}
                style={{
                    display: isStreaming ? 'block' : 'none',
                    maxWidth: '100%',     // Constrain to container
                    maxHeight: '100%',    // Constrain to container
                    objectFit: 'contain', // Maintain aspect ratio
                    backgroundColor: '#000000' // Make it visible even when empty
                }}
            />

            {loading && (
                <Box
                    sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)'
                    }}
                >
                    <CircularProgress />
                </Box>
            )}

            {error && (
                <Typography
                    color="error"
                    sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        textAlign: 'center'
                    }}
                >
                    {error}
                </Typography>
            )}

            {!isStreaming && !loading && (
                <Typography
                    sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        color: 'text.secondary',
                        textAlign: 'center'
                    }}
                >
                    {source === 'camera' ? 'Camera not started' : 'No video selected'}
                </Typography>
            )}
        </Box>
    );
};

export default StreamRenderer;
