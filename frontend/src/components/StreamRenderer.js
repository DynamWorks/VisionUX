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

            // Handle frame data
            let img = new Image();
            let url;
            
            if (typeof frameData === 'string') {
                // Handle base64 string
                url = `data:image/jpeg;base64,${frameData}`;
            } else if (frameData instanceof Blob) {
                // Handle Blob data
                url = URL.createObjectURL(frameData);
            } else {
                throw new Error('Unsupported frame data format');
            }

            await new Promise((resolve, reject) => {
                img.onload = () => {
                    try {
                        // Initialize canvas on first frame
                        if (!canvas.width || !canvas.height) {
                            canvas.width = img.width || 1280;
                            canvas.height = img.height || 720;
                        }

                        // Clear and draw new frame
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                        // Cleanup
                        if (frameData instanceof Blob) {
                            URL.revokeObjectURL(url);
                        }
                        resolve();
                    } catch (err) {
                        reject(err);
                    }
                };
                
                img.onerror = () => {
                    if (frameData instanceof Blob) {
                        URL.revokeObjectURL(url);
                    }
                    reject(new Error('Failed to load frame'));
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
            if (!frameData) return;
            
            try {
                // Convert ArrayBuffer to Blob if needed
                const frame = frameData instanceof ArrayBuffer ? 
                    new Blob([frameData], { type: 'image/jpeg' }) : 
                    frameData;
                    
                await drawFrame(frame);
            } catch (error) {
                console.error('Error drawing frame:', error);
            }
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
        websocketService.on('edge_frame', (frameData) => {
            if (frameData) {
                handleFrame(frameData);
            }
        });
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

    // Initialize canvas and handle resizing
    useEffect(() => {
        if (!containerRef.current || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const container = containerRef.current;

        // Set initial canvas size
        canvas.width = 1280;  // Default width
        canvas.height = 720;  // Default height

        const resizeObserver = new ResizeObserver(() => {
            // Maintain aspect ratio while fitting container
            const containerWidth = container.offsetWidth;
            const containerHeight = container.offsetHeight;
            const scale = Math.min(
                containerWidth / canvas.width,
                containerHeight / canvas.height
            );

            canvas.style.width = `${canvas.width * scale}px`;
            canvas.style.height = `${canvas.height * scale}px`;
        });

        resizeObserver.observe(container);
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
