import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { websocketService } from '../services/websocket';
import useStore from '../store';

const StreamRenderer = ({ source, isStreaming }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const { setIsStreaming, setStreamMetrics } = useStore();
    const lastFrameTimeRef = useRef(Date.now());
    const fpsCounterRef = useRef({ frames: 0, lastUpdate: Date.now() });

    const updateMetrics = useCallback(() => {
        const now = Date.now();
        const elapsed = now - fpsCounterRef.current.lastUpdate;

        if (elapsed >= 1000) {
            const fps = Math.round((fpsCounterRef.current.frames * 1000) / elapsed);
            setStreamMetrics(prev => ({
                ...prev,
                fps,
                timestamp: now
            }));

            fpsCounterRef.current = { frames: 0, lastUpdate: now };
        }

        fpsCounterRef.current.frames++;
    }, [setStreamMetrics]);

    const drawFrame = useCallback(async (frameData) => {
        if (!canvasRef.current || !containerRef.current) return;

        try {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            let imageBitmap;

            // Convert frame data to ImageBitmap
            let blob;
            if (frameData instanceof Blob) {
                blob = frameData;
            } else if (frameData instanceof ArrayBuffer || frameData instanceof Uint8Array) {
                blob = new Blob([frameData], { type: 'image/jpeg' });
            } else if (typeof frameData === 'string' && frameData.startsWith('data:')) {
                const response = await fetch(frameData);
                blob = await response.blob();
            } else if (Array.isArray(frameData)) {
                // Handle array data (e.g. from tuple)
                const uint8Array = new Uint8Array(frameData);
                blob = new Blob([uint8Array], { type: 'image/jpeg' });
            } else {
                throw new Error('Invalid frame data format');
            }

            try {
                imageBitmap = await createImageBitmap(blob);
            } catch (err) {
                console.error('Error creating ImageBitmap:', err);
                throw new Error('Failed to create image from frame data');
            }

            // Calculate dimensions preserving aspect ratio
            const containerWidth = containerRef.current.offsetWidth;
            const containerHeight = containerRef.current.offsetHeight;

            const scale = Math.min(
                containerWidth / imageBitmap.width,
                containerHeight / imageBitmap.height
            );

            const width = imageBitmap.width * scale;
            const height = imageBitmap.height * scale;

            // Update canvas size if needed
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }

            // Clear canvas and draw frame
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(imageBitmap, 0, 0, width, height);

            // Cleanup
            imageBitmap.close();

            // Update metrics
            updateMetrics();
            lastFrameTimeRef.current = Date.now();

        } catch (err) {
            console.error('Error drawing frame:', err);
            setError('Error displaying frame');
        }
    }, [updateMetrics]);

    // Handle WebSocket frames
    useEffect(() => {
        if (!isStreaming || source !== 'camera') return;

        const handleFrame = async (frameData) => {
            setLoading(false);
            await drawFrame(frameData);
        };

        websocketService.on('frame', handleFrame);
        setLoading(true);
        setIsStreaming(true);

        return () => {
            websocketService.off('frame', handleFrame);
            setIsStreaming(false);
        };
    }, [isStreaming, source, drawFrame, setIsStreaming]);

    // Handle container resizing
    useEffect(() => {
        if (!containerRef.current || !canvasRef.current) return;

        const resizeObserver = new ResizeObserver(() => {
            if (canvasRef.current) {
                const canvas = canvasRef.current;
                const container = containerRef.current;

                canvas.style.width = '100%';
                canvas.style.height = '100%';
                canvas.width = container.offsetWidth;
                canvas.height = container.offsetHeight;
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
                position: 'relative'
            }}
        >
            <canvas
                ref={canvasRef}
                style={{
                    display: isStreaming ? 'block' : 'none',
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                }}
            />

            {loading && (
                <CircularProgress
                    sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)'
                    }}
                />
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
