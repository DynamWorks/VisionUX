import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import useVideoStore from '../stores/videoStore';

const CustomViewer = ({ websocket }) => {
    const canvasRef = useRef(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [fps, setFps] = useState(0);
    const frameCountRef = useRef(0);
    const lastFrameTimeRef = useRef(Date.now());
    const { setStreamMetrics } = useVideoStore();

    const drawMetadataOverlays = useCallback((ctx, metadata) => {
        if (!metadata) return;

        // Draw object detection boxes
        if (metadata.detections) {
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.font = '12px Arial';
            
            metadata.detections.forEach(det => {
                const [x1, y1, x2, y2] = det.bbox;
                ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                
                // Draw label with background
                const label = `${det.class} ${Math.round(det.confidence * 100)}%`;
                const textMetrics = ctx.measureText(label);
                const padding = 2;
                
                ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.fillRect(
                    x1, 
                    y1 - 20, 
                    textMetrics.width + padding * 2, 
                    20
                );
                
                ctx.fillStyle = '#00ff00';
                ctx.fillText(label, x1 + padding, y1 - 5);
            });
        }

        // Draw motion regions
        if (metadata.motion?.regions) {
            ctx.strokeStyle = '#ff0000';
            ctx.lineWidth = 2;
            
            metadata.motion.regions.forEach(region => {
                const [x1, y1, x2, y2] = region.bbox;
                ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                
                // Draw motion intensity indicator
                if (region.intensity) {
                    ctx.fillStyle = `rgba(255, 0, 0, ${region.intensity})`;
                    ctx.fillRect(x1, y1, x2-x1, y2-y1);
                }
            });
        }

        // Draw edge detection overlay if present
        if (metadata.edges) {
            const imageData = new ImageData(
                new Uint8ClampedArray(metadata.edges.data),
                metadata.edges.width,
                metadata.edges.height
            );
            ctx.putImageData(imageData, 0, 0);
        }
    }, []);

    const processFrame = useCallback(async (frameData, metadata = null) => {
        try {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            
            // Create image from frame data
            const blob = new Blob([frameData], { type: 'image/jpeg' });
            const imageUrl = URL.createObjectURL(blob);
            
            const img = new Image();
            img.onload = () => {
                // Resize canvas if needed
                if (canvas.width !== img.width || canvas.height !== img.height) {
                    canvas.width = img.width;
                    canvas.height = img.height;
                }

                // Clear and draw new frame
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);

                // Draw metadata overlays if present
                if (metadata) {
                    drawMetadataOverlays(ctx, metadata);
                }

                // Update FPS counter
                frameCountRef.current++;
                const now = Date.now();
                if (frameCountRef.current % 30 === 0) {
                    const newFps = Math.round(1000 / (now - lastFrameTimeRef.current));
                    setFps(newFps);
                    setStreamMetrics({
                        fps: newFps,
                        frameCount: frameCountRef.current,
                        resolution: `${img.width}x${img.height}`
                    });
                }
                lastFrameTimeRef.current = now;

                URL.revokeObjectURL(imageUrl);
            };
            
            img.src = imageUrl;
            setLoading(false);
            
        } catch (error) {
            console.error('Error processing frame:', error);
            setError(error.message);
        }
    }, [drawMetadataOverlays, setStreamMetrics]);

    useEffect(() => {
        if (!websocket) {
            setError('No WebSocket connection');
            return;
        }

        let frameBuffer = null;
        
        const handleMessage = async (event) => {
            try {
                if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
                    // Store frame data and wait for metadata
                    frameBuffer = event.data;
                } else {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'camera_frame') {
                            // Process frame immediately since metadata came first
                            if (frameBuffer) {
                                await processFrame(frameBuffer, data);
                                frameBuffer = null;
                            }
                        }
                    } catch (e) {
                        // If not valid JSON, process frame without metadata
                        if (frameBuffer) {
                            await processFrame(frameBuffer);
                            frameBuffer = null;
                        }
                    }
                }
            } catch (error) {
                console.error('Error handling message:', error);
                setError(error.message);
            }
        };

        websocket.on('message', handleMessage);

        return () => {
            websocket.off('message', handleMessage);
        };
    }, [websocket, processFrame]);

    return (
        <Box sx={{
            width: '100%',
            height: '600px',
            bgcolor: '#1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            position: 'relative'
        }}>
            <Box sx={{
                position: 'absolute',
                top: 8,
                right: 8,
                bgcolor: 'rgba(0, 0, 0, 0.5)',
                padding: '4px 8px',
                borderRadius: 1,
                zIndex: 1
            }}>
                <Typography variant="caption" sx={{ color: 'white' }}>
                    {fps} FPS
                </Typography>
            </Box>
            
            <Box sx={{
                flex: 1,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center'
            }}>
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
                    <Box sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        color: 'error.main',
                        textAlign: 'center',
                        padding: 2
                    }}>
                        <Typography variant="h6" color="error">
                            Error
                        </Typography>
                        <Typography variant="body2">
                            {error}
                        </Typography>
                    </Box>
                )}
                <canvas
                    ref={canvasRef}
                    style={{
                        maxWidth: '100%',
                        maxHeight: '100%',
                        objectFit: 'contain',
                        display: loading ? 'none' : 'block'
                    }}
                />
            </Box>
        </Box>
    );
};

export default CustomViewer;
