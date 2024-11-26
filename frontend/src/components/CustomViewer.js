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
        if (!canvasRef.current) return;
        
        try {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d', { alpha: false });
            
            // Handle different frame data types
            let img;
            if (frameData instanceof Blob) {
                img = await createImageFromBlob(frameData);
            } else if (frameData instanceof ArrayBuffer) {
                const blob = new Blob([frameData], { type: 'image/jpeg' });
                img = await createImageFromBlob(blob);
            } else {
                throw new Error('Invalid frame data type');
            }

            // Initialize canvas size on first frame
            if (canvas.width !== img.width || canvas.height !== img.height) {
                canvas.width = img.width;
                canvas.height = img.height;
                // Enable image smoothing for better quality
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
            }

            // Clear previous frame
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw new frame with error handling
            try {
                ctx.drawImage(img, 0, 0);
            } catch (e) {
                console.error('Error drawing frame:', e);
                return;
            }

            // Process metadata overlays
            if (metadata) {
                drawMetadataOverlays(ctx, metadata);
            }

            // Update metrics
            updateMetrics(img.width, img.height);
            setLoading(false);
            setError(null);

        } catch (error) {
            console.error('Frame processing error:', error);
            setError(`Frame processing failed: ${error.message}`);
            setLoading(false);
        }
    }, [drawMetadataOverlays, setStreamMetrics]);

    // Helper function to create Image from Blob
    const createImageFromBlob = (blob) => {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(blob);
            const img = new Image();
            
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };
            
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to load image'));
            };
            
            img.src = url;
        });
    };

    // Update performance metrics
    const updateMetrics = useCallback((width, height) => {
        frameCountRef.current++;
        const now = Date.now();
        
        if (frameCountRef.current % 30 === 0) {
            const timeDiff = now - lastFrameTimeRef.current;
            if (timeDiff > 0) {
                const newFps = Math.round(1000 / timeDiff);
                setFps(newFps);
                setStreamMetrics({
                    fps: newFps,
                    frameCount: frameCountRef.current,
                    resolution: `${width}x${height}`,
                    timestamp: now
                });
            }
        }
        lastFrameTimeRef.current = now;
    }, [setStreamMetrics]);

    useEffect(() => {
        if (!websocket) {
            setError('WebSocket not connected');
            return;
        }

        let frameQueue = [];
        let processingFrame = false;

        const processNextFrame = async () => {
            if (processingFrame || frameQueue.length === 0) return;
            
            processingFrame = true;
            const { frameData, metadata } = frameQueue.shift();
            
            try {
                await processFrame(frameData, metadata);
            } catch (error) {
                console.error('Error processing frame:', error);
            } finally {
                processingFrame = false;
                // Process next frame if available
                if (frameQueue.length > 0) {
                    requestAnimationFrame(processNextFrame);
                }
            }
        };

        const handleMessage = (event) => {
            try {
                if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
                    frameQueue.push({ frameData: event.data, metadata: null });
                } else {
                    const data = JSON.parse(event.data);
                    if (data.type === 'camera_frame' && frameQueue.length > 0) {
                        frameQueue[frameQueue.length - 1].metadata = data;
                    }
                }
                
                if (!processingFrame) {
                    requestAnimationFrame(processNextFrame);
                }
            } catch (error) {
                console.error('Message handling error:', error);
                setError(`Message handling failed: ${error.message}`);
            }
        };

        websocket.on('message', handleMessage);
        
        return () => {
            websocket.off('message', handleMessage);
            frameQueue = [];
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
