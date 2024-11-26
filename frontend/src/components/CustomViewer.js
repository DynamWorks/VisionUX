import React, { useRef, useEffect, useState } from 'react';
import { Box, CircularProgress } from '@mui/material';

const CustomViewer = ({ websocket }) => {
    const canvasRef = useRef(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!websocket) {
            setError('No WebSocket connection');
            return;
        }

        let frameCount = 0;
        const startTime = Date.now();
        let lastFrameTime = startTime;

        const handleFrame = async (event) => {
            try {
                // Check if the message is binary data
                if (event instanceof Blob) {
                    setLoading(false);
                    const binaryData = event;

                    if (!binaryData) return;

                    // Create blob and image
                    const blob = new Blob([binaryData], { type: 'image/jpeg' });
                    const imageUrl = URL.createObjectURL(blob);
                    
                    const img = new Image();
                    img.onload = () => {
                        const canvas = canvasRef.current;
                        if (!canvas) return;

                        const ctx = canvas.getContext('2d');
                        
                        // Resize canvas if needed
                        if (canvas.width !== img.width || canvas.height !== img.height) {
                            canvas.width = img.width;
                            canvas.height = img.height;
                        }

                        // Clear and draw new frame
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);

                        // Draw any metadata overlays
                        if (event instanceof Blob) {
                            // Wait for next message which should contain metadata
                            websocket.once('message', (metadataEvent) => {
                                try {
                                    const metadata = JSON.parse(metadataEvent);
                                    if (metadata && metadata.metadata) {
                                        drawMetadataOverlays(ctx, metadata.metadata);
                                    }
                                } catch (error) {
                                    console.warn('Non-JSON metadata message received');
                                }
                            }, { once: true }); // Only listen once for metadata
                        }

                        // Calculate and log FPS every 30 frames
                        frameCount++;
                        if (frameCount % 30 === 0) {
                            const now = Date.now();
                            const fps = Math.round(1000 / (now - lastFrameTime));
                            console.log(`Streaming at ${fps} FPS`);
                            lastFrameTime = now;
                        }

                        URL.revokeObjectURL(imageUrl);
                    };
                    
                    img.src = imageUrl;
                }
            } catch (error) {
                console.error('Error processing frame:', error);
                setError(error.message);
            }
        };

        const drawMetadataOverlays = (ctx, metadata) => {
            // Draw object detection boxes
            if (metadata.detections) {
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.font = '12px Arial';
                
                metadata.detections.forEach(det => {
                    const [x1, y1, x2, y2] = det.bbox;
                    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                    
                    // Draw label
                    ctx.fillStyle = '#00ff00';
                    ctx.fillText(
                        `${det.class} ${Math.round(det.confidence * 100)}%`,
                        x1, y1 - 5
                    );
                });
            }

            // Draw motion regions
            if (metadata.motion?.regions) {
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 2;
                
                metadata.motion.regions.forEach(region => {
                    const [x1, y1, x2, y2] = region.bbox;
                    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                });
            }
        };

        websocket.on('message', handleFrame);

        return () => {
            websocket.off('message', handleFrame);
        };
    }, [websocket]);

    return (
        <Box sx={{
            width: '100%',
            height: '600px',
            bgcolor: '#1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden',
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
                    color: 'error.main'
                }}>
                    Error: {error}
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
    );
};

export default CustomViewer;
