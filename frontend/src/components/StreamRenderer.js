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
    const overlayCanvasRef = useRef(null);
    
    const drawFrame = useCallback(async (frameData, metadata) => {
        if (!canvasRef.current || !containerRef.current) return;

        try {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            const overlayCanvas = overlayCanvasRef.current;
            const overlayCtx = overlayCanvas?.getContext('2d');

            // Handle different types of frame data
            if (metadata?.type === 'drawing') {
                if (!overlayCtx) return;
                
                // Clear previous overlay
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                
                // Draw new overlay
                const drawingBlob = new Blob([frameData], { type: 'image/png' });
                const drawingUrl = URL.createObjectURL(drawingBlob);
                const drawingImg = new Image();
                
                await new Promise((resolve, reject) => {
                    drawingImg.onload = resolve;
                    drawingImg.onerror = reject;
                    drawingImg.src = drawingUrl;
                });
                
                overlayCtx.drawImage(drawingImg, 0, 0);
                URL.revokeObjectURL(drawingUrl);
                return;
            }

            // Handle regular video frame
            if (!frameData || frameData.size === 0) {
                console.warn('Empty frame data received');
                return;
            }

            // Convert binary frame data to Blob with validation
            let blob;
            try {
                blob = new Blob([frameData], { type: 'image/jpeg' });
                if (blob.size === 0) {
                    console.warn('Empty blob created from frame data');
                    return;
                }
            } catch (error) {
                console.error('Error creating blob:', error);
                return;
            }

            const url = URL.createObjectURL(blob);
            console.debug('Created URL from binary frame data:', url, 'blob size:', blob.size);

            // Create new image
            let img = new Image();

            await new Promise((resolve, reject) => {
                img.onload = () => {
                    try {
                        console.debug('Image loaded:', img.width, 'x', img.height);
                        
                        // Set canvas dimensions if needed
                        if (canvas.width !== 1280 || canvas.height !== 720) {
                            canvas.width = 1280;
                            canvas.height = 720;
                            console.debug('Canvas size set to:', canvas.width, 'x', canvas.height);
                        }

                        // Clear canvas
                        ctx.fillStyle = '#000000';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);

                        // Calculate scaling to maintain aspect ratio
                        const scale = Math.min(
                            canvas.width / img.width,
                            canvas.height / img.height
                        );
                        const x = (canvas.width - img.width * scale) / 2;
                        const y = (canvas.height - img.height * scale) / 2;

                        // Draw frame
                        ctx.drawImage(
                            img,
                            x, y,
                            img.width * scale,
                            img.height * scale
                        );

                        // Cleanup
                        if (frameData instanceof Blob) {
                            URL.revokeObjectURL(url);
                        }
                        resolve();
                    } catch (err) {
                        console.error('Draw error:', err);
                        reject(err);
                    }
                };
                
                img.onerror = (err) => {
                    console.error('Image load error:', err);
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
            setError(`Error displaying frame: ${err.message}`);
        }
    }, []);


    // Setup camera stream
    useEffect(() => {
        if (!isStreaming || source !== 'camera') {
            return;
        }

        let videoStream = null;
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });
                
                videoStream = stream;
                const video = document.createElement('video');
                video.srcObject = stream;
                video.play();

                const captureFrame = () => {
                    if (!canvasRef.current || !isStreaming) return;
                    
                    const canvas = canvasRef.current;
                    const ctx = canvas.getContext('2d');
                    
                    // Set canvas size if needed
                    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                    }

                    // Draw video frame to canvas
                    ctx.drawImage(video, 0, 0);

                    // For analysis, periodically send frame to backend
                    if (Math.random() < 0.1) { // ~10% of frames
                        canvas.toBlob(blob => {
                            if (websocketService.isConnected()) {
                                websocketService.emit('frame', blob);
                            }
                        }, 'image/jpeg', 0.85);
                    }

                    requestAnimationFrame(captureFrame);
                };

                video.onloadedmetadata = () => {
                    setLoading(false);
                    requestAnimationFrame(captureFrame);
                };

            } catch (error) {
                console.error('Error accessing camera:', error);
                setError(error.message);
                setLoading(false);
            }
        };

        setLoading(true);
        startCamera();

        // Cleanup function
        return () => {
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
            }
        };
    }, [isStreaming, source]);

    // Initialize canvas and handle resizing
    useEffect(() => {
        if (!containerRef.current || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        const container = containerRef.current;

        // Set initial canvas sizes
        canvas.width = 1280;
        canvas.height = 720;
        
        if (overlayCanvas) {
            overlayCanvas.width = 1280;
            overlayCanvas.height = 720;
        }

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
            <Box sx={{ position: 'relative' }}>
                <canvas
                    ref={canvasRef}
                    style={{
                        display: isStreaming ? 'block' : 'none',
                        maxWidth: '100%',
                        maxHeight: '100%',
                        objectFit: 'contain',
                        backgroundColor: '#000000'
                    }}
                />
                <canvas
                    ref={overlayCanvasRef}
                    style={{
                        display: isStreaming ? 'block' : 'none',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none'
                    }}
                />
            </Box>

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
