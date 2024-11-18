import React, { useRef, useEffect } from 'react';
import { Box } from '@mui/material';

const CameraFeed = ({
    stream,
    isStreaming,
    selectedFeature,
    onFrame
}) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        if (videoRef.current) {
            if (isStreaming && stream) {
                videoRef.current.srcObject = stream;
            } else {
                videoRef.current.srcObject = null;
            }
        }
    }, [isStreaming, stream]);

    useEffect(() => {
        let animationFrame;
        const processFrame = async () => {
            if (videoRef.current && canvasRef.current && isStreaming) {
                const video = videoRef.current;
                const canvas = canvasRef.current;
                const context = canvas.getContext('2d');

                // Match canvas size to video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Draw video frame to canvas
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Process frame if callback provided
                if (onFrame) {
                    await onFrame(canvas, context);
                }

                // Request next frame
                animationFrame = requestAnimationFrame(processFrame);
            }
        };

        if (isStreaming) {
            processFrame();
        }

        return () => {
            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
            }
        };
    }, [isStreaming, onFrame]);

    return (
        <Box sx={{ position: 'relative', width: '100%', bgcolor: 'black', borderRadius: 1, overflow: 'hidden' }}>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: 'auto' }}
            />
            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%'
                }}
            />
        </Box>
    );
};

export default CameraFeed;
