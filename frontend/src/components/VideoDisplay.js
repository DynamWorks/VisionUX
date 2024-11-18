import React, { useRef, useEffect } from 'react';
import { Box } from '@mui/material';

const VideoDisplay = ({ stream, videoFile }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        if (videoRef.current) {
            if (stream) {
                videoRef.current.srcObject = stream;
            } else if (videoFile) {
                videoRef.current.src = URL.createObjectURL(videoFile);
            }
        }
    }, [stream, videoFile]);

    return (
        <Box sx={{ 
            position: 'relative',
            width: '100%',
            height: '500px',
            backgroundColor: '#000',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                }}
            />
            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none'
                }}
            />
        </Box>
    );
};

export default VideoDisplay;
