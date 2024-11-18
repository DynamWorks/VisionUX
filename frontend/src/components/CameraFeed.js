import React, { useRef, useEffect, useState } from 'react';
import { Box, Button, ButtonGroup } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';
import RerunViewer from './RerunViewer';

const CameraFeed = ({
    stream,
    isStreaming,
    videoFile,
    selectedFeature,
    onFrame,
    onPause,
    onResume,
    onStop
}) => {
    const [isPaused, setIsPaused] = useState(false);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        if (videoRef.current) {
            if (isStreaming && stream) {
                videoRef.current.srcObject = stream;
                videoRef.current.src = '';
            } else if (videoFile) {
                videoRef.current.srcObject = null;
                videoRef.current.src = URL.createObjectURL(videoFile);
            } else {
                videoRef.current.srcObject = null;
                videoRef.current.src = '';
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

    const handlePause = () => {
        setIsPaused(true);
        onPause?.();
    };

    const handleResume = () => {
        setIsPaused(false);
        onResume?.();
    };

    const handleStop = () => {
        setIsPaused(false);
        onStop?.();
    };

    return (
        <Box sx={{ width: '100%' }}>
            <RerunViewer 
                stream={stream} 
                isStreaming={isStreaming && !isPaused} 
            />
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                <ButtonGroup variant="contained">
                    {isPaused ? (
                        <Button 
                            onClick={handleResume}
                            disabled={!isStreaming}
                            startIcon={<PlayArrowIcon />}
                        >
                            Resume
                        </Button>
                    ) : (
                        <Button 
                            onClick={handlePause}
                            disabled={!isStreaming}
                            startIcon={<PauseIcon />}
                        >
                            Pause
                        </Button>
                    )}
                    <Button 
                        onClick={handleStop}
                        disabled={!isStreaming}
                        startIcon={<StopIcon />}
                        color="error"
                    >
                        Stop
                    </Button>
                </ButtonGroup>
            </Box>
        </Box>
    );
};

export default CameraFeed;
