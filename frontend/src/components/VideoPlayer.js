import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Box, IconButton, LinearProgress, Typography } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';
import ReplayIcon from '@mui/icons-material/Replay';
import { websocketService } from '../services/websocket';
import useStore from '../store';

const VideoPlayer = ({ file, visualizationPath }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const animationFrameRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState(null);
    const { 
        setIsStreaming, 
        setStreamMetrics,
        showEdgeVisualization,
        showObjectVisualization,
        currentVisualization
    } = useStore();
    const lastFrameTimeRef = useRef(Date.now());
    const frameCountRef = useRef(0);

    const updateStreamMetrics = useCallback(() => {
        const now = Date.now();
        const elapsed = now - lastFrameTimeRef.current;

        if (elapsed >= 1000) {
            const fps = Math.round((frameCountRef.current * 1000) / elapsed);
            setStreamMetrics({
                fps,
                frameCount: frameCountRef.current,
                timestamp: now,
                resolution: videoRef.current ?
                    `${videoRef.current.videoWidth}x${videoRef.current.videoHeight}` :
                    undefined,
                duration: videoRef.current?.duration,
                currentTime: videoRef.current?.currentTime
            });

            frameCountRef.current = 0;
            lastFrameTimeRef.current = now;
        }

        frameCountRef.current++;
    }, [setStreamMetrics]);

    const startVideoProcessing = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current) return;

        try {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            // Set canvas size to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const processFrame = () => {
                if (!video.paused && !video.ended) {
                    // Draw frame to canvas
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    // Convert to JPEG and send via WebSocket
                    canvas.toBlob(
                        (blob) => {
                            if (websocketService.isConnected()) {
                                websocketService.emit('frame', blob);

                                websocketService.emit('frame_metadata', {
                                    timestamp: Date.now(),
                                    width: canvas.width,
                                    height: canvas.height,
                                    currentTime: video.currentTime,
                                    duration: video.duration,
                                    filename: file.name
                                });

                                updateStreamMetrics();
                            }
                        },
                        'image/jpeg',
                        0.85
                    );

                    setProgress((video.currentTime / video.duration) * 100);
                    animationFrameRef.current = requestAnimationFrame(processFrame);
                } else if (video.ended) {
                    setIsPlaying(false);
                    setIsStreaming(false);
                    websocketService.emit('stream_ended', { filename: file.name });
                }
            };

            // Start processing frames
            animationFrameRef.current = requestAnimationFrame(processFrame);
            setIsStreaming(true);

        } catch (err) {
            console.error('Error processing video:', err);
            setError('Error processing video frames');
            setIsPlaying(false);
            setIsStreaming(false);
        }
    }, [file, setIsStreaming, updateStreamMetrics]);

    const stopVideoProcessing = useCallback(() => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.pause();
            videoRef.current.currentTime = 0;
        }

        setIsPlaying(false);
        setIsStreaming(false);
        setProgress(0);
        frameCountRef.current = 0;
        websocketService.emit('stop_stream');
    }, [setIsStreaming]);

    // Handle video loading
    useEffect(() => {
        if (!file || !file.name) {
            setError('Invalid file');
            setIsLoading(false);
            return;
        }

        const loadVideo = async () => {
            try {
                // Determine which video to load based on visualization toggles
                let videoPath;
                if (showEdgeVisualization && currentVisualization) {
                    videoPath = currentVisualization.replace(/^tmp_content\//, '');
                } else if (showObjectVisualization && currentVisualization) {
                    videoPath = currentVisualization.replace(/^tmp_content\//, '');
                } else {
                    videoPath = `uploads/${file.name}`;
                }
                
                console.log('Loading video from:', videoPath);
            
                // Add cache buster to prevent browser caching
                const cacheBuster = `?t=${Date.now()}`;
                const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/tmp_content/${videoPath}${cacheBuster}`);
            
                if (!response.ok) throw new Error('Failed to load video');
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                if (videoRef.current) {
                    videoRef.current.src = url;
                    videoRef.current.load(); // Force reload of video
                    setIsLoading(true);
                    setError(null);

                    videoRef.current.onloadedmetadata = () => {
                        setIsLoading(false);
                        const currentTime = videoRef.current.currentTime;
                        
                        // Preserve playback state when switching
                        if (isPlaying) {
                            videoRef.current.play();
                            startVideoProcessing();
                        }
                    };

                    videoRef.current.onerror = () => {
                        setError('Error loading video file');
                        setIsLoading(false);
                    };
                }

                // Cleanup function for URL
                return () => {
                    URL.revokeObjectURL(url);
                };
            } catch (err) {
                console.error('Error loading video:', err);
                setError(err.message);
                setIsLoading(false);
                return null;
            }
        };

        let videoUrl;
        loadVideo().then(url => {
            videoUrl = url;
        });

        return () => {
            if (videoUrl) {
                URL.revokeObjectURL(videoUrl);
            }
        };
    }, [file, showEdgeVisualization, currentVisualization]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopVideoProcessing();
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [stopVideoProcessing]);

    const handlePlay = async () => {
        if (!videoRef.current) return;

        try {
            await videoRef.current.play();
            setIsPlaying(true);
            startVideoProcessing();
        } catch (err) {
            console.error('Error playing video:', err);
            setError('Error playing video');
        }
    };

    const handlePause = () => {
        if (!videoRef.current) return;

        videoRef.current.pause();
        setIsPlaying(false);

        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
    };

    const handleStop = () => {
        stopVideoProcessing();
    };

    const handleRestart = () => {
        if (!videoRef.current) return;

        videoRef.current.currentTime = 0;
        handlePlay();
    };

    return (
        <Box sx={{ width: '100%', position: 'relative' }}>
            <video
                ref={videoRef}
                style={{ display: 'none' }}
                playsInline
                muted
                loop
            />
            <canvas
                ref={canvasRef}
                style={{
                    display: 'block',
                    width: '100%',
                    height: 'auto',
                    backgroundColor: '#000'
                }}
            />

            {error && (
                <Typography
                    color="error"
                    sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        bgcolor: 'rgba(0,0,0,0.7)',
                        padding: 2,
                        borderRadius: 1
                    }}
                >
                    {error}
                </Typography>
            )}

            {isLoading && (
                <LinearProgress
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%'
                    }}
                />
            )}

            <Box
                sx={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    bgcolor: 'rgba(0,0,0,0.7)',
                    padding: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1
                }}
            >
                <IconButton
                    onClick={isPlaying ? handlePause : handlePlay}
                    disabled={isLoading || !!error}
                    sx={{ color: 'white' }}
                >
                    {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                </IconButton>

                <IconButton
                    onClick={handleStop}
                    disabled={isLoading || !!error || !isPlaying}
                    sx={{ color: 'white' }}
                >
                    <StopIcon />
                </IconButton>

                <IconButton
                    onClick={handleRestart}
                    disabled={isLoading || !!error}
                    sx={{ color: 'white' }}
                >
                    <ReplayIcon />
                </IconButton>

                <LinearProgress
                    variant="determinate"
                    value={progress}
                    sx={{
                        flex: 1,
                        mx: 2,
                        '& .MuiLinearProgress-bar': {
                            transition: 'none'
                        }
                    }}
                />
            </Box>
        </Box>
    );
};

export default VideoPlayer;
