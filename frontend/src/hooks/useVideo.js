import { useState, useCallback, useEffect } from 'react';
import { websocketService } from '../services/websocket';
import useStore from '../store';

const useVideo = () => {
    const [videoUrl, setVideoUrl] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [videoElement, setVideoElement] = useState(null);
    const { setIsStreaming, setCurrentVideo } = useStore();

    // Handle camera stream
    const startCameraStream = useCallback(async (deviceId) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });

            const video = document.createElement('video');
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = 1280;
            canvas.height = 720;
            const ctx = canvas.getContext('2d');

            setIsPlaying(true);
            setIsStreaming(true);

            // Start frame capture loop
            const captureFrame = async () => {
                if (!isPlaying || !websocketService.socket?.connected) {
                    stream.getTracks().forEach(track => track.stop());
                    return;
                }

                try {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Convert to blob and emit
                    const blob = await new Promise(resolve => {
                        canvas.toBlob(resolve, 'image/jpeg', 0.85);
                    });

                    const arrayBuffer = await blob.arrayBuffer();
                    
                    websocketService.emit('frame_metadata', {
                        timestamp: Date.now(),
                        width: canvas.width,
                        height: canvas.height
                    });
                    
                    websocketService.emit('frame', arrayBuffer);

                    requestAnimationFrame(captureFrame);
                } catch (error) {
                    console.error('Error capturing frame:', error);
                    if (error.message !== 'WebSocket not connected') {
                        requestAnimationFrame(captureFrame);
                    }
                }
            };

            requestAnimationFrame(captureFrame);
            return stream;

        } catch (error) {
            console.error('Error accessing camera:', error);
            throw error;
        }
    }, [isPlaying, setIsStreaming]);

    // Handle video file
    const handleFileUpload = useCallback(async (file) => {
        try {
            const url = URL.createObjectURL(file);
            setVideoUrl(url);
            setCurrentVideo(file);

            // Upload file to server
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('Upload successful:', result);

            // Start streaming via API
            const streamResponse = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/stream/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: file.name
                })
            });

            if (!streamResponse.ok) {
                throw new Error(`Failed to start stream: ${streamResponse.statusText}`);
            }

            setIsPlaying(true);
            setIsStreaming(true);
            return url;

        } catch (error) {
            console.error('Error handling file:', error);
            throw error;
        }
    }, [setCurrentVideo, setIsStreaming]);

    const stopVideo = useCallback(async () => {
        setIsPlaying(false);
        setIsStreaming(false);

        // Stop all camera tracks
        if (videoElement?.srcObject instanceof MediaStream) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }

        if (videoUrl) {
            URL.revokeObjectURL(videoUrl);
            setVideoUrl(null);
        }

        if (videoElement) {
            videoElement.pause();
            videoElement.currentTime = 0;
        }

        // Stop stream on server
        try {
            await fetch(`${process.env.REACT_APP_API_URL}/api/v1/stream/stop`, {
                method: 'POST'
            });
        } catch (error) {
            console.error('Error stopping stream:', error);
        }
    }, [videoUrl, videoElement]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (videoUrl) {
                URL.revokeObjectURL(videoUrl);
            }
        };
    }, [videoUrl]);

    return {
        videoUrl,
        isPlaying,
        setVideoElement,
        handleFileUpload,
        startCameraStream,
        stopVideo,
        setIsPlaying
    };
};

export default useVideo;
