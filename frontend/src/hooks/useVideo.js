import { useState, useCallback } from 'react';

const useVideo = () => {
    const [videoUrl, setVideoUrl] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [videoElement, setVideoElement] = useState(null);

    const handleFileUpload = useCallback((file) => {
        const url = URL.createObjectURL(file);
        setVideoUrl(url);
        return url;
    }, []);

    const startVideo = useCallback(() => {
        if (videoElement && videoUrl) {
            videoElement.play();
            setIsPlaying(true);
        }
    }, [videoElement, videoUrl]);

    const pauseVideo = useCallback(() => {
        if (videoElement) {
            videoElement.pause();
            setIsPlaying(false);
        }
    }, [videoElement]);

    const stopVideo = useCallback(() => {
        if (videoElement) {
            videoElement.pause();
            videoElement.currentTime = 0;
            setIsPlaying(false);
        }
    }, [videoElement]);

    return {
        videoUrl,
        isPlaying,
        setVideoElement,
        handleFileUpload,
        startVideo,
        pauseVideo,
        stopVideo
    };
};

export default useVideo;
