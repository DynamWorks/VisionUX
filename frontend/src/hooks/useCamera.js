import { useState, useEffect, useCallback } from 'react';

const useCamera = () => {
    const [devices, setDevices] = useState([]);
    const [selectedDevice, setSelectedDevice] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [stream, setStream] = useState(null);

    const getDevices = useCallback(async () => {
        try {
            // We need to call getUserMedia first to trigger the browser's permission request
            await navigator.mediaDevices.getUserMedia({ video: true });
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            setDevices(videoDevices);
            if (videoDevices.length > 0 && !selectedDevice) {
                setSelectedDevice(videoDevices[0].deviceId);
            }
        } catch (error) {
            console.error('Error enumerating devices:', error);
        }
    }, [selectedDevice]);

    useEffect(() => {
        getDevices();
    }, [getDevices]);

    const startCamera = useCallback(async () => {
        if (selectedDevice) {
            try {
                const newStream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: selectedDevice }
                });
                setStream(newStream);
                setIsStreaming(true);
            } catch (error) {
                console.error('Error starting camera:', error);
            }
        }
    }, [selectedDevice]);

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
            setIsStreaming(false);
        }
    }, [stream]);

    // Remove the useEffect that automatically starts the camera

    const refreshDevices = useCallback(() => {
        getDevices();
    }, [getDevices]);

    return {
        devices,
        selectedDevice,
        setSelectedDevice,
        isStreaming,
        startCamera,
        stopCamera,
        refreshDevices,
        stream
    };
};

export default useCamera;
