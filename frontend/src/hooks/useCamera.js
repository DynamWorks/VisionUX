import { useState, useEffect, useCallback } from 'react';

const useCamera = () => {
    const [devices, setDevices] = useState([]);
    const [selectedDevice, setSelectedDevice] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [stream, setStream] = useState(null);

    const getDevices = useCallback(async () => {
        try {
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

    const startCamera = useCallback(async (deviceId) => {
        try {
            const constraints = {
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            setStream(mediaStream);
            setIsStreaming(true);
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
            setIsStreaming(false);
        }
    }, [stream]);

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
