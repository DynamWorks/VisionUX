import React, { useState, useCallback, useEffect } from 'react';
import { Box, IconButton, Typography, useTheme, useMediaQuery, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import VideocamIcon from '@mui/icons-material/Videocam';
import useStore from '../store';
import { websocketService } from '../services/websocket';

const CameraSelector = () => {
    const { isStreaming, setIsStreaming } = useStore();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const [error, setError] = useState(null);
    const [devices, setDevices] = useState([]);
    const [selectedDevice, setSelectedDevice] = useState('');

    useEffect(() => {
        // Get available camera devices when component mounts
        const getDevices = async () => {
            try {
                const mediaDevices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = mediaDevices.filter(device => device.kind === 'videoinput');
                setDevices(videoDevices);
                if (videoDevices.length > 0) {
                    setSelectedDevice(videoDevices[0].deviceId);
                }
            } catch (err) {
                console.error('Error getting devices:', err);
                setError('Failed to get camera devices');
            }
        };

        getDevices();
    }, []);

    const handleStartStop = useCallback(async () => {
        if (isStreaming) {
            websocketService.emit('stop_stream');
            setIsStreaming(false);
        } else {
            try {
                // Request camera access with specific device
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: selectedDevice ? { exact: selectedDevice } : undefined
                    }
                });
                
                // Stop the stream immediately - we just needed permission
                stream.getTracks().forEach(track => track.stop());

                websocketService.emit('start_stream', { deviceId: selectedDevice });
                setIsStreaming(true);
                setError(null);
            } catch (error) {
                console.error('Camera error:', error);
                setError(error.message);
                setIsStreaming(false);
            }
        }
    }, [isStreaming, setIsStreaming, selectedDevice]);

    const handleDeviceChange = (event) => {
        setSelectedDevice(event.target.value);
        if (isStreaming) {
            // Stop current stream before switching
            websocketService.emit('stop_stream');
            setIsStreaming(false);
        }
    };

    const buttonSize = isMobile ? 36 : 42;
    const iconSize = isMobile ? 20 : 24;

    const buttonSize = isMobile ? 36 : 42;
    const iconSize = isMobile ? 20 : 24;

    return (
        <Box sx={{ mb: 2 }}>
            <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2 
            }}>
                <Box sx={{ 
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1
                }}>
                    <VideocamIcon sx={{ color: isStreaming ? '#2e7d32' : 'text.secondary' }} />
                    <Typography variant="body2" color={isStreaming ? 'primary' : 'text.secondary'}>
                        {isStreaming ? 'Camera Active' : 'Camera Inactive'}
                    </Typography>
                </Box>

                <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
                    <InputLabel>Camera Device</InputLabel>
                    <Select
                        value={selectedDevice}
                        onChange={handleDeviceChange}
                        label="Camera Device"
                        disabled={isStreaming}
                    >
                        {devices.map((device) => (
                            <MenuItem key={device.deviceId} value={device.deviceId}>
                                {device.label || `Camera ${device.deviceId.slice(0, 5)}...`}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>

                <IconButton
                    onClick={handleStartStop}
                    disabled={!selectedDevice}
                    sx={{
                        width: buttonSize,
                        height: buttonSize,
                        bgcolor: isStreaming ? '#d32f2f' : '#2e7d32',
                        color: 'white',
                        '&:hover': {
                            bgcolor: isStreaming ? '#9a0007' : '#1b5e20'
                        },
                        '&.Mui-disabled': {
                            bgcolor: '#666666'
                        }
                    }}
                >
                    {isStreaming ? 
                        <StopIcon sx={{ fontSize: iconSize }} /> : 
                        <PlayArrowIcon sx={{ fontSize: iconSize }} />
                    }
                </IconButton>

                {error && (
                    <Typography color="error" variant="body2" align="center">
                        {error}
                    </Typography>
                )}
            </Box>
        </Box>
    );
};

export default CameraSelector;
