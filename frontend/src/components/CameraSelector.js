import React from 'react';
import { Box, Select, MenuItem, Button, Typography } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import RefreshIcon from '@mui/icons-material/Refresh';

const CameraSelector = ({
    devices,
    selectedDevice,
    setSelectedDevice,
    isStreaming,
    startCamera,
    stopCamera,
    refreshDevices
}) => {
    const handleDeviceChange = (event) => {
        const deviceId = event.target.value;
        setSelectedDevice(deviceId);
        if (isStreaming) {
            stopCamera();
        }
    };

    return (
        <Box sx={{ mb: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
                Camera Selection
            </Typography>
            <Select
                fullWidth
                value={selectedDevice || ''}
                onChange={handleDeviceChange}
                disabled={isStreaming}
                sx={{ mb: 2 }}
            >
                <MenuItem value="">Select a camera</MenuItem>
                {devices.map((device) => (
                    <MenuItem key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${device.deviceId.substr(0, 5)}`}
                    </MenuItem>
                ))}
            </Select>
            <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                    variant="contained"
                    startIcon={<VideocamIcon />}
                    onClick={() => startCamera(selectedDevice)}
                    disabled={isStreaming || !selectedDevice}
                    sx={{ flex: 1 }}
                >
                    {isStreaming ? 'Running' : 'Start'}
                </Button>
                <Button
                    variant="outlined"
                    onClick={stopCamera}
                    disabled={!isStreaming}
                    sx={{ flex: 1 }}
                >
                    Stop
                </Button>
                <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={refreshDevices}
                    disabled={isStreaming}
                >
                    Refresh
                </Button>
            </Box>
        </Box>
    );
};

export default CameraSelector;