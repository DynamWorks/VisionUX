import React from 'react';
import { Box, Button, FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import { PlayArrow, Stop, Refresh } from '@mui/icons-material';

const CameraSelector = ({ 
    devices, 
    selectedDevice, 
    setSelectedDevice, 
    isStreaming,
    startCamera,
    stopCamera,
    refreshDevices,
    ws
}) => {
    const handleStartStop = async () => {
        if (isStreaming) {
            stopCamera();
        } else {
            startCamera(selectedDevice);
        }
    };

    return (
        <Box sx={{ mb: 2 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Camera</InputLabel>
                <Select
                    value={selectedDevice}
                    label="Camera"
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    disabled={isStreaming}
                >
                    {devices.map((device) => (
                        <MenuItem key={device.deviceId} value={device.deviceId}>
                            {device.label || `Camera ${device.deviceId}`}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                    variant="contained"
                    onClick={handleStartStop}
                    startIcon={isStreaming ? <Stop /> : <PlayArrow />}
                    sx={{
                        bgcolor: isStreaming ? '#d32f2f' : '#2e7d32',
                        '&:hover': {
                            bgcolor: isStreaming ? '#9a0007' : '#1b5e20'
                        }
                    }}
                >
                    {isStreaming ? 'Stop' : 'Start'}
                </Button>
                <Button
                    variant="outlined"
                    onClick={refreshDevices}
                    startIcon={<Refresh />}
                    disabled={isStreaming}
                >
                    Refresh
                </Button>
            </Box>
        </Box>
    );
};

export default CameraSelector;
