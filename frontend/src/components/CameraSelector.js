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
            // Send stop command to WebSocket before stopping camera
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'stop_camera_stream'
                }));
            }
            stopCamera();
        } else {
            if (!selectedDevice) {
                alert('Please select a camera device first');
                return;
            }
            startCamera(selectedDevice);
        }
    };

    return (
        <Box sx={{ mb: 2 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
                <Select
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    disabled={isStreaming}
                    sx={{
                        '& .MuiSelect-select': {
                            whiteSpace: 'normal',
                            minHeight: '1.4375em',
                            textOverflow: 'ellipsis',
                            overflow: 'hidden'
                        }
                    }}
                >
                    {devices.map((device) => (
                        <MenuItem 
                            key={device.deviceId} 
                            value={device.deviceId}
                            sx={{
                                whiteSpace: 'normal',
                                wordBreak: 'break-word'
                            }}
                        >
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
