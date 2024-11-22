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
    refreshDevices,
    ws
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
                    onClick={() => {
                        if (ws?.readyState === WebSocket.OPEN) {
                            // First stop any playing video
                            ws.send(JSON.stringify({
                                type: 'stop_video_stream'
                            }));
                            // Then start camera stream
                            ws.send(JSON.stringify({
                                type: 'start_camera_stream',
                                deviceId: selectedDevice
                            }));
                        }
                        startCamera(selectedDevice);
                    }}
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
            // Send stop message to WebSocket
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'stop_camera_stream' }));
            }
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
