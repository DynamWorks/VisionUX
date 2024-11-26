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
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });
                
                const video = document.createElement('video');
                video.srcObject = stream;
                video.play();

                const canvas = document.createElement('canvas');
                canvas.width = 1280;
                canvas.height = 720;
                const ctx = canvas.getContext('2d');

                // Start frame capture loop
                const captureFrame = async () => {
                    if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) {
                        stream.getTracks().forEach(track => track.stop());
                        return;
                    }

                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    try {
                        const blob = await new Promise(resolve => {
                            canvas.toBlob(resolve, 'image/jpeg', 0.8);
                        });

                        // Send frame type indicator first
                        const frameMetadata = {
                            type: 'camera_frame',
                            timestamp: Date.now(),
                            width: canvas.width,
                            height: canvas.height
                        };
                        console.log('Sending frame metadata:', frameMetadata);
                        ws.send(JSON.stringify(frameMetadata));
                        
                        // Wait a bit to ensure metadata is processed
                        await new Promise(resolve => setTimeout(resolve, 50));
                        
                        // Then send the actual frame data
                        console.log('Sending binary frame data, size:', blob.size);
                        ws.send(blob);
                        
                        // Wait for frame processed confirmation
                        const confirmation = await new Promise((resolve, reject) => {
                            const timeout = setTimeout(() => {
                                reject(new Error('Frame confirmation timeout'));
                            }, 5000);
                            
                            const handler = (event) => {
                                try {
                                    const data = JSON.parse(event.data);
                                    if (data.type === 'frame_processed') {
                                        ws.removeEventListener('message', handler);
                                        clearTimeout(timeout);
                                        resolve(data);
                                    }
                                } catch (e) {
                                    console.warn('Non-JSON message received:', event.data);
                                }
                            };
                            
                            ws.addEventListener('message', handler);
                        });
                        
                        console.log('Frame processed:', confirmation);
                        requestAnimationFrame(captureFrame);
                    } catch (error) {
                        console.error('Error sending frame:', error);
                        if (error.message !== 'Frame confirmation timeout') {
                            requestAnimationFrame(captureFrame);
                        }
                    }
                };

                video.onloadedmetadata = () => {
                    startCamera(selectedDevice);
                    requestAnimationFrame(captureFrame);
                };
            } catch (error) {
                console.error('Error accessing camera:', error);
                alert('Failed to access camera: ' + error.message);
            }
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
