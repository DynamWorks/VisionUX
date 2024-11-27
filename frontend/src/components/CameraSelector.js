import React, { useState, useCallback } from 'react';
import { Box, IconButton, Typography, useTheme, useMediaQuery } from '@mui/material';
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

    const handleStartStop = useCallback(async () => {
        if (isStreaming) {
            websocketService.emit('stop_stream');
            setIsStreaming(false);
        } else {
            try {
                const hasAccess = await websocketService.requestCameraAccess();
                if (!hasAccess) {
                    throw new Error('Camera access denied');
                }

                websocketService.emit('start_stream');
                setIsStreaming(true);
                setError(null);
            } catch (error) {
                console.error('Camera error:', error);
                setError(error.message);
                setIsStreaming(false);
            }
        }
    }, [isStreaming, setIsStreaming]);

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

                <IconButton
                    onClick={handleStartStop}
                    sx={{
                        width: buttonSize,
                        height: buttonSize,
                        bgcolor: isStreaming ? '#d32f2f' : '#2e7d32',
                        color: 'white',
                        '&:hover': {
                            bgcolor: isStreaming ? '#9a0007' : '#1b5e20'
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
