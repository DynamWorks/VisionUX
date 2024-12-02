import React from 'react';
import { Box, ToggleButtonGroup, ToggleButton, useTheme, useMediaQuery } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideoFileIcon from '@mui/icons-material/VideoFile';
import useStore from '../store';

const InputSelector = () => {
    const { inputMode, setInputMode } = useStore();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

    const handleModeChange = (event, newMode) => {
        if (newMode !== null) {
            setInputMode(newMode);
        }
    };

    return (
        <Box sx={{ mb: 2 }}>
            <ToggleButtonGroup
                value={inputMode}
                exclusive
                onChange={handleModeChange}
                aria-label="input mode"
                fullWidth
                size={isMobile ? 'small' : 'medium'}
                sx={{
                    bgcolor: '#1a1a1a',
                    '& .MuiToggleButton-root': {
                        color: 'rgba(255, 255, 255, 0.7)',
                        '&.Mui-selected': {
                            color: 'white',
                            bgcolor: '#2e7d32',
                            '&:hover': {
                                bgcolor: '#1b5e20'
                            }
                        },
                        '&:hover': {
                            bgcolor: 'rgba(46, 125, 50, 0.1)'
                        }
                    }
                }}
            >
                <ToggleButton 
                    value="camera" 
                    aria-label="camera"
                    sx={{
                        display: 'flex',
                        gap: 1,
                        py: 1.5
                    }}
                >
                    <VideocamIcon />
                    Camera
                </ToggleButton>
                <ToggleButton 
                    value="upload" 
                    aria-label="upload"
                    sx={{
                        display: 'flex',
                        gap: 1,
                        py: 1.5
                    }}
                >
                    <VideoFileIcon />
                    Upload
                </ToggleButton>
            </ToggleButtonGroup>
        </Box>
    );
};

export default InputSelector;
