import React from 'react';
import { Box, ToggleButtonGroup, ToggleButton, Typography } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideoFileIcon from '@mui/icons-material/VideoFile';

const InputSelector = ({ inputType, setInputType }) => {
    return (
        <Box sx={{ mb: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
                Select Input Source
            </Typography>
            <ToggleButtonGroup
                value={inputType}
                exclusive
                onChange={(e, newValue) => {
                    if (newValue !== null) {
                        setInputType(newValue);
                    }
                }}
                fullWidth
            >
                <ToggleButton value="camera" aria-label="camera">
                    <VideocamIcon sx={{ mr: 1 }} />
                    Camera
                </ToggleButton>
                <ToggleButton value="upload" aria-label="upload">
                    <VideoFileIcon sx={{ mr: 1 }} />
                    Upload Video
                </ToggleButton>
            </ToggleButtonGroup>
        </Box>
    );
};

export default InputSelector;
