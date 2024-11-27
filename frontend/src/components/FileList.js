import React from 'react';
import { Box, List, ListItem, ListItemIcon, ListItemText, Typography, IconButton } from '@mui/material';
import VideoFileIcon from '@mui/icons-material/VideoFile';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import useStore from '../store';

const FileList = () => {
    const { uploadedFiles = [], currentVideo, setCurrentVideo } = useStore();

    const handleVideoSelect = (file) => {
        setCurrentVideo(file);
    };

    // Empty state
    if (uploadedFiles.length === 0) {
        return (
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                    No files uploaded yet
                </Typography>
            </Box>
        );
    }

    return (
        <Box sx={{ mt: 2 }}>
            <Typography variant="h6" sx={{ mb: 1 }}>
                Uploaded Files
            </Typography>
            <List>
                {uploadedFiles.map((file, index) => (
                    <ListItem 
                        key={index}
                        sx={{
                            mb: 1,
                            borderRadius: 1,
                            flexDirection: 'column',
                            alignItems: 'stretch',
                            bgcolor: currentVideo?.name === file.name ? '#2a2a2a' : '#1a1a1a',
                            transition: 'background-color 0.2s ease'
                        }}
                    >
                        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: 1 }}>
                            <ListItemIcon>
                                <VideoFileIcon sx={{ color: '#bd9544' }} />
                            </ListItemIcon>
                            <ListItemText 
                                primary={
                                    <Typography variant="body1" sx={{ color: 'white' }}>
                                        {file.name}
                                    </Typography>
                                }
                                secondary={
                                    <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                                        {file.size ? `Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB` : ''}
                                    </Typography>
                                }
                            />
                        </Box>
                        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                            <IconButton
                                onClick={() => handleVideoSelect(file)}
                                sx={{
                                    bgcolor: currentVideo?.name === file.name ? '#1b5e20' : '#2e7d32',
                                    color: 'white',
                                    '&:hover': {
                                        bgcolor: currentVideo?.name === file.name ? '#1b5e20' : '#1b5e20'
                                    },
                                    transition: 'background-color 0.2s ease'
                                }}
                            >
                                <PlayArrowIcon />
                            </IconButton>
                        </Box>
                    </ListItem>
                ))}
            </List>
        </Box>
    );
};

export default FileList;
