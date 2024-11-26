import React from 'react';
import { Box, List, ListItem, ListItemIcon, ListItemText, Typography, Button } from '@mui/material';
import VideoFileIcon from '@mui/icons-material/VideoFile';

const FileList = ({ files, onFileSelect, activeFile, isPlaying, style }) => {
    if (!files) {
        return (
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                    Loading files...
                </Typography>
            </Box>
        );
    }
    
    if (files.length === 0) {
        return (
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                    No files uploaded yet
                </Typography>
            </Box>
        );
    }

    return (
        <Box sx={{ mt: 2, ...(style || {}) }}>
            <Typography variant="h6" sx={{ mb: 1 }}>
                Uploaded Files
            </Typography>
            <List>
                {files.map((file, index) => {
                    const isActive = activeFile && activeFile.name === file.name;
                    return (
                        <ListItem 
                            key={index}
                            sx={{
                                bgcolor: isActive ? 'action.selected' : 'background.paper',
                                mb: 1,
                                borderRadius: 1,
                                flexDirection: 'column',
                                alignItems: 'stretch'
                            }}
                        >
                            <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: 1 }}>
                                <ListItemIcon>
                                    <VideoFileIcon sx={{ color: 'secondary.main' }} />
                                </ListItemIcon>
                                <ListItemText 
                                    primary={file.name}
                                    secondary={file.size ? `Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB` : ''}
                                />
                            </Box>
                            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                                <Button
                                    size="small"
                                    variant={isPlaying ? "outlined" : "contained"}
                                    color={isPlaying ? "error" : "primary"}
                                    onClick={() => onFileSelect(file)}
                                >
                                    {isPlaying ? "Stop" : "Start"}
                                </Button>
                            </Box>
                        </ListItem>
                    );
                })}
            </List>
        </Box>
    );
};

export default FileList;
