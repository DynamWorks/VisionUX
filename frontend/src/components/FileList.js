import React from 'react';
import { Box, List, ListItem, ListItemIcon, ListItemText, Typography } from '@mui/material';
import VideoFileIcon from '@mui/icons-material/VideoFile';

const FileList = ({ files, onFileSelect }) => {
    if (!files || files.length === 0) {
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
                {files.map((file, index) => (
                    <ListItem 
                        key={index}
                        button
                        onClick={() => onFileSelect(file)}
                        sx={{
                            bgcolor: 'background.paper',
                            mb: 1,
                            borderRadius: 1,
                            '&:hover': {
                                bgcolor: 'action.hover'
                            }
                        }}
                    >
                        <ListItemIcon>
                            <VideoFileIcon sx={{ color: 'secondary.main' }} />
                        </ListItemIcon>
                        <ListItemText 
                            primary={file.name}
                            secondary={`Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB`}
                        />
                    </ListItem>
                ))}
            </List>
        </Box>
    );
};

export default FileList;
