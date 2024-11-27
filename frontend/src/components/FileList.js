import React, { useCallback } from 'react';
import { Box, List, ListItem, ListItemIcon, ListItemText, Typography, IconButton } from '@mui/material';
import VideoFileIcon from '@mui/icons-material/VideoFile';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import useStore from '../store';

const FileList = () => {
    const { uploadedFiles, currentVideo, setCurrentVideo, setUploadedFiles } = useStore();

    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            // Refresh file list
            const listResponse = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/files/list`);
            const data = await listResponse.json();
            setUploadedFiles(data.files || []);

        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload file: ' + error.message);
        }
    }, [setUploadedFiles]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': ['.mp4', '.webm', '.ogg']
        },
        maxFiles: 1
    });

    const handleVideoSelect = (file) => {
        setCurrentVideo(file);
    };

    return (
        <Box sx={{ mt: 2 }}>
            {/* Upload Zone */}
            <Box
                {...getRootProps()}
                sx={{
                    border: '2px dashed #bd9544',
                    borderRadius: 1,
                    p: 2,
                    mb: 2,
                    textAlign: 'center',
                    cursor: 'pointer',
                    bgcolor: isDragActive ? 'rgba(189, 149, 68, 0.1)' : 'transparent',
                    '&:hover': {
                        bgcolor: 'rgba(189, 149, 68, 0.1)'
                    }
                }}
            >
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 40, color: '#bd9544', mb: 1 }} />
                <Typography>
                    {isDragActive
                        ? "Drop the video here"
                        : "Drag & drop a video file here, or click to select"}
                </Typography>
            </Box>

            {/* File List */}
            {(!uploadedFiles || uploadedFiles.length === 0) ? (
                <Box sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                        No files uploaded yet
                    </Typography>
                </Box>
            ) : (

                <Box>
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
            )}
        </Box>
    );
};

export default FileList;
