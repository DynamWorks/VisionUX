import React from 'react';
import { Box, Button, Typography } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const VideoUpload = ({ onUpload }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        accept: {
            'video/*': ['.mp4', '.webm', '.ogg']
        },
        maxFiles: 1,
        onDrop: files => {
            if (files?.[0]) {
                onUpload(files[0]);
            }
        }
    });

    return (
        <Box sx={{ mb: 2 }}>
            <Box
                {...getRootProps()}
                sx={{
                    border: '2px dashed #bd9544',
                    borderRadius: 1,
                    p: 2,
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
        </Box>
    );
};

export default VideoUpload;
