import React from 'react';
import { Box, Typography } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const VideoUpload = ({ onUpload, setUploadedFiles, disabled }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        disabled,
        accept: {
            'video/*': ['.mp4', '.webm', '.ogg']
        },
        maxFiles: 1,
        multiple: false,
        onDrop: files => {
            if (files?.[0]) {
                const file = files[0];
                // Verify it's a video file
                if (!file.type.startsWith('video/')) {
                    alert('Please upload a video file');
                    return;
                }
                onUpload(file);
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
                    bgcolor: disabled ? 'rgba(0, 0, 0, 0.1)' : (isDragActive ? 'rgba(189, 149, 68, 0.1)' : 'transparent'),
                    opacity: disabled ? 0.5 : 1,
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    '&:hover': {
                        bgcolor: 'rgba(189, 149, 68, 0.1)'
                    }
                }}
            >
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 40, color: '#bd9544', mb: 1 }} />
                <Typography>
                    {disabled 
                        ? "Stop video stream to enable upload"
                        : (isDragActive
                            ? "Drop the video here"
                            : "Drag & drop a video file here, or click to select")}
                </Typography>
            </Box>
        </Box>
    );
};

export default VideoUpload;
