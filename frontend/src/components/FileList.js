import React, { useCallback, useState } from 'react';
import { Box, List, ListItem, ListItemIcon, ListItemText, Typography, IconButton, LinearProgress } from '@mui/material';
import VideoFileIcon from '@mui/icons-material/VideoFile';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import useStore from '../store';
import useChat from '../hooks/useChat';

const FileList = () => {
    const { uploadedFiles, currentVideo, setCurrentVideo, setUploadedFiles, autoAnalysisEnabled } = useStore();
    const { addMessage } = useChat();

    const [uploadProgress, setUploadProgress] = useState(0);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);

    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;

        setIsUploading(true);
        setUploadProgress(0);
        setUploadError(null);

        try {

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/upload`, {
                method: 'POST',
                body: formData,
                onUploadProgress: (progressEvent) => {
                    const progress = (progressEvent.loaded / progressEvent.total) * 100;
                    setUploadProgress(Math.round(progress));
                }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Upload failed: ${response.statusText}`);
            }

            // Refresh file list after successful upload
            const listResponse = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/files/list`);
            if (!listResponse.ok) {
                throw new Error(`Failed to fetch files: ${listResponse.statusText}`);
            }
            const listData = await listResponse.json();
            if (listData.error) {
                throw new Error(listData.error);
            }
            const files = listData.files || [];
            setUploadedFiles(files);
            
            // Clear old session data if this is a new upload
            if (file) {
                localStorage.clear();
            }
            
            // Auto-select and analyze logic:
            let fileToSelect = null;
            
            // 1. If there's only one file, select it
            if (files.length === 1) {
                fileToSelect = files[0];
            } 
            // 2. If this is a new upload, select the uploaded file
            else if (file && files.length > 0) {
                fileToSelect = files.find(f => f.name === file.name);
            }
            // 3. Otherwise maintain current selection if it still exists
            else if (currentVideo) {
                fileToSelect = files.find(f => f.name === currentVideo.name);
            }

            if (fileToSelect) {
                handleVideoSelect(fileToSelect, true); // Pass true to indicate this is from upload
            }
            
            setUploadProgress(100);

        } catch (error) {
            console.error('Upload error:', error);
            setUploadError(error.message);
        } finally {
            setIsUploading(false);
        }
    }, [setUploadedFiles]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': ['.mp4', '.webm', '.ogg']
        },
        maxFiles: 1
    });

    const handleVideoSelect = async (file, isFromUpload = false) => {
        setCurrentVideo(file);

        // Only trigger auto-analysis if enabled and this is from an upload
        if (autoAnalysisEnabled && isFromUpload) {
            try {
                console.log('Triggering auto-analysis for:', file.name);
                addMessage('system', 'Auto-analyzing scene...');
                
                const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/analyze_scene`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        stream_type: 'video',
                        video_file: file.name,
                        num_frames: 8
                    })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Scene analysis failed');
                }

                const data = await response.json();
                if (data.scene_analysis?.description) {
                    addMessage('system', `Auto Scene Analysis:\n${data.scene_analysis.description}`);
                    
                    // Also update analysis results in store if available
                    if (data.results) {
                        setAnalysisResults(data.results);
                    }
                }
            } catch (error) {
                console.error('Auto analysis error:', error);
                addMessage('error', `Auto analysis failed: ${error.message}`);
            }
        }
    };

    return (
        <Box sx={{ mt: 2 }}>
            {/* Upload Zone */}
            <Box sx={{ width: '100%', mb: 2 }}>
                <Box
                    {...getRootProps()}
                    sx={{
                        border: '2px dashed #bd9544',
                        borderRadius: 1,
                        p: 2,
                        textAlign: 'center',
                        cursor: isUploading ? 'not-allowed' : 'pointer',
                        bgcolor: isDragActive ? 'rgba(189, 149, 68, 0.1)' : 'transparent',
                        opacity: isUploading ? 0.7 : 1,
                        '&:hover': {
                            bgcolor: isUploading ? 'transparent' : 'rgba(189, 149, 68, 0.1)'
                        }
                    }}
                >
                    <input {...getInputProps()} disabled={isUploading} />
                    <CloudUploadIcon sx={{ fontSize: 40, color: '#bd9544', mb: 1 }} />
                    <Typography>
                        {isUploading ? "Uploading..." :
                         isDragActive ? "Drop the video here" :
                         "Drag & drop a video file here, or click to select"}
                    </Typography>
                </Box>
                
                {isUploading && (
                    <Box sx={{ mt: 2 }}>
                        <LinearProgress 
                            variant="determinate" 
                            value={uploadProgress}
                            sx={{
                                height: 8,
                                borderRadius: 4,
                                bgcolor: 'rgba(189, 149, 68, 0.2)',
                                '& .MuiLinearProgress-bar': {
                                    bgcolor: '#bd9544'
                                }
                            }}
                        />
                        <Typography variant="body2" color="textSecondary" align="center" sx={{ mt: 1 }}>
                            {uploadProgress}% uploaded
                        </Typography>
                    </Box>
                )}

                {uploadError && (
                    <Typography 
                        color="error" 
                        variant="body2" 
                        align="center"
                        sx={{ mt: 1 }}
                    >
                        {uploadError}
                    </Typography>
                )}
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
