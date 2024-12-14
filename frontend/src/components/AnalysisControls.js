import React, { useState } from 'react';
import { Box, Button, ToggleButton, Switch, FormControlLabel, ButtonGroup } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TimelineIcon from '@mui/icons-material/Timeline';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import ChatIcon from '@mui/icons-material/Chat';
import CompareIcon from '@mui/icons-material/Compare';
import CircularProgress from '@mui/material/CircularProgress';
import useStore from '../store';

const AnalysisControls = ({ onSceneAnalysis, onEdgeDetection }) => {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const { 
        currentVideo, 
        isEdgeDetectionEnabled, 
        setEdgeDetectionEnabled,
        isObjectDetectionEnabled, 
        setObjectDetectionEnabled,
        autoAnalysisEnabled,
        setAutoAnalysisEnabled,
        isRagEnabled,
        setRagEnabled,
        showEdgeVisualization,
        setShowEdgeVisualization,
        showObjectVisualization,
        setShowObjectVisualization,
        currentVisualization,
        setCurrentVisualization,
        addMessage
    } = useStore();

    const handleSceneAnalysis = async () => {
        try {
            if (!currentVideo) {
                throw new Error('No video selected');
            }

            setIsAnalyzing(true);
            await onSceneAnalysis();
        } catch (error) {
            console.error('Scene analysis failed:', error);
        } finally {
            setIsAnalyzing(false);
        }
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 2 }}>
            <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'flex-end',
                gap: 2 
            }}>
                <FormControlLabel
                    control={
                        <Switch
                            checked={autoAnalysisEnabled}
                            onChange={(e) => setAutoAnalysisEnabled(e.target.checked)}
                        />
                    }
                    label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <AutoFixHighIcon />
                            <span>Auto Analysis</span>
                        </Box>
                    }
                />
                {currentVideo && (
                    <>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={showEdgeVisualization}
                                    onChange={(e) => {
                                        setShowEdgeVisualization(e.target.checked);
                                        if (e.target.checked) {
                                            setShowObjectVisualization(false);
                                        }
                                    }}
                                    disabled={!currentVideo || !currentVisualization}
                                />
                            }
                            label={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <CompareIcon />
                                    <span>Show Edge Detection</span>
                                </Box>
                            }
                        />
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={showObjectVisualization}
                                    onChange={(e) => {
                                        setShowObjectVisualization(e.target.checked);
                                        if (e.target.checked) {
                                            setShowEdgeVisualization(false);
                                        }
                                    }}
                                    disabled={!currentVideo || !currentVisualization?.includes('_objects')}
                                />
                            }
                            label={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <CompareIcon />
                                    <span>Show Object Detection</span>
                                </Box>
                            }
                        />
                    </>
                )}
            </Box>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button
                variant="contained"
                startIcon={isAnalyzing ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
                onClick={handleSceneAnalysis}
                disabled={isAnalyzing || !currentVideo}
                sx={{ 
                    flex: 1,
                    position: 'relative',
                    '& .MuiCircularProgress-root': {
                        position: 'absolute',
                        left: '50%',
                        transform: 'translateX(-50%)'
                    }
                }}
            >
                {isAnalyzing ? 'Analyzing...' : 'Scene Analysis'}
            </Button>
            <Button
                variant="contained"
                startIcon={isEdgeDetectionEnabled ? <CircularProgress size={20} color="inherit" /> : <TimelineIcon />}
                onClick={() => {
                    if (!currentVideo) return;
                    
                    setEdgeDetectionEnabled(true);
                    fetch(`${process.env.REACT_APP_API_URL}/api/v1/detect_edges`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            video_file: currentVideo.name
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        // Add RAG response and handle tool execution
                        if (data.rag_response) {
                            addMessage('System', data.rag_response);
                        }
                        console.log('Edge detection complete:', data);
                        if (data.visualization) {
                            setCurrentVisualization(data.visualization);
                            // Only set visualization if not already set by chat response
                            const { showEdgeVisualization } = useStore.getState();
                            if (!showEdgeVisualization) {
                                setShowEdgeVisualization(true);
                                setShowObjectVisualization(false);
                            }
                            // Force video reload to trigger auto-play
                            const videoElement = document.querySelector('video');
                            if (videoElement) {
                                videoElement.load();
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Edge detection failed:', error);
                    })
                    .finally(() => {
                        setEdgeDetectionEnabled(false);
                    });
                }}
                disabled={!currentVideo || isEdgeDetectionEnabled}
                sx={{ flex: 1 }}
            >
                {isEdgeDetectionEnabled ? 'Processing...' : 'Edge Detection'}
            </Button>
            <Button
                variant="contained"
                startIcon={isObjectDetectionEnabled ? <CircularProgress size={20} color="inherit" /> : <TimelineIcon />}
                onClick={() => {
                    if (!currentVideo) return;
                    setObjectDetectionEnabled(true);
                    fetch(`${process.env.REACT_APP_API_URL}/api/v1/detect_objects`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            video_file: currentVideo.name
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            throw new Error(data.error);
                        }

                        // Add RAG response and handle tool execution
                        if (data.rag_response) {
                            addMessage('System', data.rag_response);
                        }
                        // Handle successful object detection
                        console.log('Object detection complete:', data);
                        if (data.visualization) {
                            setCurrentVisualization(data.visualization);
                            // Only set visualization if not already set by chat response
                            const { showObjectVisualization } = useStore.getState();
                            if (!showObjectVisualization) {
                                setShowObjectVisualization(true);
                                setShowEdgeVisualization(false);
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Object detection failed:', error);
                    })
                    .finally(() => {
                        setObjectDetectionEnabled(false);
                    });
                }}
                disabled={!currentVideo || isObjectDetectionEnabled}
                sx={{ flex: 1 }}
            >
                {isObjectDetectionEnabled ? 'Processing...' : 'Object Detection'}
            </Button>
        </Box>
    </Box>
    );
};

export default AnalysisControls;
