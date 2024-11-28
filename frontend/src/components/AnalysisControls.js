import React, { useState } from 'react';
import { Box, Button } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TimelineIcon from '@mui/icons-material/Timeline';
import CircularProgress from '@mui/material/CircularProgress';
import useStore from '../store';

const AnalysisControls = ({ onSceneAnalysis, onEdgeDetection }) => {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const { currentVideo } = useStore();

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
        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
            <Button
                variant="contained"
                startIcon={isAnalyzing ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
                onClick={handleSceneAnalysis}
                disabled={isAnalyzing || !currentVideo}
                sx={{ flex: 1 }}
            >
                {isAnalyzing ? 'Analyzing...' : 'Scene Analysis'}
            </Button>
            <Button
                variant="contained"
                startIcon={<TimelineIcon />}
                onClick={onEdgeDetection}
                disabled={!currentVideo}
                sx={{ flex: 1 }}
            >
                Edge Detection
            </Button>
        </Box>
    );
};

export default AnalysisControls;
