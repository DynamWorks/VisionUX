import React, { useState } from 'react';
import { Box, Button, ToggleButton, Switch, FormControlLabel } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TimelineIcon from '@mui/icons-material/Timeline';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import CircularProgress from '@mui/material/CircularProgress';
import useStore from '../store';

const AnalysisControls = ({ onSceneAnalysis, onEdgeDetection }) => {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const { 
        currentVideo, 
        isEdgeDetectionEnabled, 
        setEdgeDetectionEnabled,
        autoAnalysisEnabled,
        setAutoAnalysisEnabled
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
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
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
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
                variant="contained"
                startIcon={isAnalyzing ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
                onClick={handleSceneAnalysis}
                disabled={isAnalyzing || !currentVideo}
                sx={{ flex: 1 }}
            >
                {isAnalyzing ? 'Analyzing...' : 'Scene Analysis'}
            </Button>
            <ToggleButton
                value="edge"
                selected={isEdgeDetectionEnabled}
                onChange={() => {
                    setEdgeDetectionEnabled(!isEdgeDetectionEnabled);
                    onEdgeDetection(!isEdgeDetectionEnabled);
                }}
                disabled={!currentVideo}
                sx={{ 
                    flex: 1,
                    bgcolor: isEdgeDetectionEnabled ? 'primary.main' : 'inherit',
                    '&.Mui-selected': {
                        bgcolor: 'primary.main',
                        '&:hover': {
                            bgcolor: 'primary.dark'
                        }
                    }
                }}
            >
                <TimelineIcon sx={{ mr: 1 }} />
                Edge Detection
            </ToggleButton>
        </Box>
    </Box>
    );
};

export default AnalysisControls;
