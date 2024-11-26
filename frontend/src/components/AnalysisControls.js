import React from 'react';
import { Box, Button } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TimelineIcon from '@mui/icons-material/Timeline';

const AnalysisControls = ({ onSceneAnalysis, onEdgeDetection, disabled }) => {
    return (
        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
            <Button
                variant="contained"
                startIcon={<VisibilityIcon />}
                onClick={onSceneAnalysis}
                disabled={disabled}
                sx={{ flex: 1 }}
            >
                Scene Analysis
            </Button>
            <Button
                variant="contained"
                startIcon={<TimelineIcon />}
                onClick={onEdgeDetection}
                disabled={disabled}
                sx={{ flex: 1 }}
            >
                Edge Detection
            </Button>
        </Box>
    );
};

export default AnalysisControls;
