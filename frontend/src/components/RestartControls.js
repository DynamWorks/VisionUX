import React from 'react';
import { Box, Button } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import RestartAltIcon from '@mui/icons-material/RestartAlt';

const RestartControls = ({ onRestartWebSockets }) => {
    return (
        <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
                variant="outlined"
                size="small"
                startIcon={<RefreshIcon />}
                onClick={onRestartWebSockets}
                sx={{ color: 'white', borderColor: 'white' }}
            >
                Restart WebSocket
            </Button>
        </Box>
    );
};

export default RestartControls;
