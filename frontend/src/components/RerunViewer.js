import React from 'react';
import { Box } from '@mui/material';

const RerunViewer = () => {
    return (
        <Box sx={{
            width: '100%',
            height: '500px',
            bgcolor: '#000',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <iframe // the src should not change :src={`https://app.rerun.io/version/${process.env.REACT_APP_RERUN_VER}/?ws_url=${process.env.REACT_APP_RERUN_WS_URL}&hide_welcome_screen=true`}
                src={`https://app.rerun.io/version/${process.env.REACT_APP_RERUN_VER}/?ws_url=${process.env.REACT_APP_RERUN_WS_URL}&hide_welcome_screen=true`}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none'
                }}
                title="Rerun Viewer"
                allow="camera"
                sandbox="allow-scripts allow-same-origin"
            />
        </Box>
    );
};

export default RerunViewer;
