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
            <iframe
                src="https://app.rerun.io/web-viewer/0.14.1/"
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none'
                }}
                title="Rerun Viewer"
                allow="camera"
            />
        </Box>
    );
};

export default RerunViewer;
