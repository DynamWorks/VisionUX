import React, { useState, useEffect } from 'react';
import { Box } from '@mui/material';

const RerunViewer = () => {

    useEffect(() => {
        // Only handle viewer connection state
        const checkConnection = () => {
            const iframe = document.querySelector('iframe');
            if (iframe) {
                // Connection state is handled by parent component
                console.log('Rerun viewer iframe detected');
            }
        };

        const interval = setInterval(checkConnection, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <Box sx={{
            width: '100%',
            height: '500px',
            bgcolor: '#000',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <iframe
                src={`${process.env.REACT_APP_RERUN_URL}?url=${process.env.REACT_APP_RERUN_WS_URL}`}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none'
                }}
                title="Rerun Viewer"
                allow="camera"
                sandbox="allow-scripts allow-same-origin allow-forms"
            />
        </Box>
    );
};

export default RerunViewer;
