import React, { useEffect } from 'react';
import { Box } from '@mui/material';

const RerunViewer = () => {
    useEffect(() => {
        // Load Rerun viewer script
        const script = document.createElement('script');
        script.src = 'https://app.rerun.io/web-viewer/0.20.0/web-viewer.js';
        script.async = true;
        document.body.appendChild(script);

        return () => {
            document.body.removeChild(script);
        };
    }, []);

    return (
        <Box sx={{
            width: '100%',
            height: '500px',
            bgcolor: '#000',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <rerun-viewer 
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none'
                }}
                recording-id="video_analytics"
                ws-url="ws://127.0.0.1:4321"
            />
        </Box>
    );
};

export default RerunViewer;
