import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';

const RerunViewer = () => {
    const viewerRef = useRef(null);

    useEffect(() => {
        const loadViewer = async () => {
            const script = document.createElement('script');
            script.src = 'https://app.rerun.io/web-viewer/0.20.0/web-viewer.js';
            script.async = true;
            
            script.onload = () => {
                if (viewerRef.current) {
                    const viewer = viewerRef.current;
                    viewer.setAttribute('recording-id', 'video_analytics');
                    viewer.setAttribute('ws-url', process.env.REACT_APP_RERUN_WS_URL || 'ws://localhost:4321');
                    viewer.setAttribute('auto-connect', 'true');
                }
            };

            document.body.appendChild(script);
            return () => {
                document.body.removeChild(script);
            };
        };

        loadViewer();
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
                ref={viewerRef}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none'
                }}
            />
        </Box>
    );
};

export default RerunViewer;
