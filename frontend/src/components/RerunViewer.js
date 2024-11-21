import React, { useState, useEffect } from 'react';
import { Box, Typography } from '@mui/material';

const RerunViewer = () => {

    const [isConnected, setIsConnected] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const maxRetries = 5;

    useEffect(() => {
        const checkConnection = () => {
            const iframe = document.querySelector('iframe');
            if (iframe) {
                try {
                    // Check if iframe content is accessible
                    const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                    setIsConnected(true);
                    setRetryCount(0);
                    console.log('Rerun viewer connected successfully');
                } catch (error) {
                    console.warn('Rerun viewer connection check failed:', error);
                    setIsConnected(false);
                    
                    if (retryCount < maxRetries) {
                        setRetryCount(prev => prev + 1);
                        // Reload iframe
                        iframe.src = iframe.src;
                    }
                }
            }
        };

        const interval = setInterval(checkConnection, 5000);
        return () => clearInterval(interval);
    }, [retryCount]);

    return (
        <Box sx={{
            width: '100%',
            height: '500px',
            bgcolor: '#000',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <iframe
                src="http://localhost:9090"
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    opacity: isConnected ? 1 : 0.5,
                    transition: 'opacity 0.3s ease'
                }}
                title="Rerun Viewer"
                allow="camera"
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
                onError={(e) => {
                    console.error('Rerun viewer iframe error:', e);
                    setIsConnected(false);
                }}
            />
            {!isConnected && retryCount >= maxRetries && (
                <Box sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center',
                    color: 'error.main'
                }}>
                    <Typography variant="h6">
                        Failed to connect to Rerun viewer
                    </Typography>
                    <Typography variant="body2">
                        Please check if the Rerun server is running
                    </Typography>
                </Box>
            )}
        </Box>
    );
};

export default RerunViewer;
