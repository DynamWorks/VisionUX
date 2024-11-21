import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';

const RerunViewer = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const maxRetries = 5;

    useEffect(() => {
        const checkConnection = () => {
            const iframe = document.querySelector('iframe');
            if (iframe) {
                try {
                    const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                    setIsConnected(true);
                    setRetryCount(0);
                    setIsLoading(false);
                    console.log('Rerun viewer connected successfully');
                } catch (error) {
                    console.warn('Rerun viewer connection check failed:', error);
                    setIsConnected(false);

                    if (retryCount < maxRetries) {
                        setRetryCount(prev => prev + 1);
                        iframe.src = iframe.src;
                    } else {
                        setIsLoading(false);
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
            height: '600px',
            bgcolor: '#1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden',
            position: 'relative',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
        }}>
            {isLoading ? (
                <CircularProgress sx={{ color: '#bd9544' }} />
            ) : (
                <>
                    <iframe
                        src={`${process.env.REACT_APP_RERUN_URL || 'http://localhost:9090'}?url=${process.env.REACT_APP_RERUN_WS_URL || 'ws://localhost:4321'}`}
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
                            setIsLoading(false);
                        }}
                        onLoad={() => setIsLoading(false)}
                    />
                    {!isConnected && retryCount >= maxRetries && (
                        <Box sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            textAlign: 'center',
                            color: '#bd9544',
                            bgcolor: 'rgba(0, 0, 0, 0.8)',
                            p: 3,
                            borderRadius: 2
                        }}>
                            <Typography variant="h6" sx={{ mb: 1 }}>
                                Failed to connect to Rerun viewer
                            </Typography>
                            <Typography variant="body2">
                                Please check if the Rerun server is running
                            </Typography>
                        </Box>
                    )}
                </>
            )}
        </Box>
    );
};

export default RerunViewer;
