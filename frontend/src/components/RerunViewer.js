import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography, CircularProgress, IconButton } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';

const RerunViewer = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const maxRetries = 5;

    const rerunWebUrl = process.env.REACT_APP_RERUN_WEB_URL || 'http://localhost:9090';
    const rerunWsUrl = process.env.REACT_APP_RERUN_WS_URL || 'ws://localhost:4321';

    const checkConnection = useCallback(async () => {
        const iframe = document.querySelector('iframe');
        if (!iframe) return;

        try {
            // Only check HTTP health endpoint
            const response = await fetch(`${rerunWebUrl}/health`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                },
                mode: 'cors'
            });

            if (!response.ok) {
                throw new Error('Rerun server not responding');
            }

            // Update connection state based on health check
            setIsConnected(true);
            setRetryCount(0);
            setIsLoading(false);
            setError(null);
            console.log('Rerun viewer health check successful');
        } catch (error) {
            console.warn('Rerun viewer connection check failed:', error);
            setIsConnected(false);
            setError(error.message);

            if (retryCount < maxRetries) {
                setRetryCount(prev => prev + 1);
                // Reload iframe with updated URL
                const url = `${rerunWebUrl}?url=${encodeURIComponent(rerunWsUrl)}`;
                iframe.src = url;
            } else {
                setIsLoading(false);
            }
        }
    }, [rerunWebUrl, rerunWsUrl, retryCount]);

    useEffect(() => {
        // Only check once after initialization
        //checkConnection();
    }, [checkConnection]);

    const handleRefresh = () => {
        setIsLoading(true);
        setRetryCount(0);
        setError(null);
        checkConnection();
    };

    return (
        <Box sx={{
            width: '100%',
            height: '600px',
            bgcolor: '#1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden',
            position: 'relative'
        }}>
            <Box sx={{
                position: 'absolute',
                top: 8,
                right: 8,
                zIndex: 1000
            }}>
                <IconButton
                    onClick={handleRefresh}
                    sx={{ color: '#bd9544' }}
                    disabled={isLoading}
                >
                    <RefreshIcon />
                </IconButton>
            </Box>

            <Box sx={{
                width: '100%',
                height: '100%',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                position: 'relative'
            }}>
                {isLoading && (
                    <Box sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        zIndex: 2,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: 2
                    }}>
                        <CircularProgress sx={{ color: '#bd9544' }} />
                        <Typography variant="body2" color="#bd9544">
                            Connecting to Rerun viewer...
                        </Typography>
                    </Box>
                )}
                <>
                    <iframe
                        src={`${rerunWebUrl}`}
                        style={{
                            width: '100%',
                            height: '100%',
                            border: 'none',
                            opacity: isConnected ? 1 : 0.5,
                            transition: 'opacity 0.3s ease',
                            backgroundColor: '#1a1a1a'
                        }}
                        title="Rerun Viewer"
                        allow="camera"
                        sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-presentation"
                        onError={(e) => {
                            console.error('Rerun viewer iframe error:', e);
                            setIsConnected(false);
                            setIsLoading(false);
                            setError('Failed to load Rerun viewer');
                        }}
                        onLoad={() => {
                            setIsLoading(false);
                            checkConnection();
                        }}
                    />
                    {(!isConnected && retryCount >= maxRetries) && (
                        <Box sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            textAlign: 'center',
                            color: '#bd9544',
                            bgcolor: 'rgba(0, 0, 0, 0.8)',
                            p: 3,
                            borderRadius: 2,
                            maxWidth: '80%'
                        }}>
                            {/* <Typography variant="h6" sx={{ mb: 1 }}>
                                Failed to connect to Rerun viewer
                            </Typography>
                            <Typography variant="body2" sx={{ mb: 2 }}>
                                {error || 'Please check if the Rerun server is running'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Server URL: {rerunWebUrl}
                            </Typography> */}
                        </Box>
                    )}
                </>
            </Box>
        </Box>
    );
};

export default RerunViewer;
