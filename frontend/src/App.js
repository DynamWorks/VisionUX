import React, { useEffect, useState } from 'react';
import { Container, Box, Paper, Typography } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme';
import FileList from './components/FileList';
import CameraSelector from './components/CameraSelector';
import InputSelector from './components/InputSelector';
import CustomViewer from './components/CustomViewer';
import Chat from './components/Chat';
import useStore from './store';
import { websocketService } from './services/websocket';

function App() {
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const { setUploadedFiles, inputMode } = useStore();

    // Initialize WebSocket connection
    useEffect(() => {
        websocketService.connect();
        return () => websocketService.disconnect();
    }, []);

    // Fetch uploaded files
    useEffect(() => {
        const fetchFiles = async () => {
            try {
                setIsLoading(true);
                const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/files/list`);
                if (!response.ok) {
                    throw new Error(`Failed to fetch files: ${response.statusText}`);
                }
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                setUploadedFiles(data.files || []);
                setError(null);
            } catch (error) {
                console.error('Error fetching files:', error);
                setError(error.message);
                setUploadedFiles([]);
            } finally {
                setIsLoading(false);
            }
        };

        if (inputMode === 'upload') {
            fetchFiles();
            // Poll for new files every 5 seconds
            const interval = setInterval(fetchFiles, 5000);
            return () => clearInterval(interval);
        } else {
            setUploadedFiles([]);
        }
    }, [setUploadedFiles, inputMode]);

    return (
        <ThemeProvider theme={theme}>
            <Box 
                sx={{ 
                    minHeight: '100vh',
                    bgcolor: '#0a0a0a',
                    display: 'flex',
                    flexDirection: 'column'
                }}
            >
                <Container 
                    maxWidth={false} 
                    sx={{ 
                        py: 4,
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column'
                    }}
                >
                    {/* Header */}
                    <Box sx={{ mb: 4 }}>
                        <Typography variant="h4" component="h1" gutterBottom sx={{ color: 'white' }}>
                            Vision LLM
                        </Typography>
                        {error && (
                            <Typography color="error" sx={{ mb: 2 }}>
                                Error: {error}
                            </Typography>
                        )}
                    </Box>

                    {/* Main Content */}
                    <Box 
                        sx={{ 
                            display: 'flex',
                            gap: 3,
                            flex: 1,
                            minHeight: 0,  // Important for nested flex containers
                            flexDirection: { xs: 'column', lg: 'row' }
                        }}
                    >
                        {/* Left Panel */}
                        <Paper 
                            elevation={3}
                            sx={{
                                width: { xs: '100%', lg: '300px' },
                                bgcolor: '#121212',
                                overflow: 'hidden',
                                display: 'flex',
                                flexDirection: 'column'
                            }}
                        >
                            <Box sx={{ 
                                p: 2,
                                flex: 1,
                                overflowY: 'auto',
                                display: 'flex',
                                flexDirection: 'column'
                            }}>
                                <InputSelector />
                                {isLoading ? (
                                    <Box sx={{ p: 2, textAlign: 'center' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            Loading...
                                        </Typography>
                                    </Box>
                                ) : (
                                    inputMode === 'upload' ? <FileList /> : <CameraSelector />
                                )}
                            </Box>
                        </Paper>

                        {/* Center Panel */}
                        <Paper 
                            elevation={3}
                            sx={{
                                flex: 1,
                                bgcolor: '#121212',
                                overflow: 'hidden',
                                display: 'flex',
                                flexDirection: 'column'
                            }}
                        >
                            <Box sx={{ 
                                p: 2,
                                flex: 1,
                                display: 'flex',
                                flexDirection: 'column'
                            }}>
                                <CustomViewer />
                            </Box>
                        </Paper>

                        {/* Right Panel */}
                        <Paper 
                            elevation={3}
                            sx={{
                                width: { xs: '100%', lg: '300px' },
                                bgcolor: '#121212',
                                overflow: 'hidden',
                                display: 'flex',
                                flexDirection: 'column'
                            }}
                        >
                            <Box sx={{ 
                                p: 2,
                                flex: 1,
                                overflowY: 'auto'
                            }}>
                                <Chat />
                            </Box>
                        </Paper>
                    </Box>
                </Container>
            </Box>
        </ThemeProvider>
    );
}

export default App;
