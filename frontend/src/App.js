import React, { useEffect, useState, useCallback } from 'react';
import { Container, Box, Paper, Typography, useTheme } from '@mui/material';
import AnalysisControls from './components/AnalysisControls';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme';
import FileList from './components/FileList';
import CameraSelector from './components/CameraSelector';
import InputSelector from './components/InputSelector';
import CustomViewer from './components/CustomViewer';
import Chat from './components/Chat';
import useStore from './store';
import useChat from './hooks/useChat';
import { websocketService } from './services/websocket';

function App() {
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const { setUploadedFiles, inputMode, isStreaming, setAnalysisResults } = useStore();
    const { handleSceneAnalysis: chatHandleAnalysis, addMessage } = useChat();

    // Initialize WebSocket connection
    useEffect(() => {
        websocketService.connect();
        return () => websocketService.disconnect();
    }, []);

    const handleSceneAnalysis = useCallback(async () => {
        try {
            // Get current video information from store
            const { currentVideo } = useStore.getState();

            if (!currentVideo) {
                throw new Error('No video selected');
            }

            // Add loading message to chat
            addMessage('system', 'Analyzing scene...');

            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/analyze_scene`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    stream_type: 'video',
                    video_file: currentVideo.name,
                    num_frames: 8
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Scene analysis failed (${response.status}): ${response.statusText}`);
            }

            const data = await response.json();

            if (data.chat_messages) {
                // Add each message from the response
                data.chat_messages.forEach(msg => {
                    addMessage(msg.role, msg.content);
                });
            } else if (data.scene_analysis?.description) {
                // Fallback for older response format
                addMessage('system', `Scene Analysis:\n${data.scene_analysis.description}`);
            }

            // Update analysis results in store
            if (data.results) {
                setAnalysisResults(data.results);
            }

            // Notify WebSocket about analysis completion
            if (websocketService.isConnected()) {
                websocketService.emit('analysis_complete', {
                    type: 'scene_analysis',
                    results: data
                });
            }

            return data;

        } catch (error) {
            console.error('Error in scene analysis:', error);
            addMessage('error', error.message);
            throw error;
        }
    }, [addMessage, chatHandleAnalysis]);

    const handleEdgeDetection = useCallback(async (enabled) => {
        try {
            if (enabled) {
                websocketService.emit('start_edge_detection');
                addMessage('system', 'Edge detection started');
            } else {
                websocketService.emit('stop_edge_detection');
                addMessage('system', 'Edge detection stopped');
            }
        } catch (error) {
            console.error('Error toggling edge detection:', error);
            addMessage('error', `Edge detection error: ${error.message}`);
        }
    }, [inputMode, addMessage]);

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

    // Fetch files on mount and mode change
    useEffect(() => {
        if (inputMode === 'upload') {
            fetchFiles();
        } else {
            setUploadedFiles([]);
        }
    }, [inputMode]);

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
                            minHeight: 0,
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
                                <AnalysisControls
                                    onSceneAnalysis={handleSceneAnalysis}
                                    onEdgeDetection={handleEdgeDetection}
                                    disabled={!isStreaming}
                                />
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
