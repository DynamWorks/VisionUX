import React, { useEffect, useState, useCallback } from 'react';
import logo from './assets/logo.png';
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
    const { handleSceneAnalysis: chatHandleAnalysis, addMessage, setMessages } = useChat();

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

            // Add initial analysis message
            addMessage('system', 'Scene analysis in progress...');

            if (data.chat_messages) {
                // Add each message from the response
                data.chat_messages.forEach(msg => {
                    addMessage(msg.role, msg.content);
                });
            } else if (data.scene_analysis?.description) {
                // Add scene analysis description
                addMessage('assistant', `Scene Analysis Results:\n${data.scene_analysis.description}`);
                
                // Add technical details if available
                if (data.technical_details) {
                    addMessage('system', `Technical Details:\n${JSON.stringify(data.technical_details, null, 2)}`);
                }
            }

            // Update analysis results in store
            if (data.results) {
                setAnalysisResults(data.results);
                // Add results summary to chat
                addMessage('system', `Analysis complete. Results saved.`);
            }

            // Add completion message
            addMessage('system', 'Analysis complete - results saved.');

            // Refresh chat history after analysis
            const videoState = useStore.getState();
            if (videoState.currentVideo) {
                const historyResponse = await fetch(
                    `${process.env.REACT_APP_API_URL}/api/v1/chat/history/${videoState.currentVideo.name}`
                );
                if (historyResponse.ok) {
                    const historyData = await historyResponse.json();
                    if (historyData.messages) {
                        setMessages(historyData.messages);
                    }
                }
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
                    background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
                    backgroundSize: 'cover',
                    display: 'flex',
                    flexDirection: 'column',
                    transition: 'background 0.3s ease-in-out'
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
                    <Paper 
                        elevation={3}
                        sx={{
                            mb: 4,
                            p: 3,
                            minHeight: '150px',
                            background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%)',
                            borderBottom: '2px solid rgba(255,255,255,0.2)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            boxShadow: '0 4px 20px rgba(99,102,241,0.3)',
                            position: 'relative',
                            '&::before': {
                                content: '""',
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 70%)',
                                pointerEvents: 'none'
                            }
                        }}
                    >
                        <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            width: '100%'
                        }}>
                            <img 
                                src={logo} 
                                alt="VisionUX Logo" 
                                style={{ 
                                    height: '40px',
                                    width: 'auto',
                                    animation: 'spin3D 2s infinite linear'
                                }}
                            />
                            <Typography variant="h4" component="h1" sx={{ 
                                color: '#ffffff',
                                fontWeight: 700,
                                textAlign: 'center',
                                flex: 1,
                                animation: 'titleGlow 2s ease-in-out infinite',
                                textShadow: '0 0 10px rgba(255,255,255,0.8), 0 0 20px rgba(255,255,255,0.4), 0 0 30px rgba(255,255,255,0.2)',
                                letterSpacing: '3px',
                                position: 'relative',
                                zIndex: 1,
                                '&::after': {
                                    content: '""',
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    transform: 'translate(-50%, -50%)',
                                    width: '120%',
                                    height: '120%',
                                    background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
                                    zIndex: -1
                                }
                            }}>
                                VisionUX
                            </Typography>
                            <Box sx={{ width: '40px' }} /> {/* Spacer to balance logo */}
                        </Box>
                        {error && (
                            <Typography color="error">
                                Error: {error}
                            </Typography>
                        )}
                    </Paper>

                    {/* Main Content */}
                    <Box
                        sx={{
                            display: 'flex',
                            gap: 3,
                            flex: 1,
                            minHeight: 0,
                            height: 'calc(100vh - 250px)', // Account for header and padding
                            flexDirection: { xs: 'column', lg: 'row' }
                        }}
                    >
                        {/* Main Panel */}
                        <Paper
                            elevation={3}
                            sx={{
                                flex: 2,
                                bgcolor: '#121212',
                                overflow: 'hidden',
                                display: 'flex',
                                flexDirection: 'column'
                            }}
                        >
                            <Box sx={{
                                p: 3,
                                flex: 1,
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 3,
                                width: '100%',
                                mx: 'auto',
                                height: '100%',
                                minHeight: 0 // Add this to allow proper flex behavior
                            }}>
                                <Box sx={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: 2,
                                    height: '100%'
                                }}>
                                    <Box sx={{ 
                                        width: '100%',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        mb: 2
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
                                    <Box sx={{ 
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
                                </Box>
                            </Box>
                        </Paper>

                        {/* Right Panel - Chat */}
                        <Paper
                            elevation={3}
                            sx={{
                                width: { xs: '100%', lg: '500px' },
                                bgcolor: '#121212',
                                overflow: 'hidden',
                                display: 'flex',
                                flexDirection: 'column',
                                borderLeft: '1px solid rgba(255, 255, 255, 0.1)',
                                boxShadow: '0px 0px 15px rgba(0,0,0,0.2)',
                                height: '100%',
                                minHeight: 0 // Add this to allow proper flex behavior
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
