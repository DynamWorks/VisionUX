import React, { useState } from 'react';
import { Box, TextField, IconButton, Paper, Typography, CircularProgress, useMediaQuery, useTheme } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import useChat from '../hooks/useChat';
import useStore from '../store';

const Chat = () => {
    const [message, setMessage] = useState('');
    const { messages, isLoading, sendMessage, handleSceneAnalysis } = useChat();
    const { isStreaming } = useStore();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const isTablet = useMediaQuery(theme.breakpoints.down('md'));

    const handleSend = async () => {
        if (!message.trim()) return;

        try {
            await sendMessage(message);
            setMessage('');
        } catch (error) {
            console.error('Error sending message:', error);
        }
    };

    const handleAnalysis = async () => {
        if (!isStreaming) {
            console.warn('No active stream to analyze');
            return;
        }

        try {
            setMessage('');  // Clear current message
            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/analyze_scene`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    stream_type: 'camera'
                })
            });

            if (!response.ok) {
                throw new Error(`Scene analysis failed: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.scene_analysis?.description) {
                // Add system message with analysis results
                messages.push({
                    role: 'system',
                    content: `Scene Analysis:\n${data.scene_analysis.description}`
                });
            } else {
                console.warn('No scene analysis description in response:', data);
                messages.push({
                    role: 'system',
                    content: 'Scene analysis completed but no description available.'
                });
            }

        } catch (error) {
            console.error('Error handling scene analysis:', error);
            messages.push({
                role: 'system',
                content: `Error during scene analysis: ${error.message}`
            });
        }
    };

    return (
        <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Paper
                sx={{
                    p: 2,
                    height: isMobile ? '200px' : isTablet ? '300px' : 'calc(100vh - 900px)',
                    minHeight: '200px',
                    overflowY: 'auto',
                    bgcolor: '#1a1a1a',
                    color: 'white'
                }}
            >
                {messages.map((msg, idx) => (
                    <Box
                        key={idx}
                        sx={{
                            mb: 1,
                            textAlign: msg.role === 'user' ? 'right' : 'left',
                            maxWidth: isMobile ? '100%' : '80%',
                            marginLeft: msg.role === 'user' ? 'auto' : 0,
                            marginRight: msg.role === 'user' ? 0 : 'auto'
                        }}
                    >
                        <Typography
                            variant={isMobile ? 'body2' : 'body1'}
                            sx={{
                                display: 'inline-block',
                                bgcolor: msg.role === 'user' ? '#2c5282' : 
                                        msg.role === 'assistant' ? '#4a5568' :
                                        msg.role === 'system' ? '#38a169' : '#e53e3e',
                                p: isMobile ? 1 : 2,
                                borderRadius: 1,
                                maxWidth: '100%',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word'
                            }}
                        >
                            {msg.content}
                        </Typography>
                    </Box>
                ))}
            </Paper>
            <Box sx={{ 
                display: 'flex', 
                gap: 1,
                flexDirection: 'row',
                alignItems: 'flex-start'
            }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Ask about the video..."
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSend();
                        }
                    }}
                    disabled={isLoading}
                    multiline={isMobile}
                    rows={isMobile ? 2 : 1}
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            bgcolor: '#1a1a1a',
                            color: 'white',
                            '& fieldset': {
                                borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover fieldset': {
                                borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                        }
                    }}
                />
                <IconButton
                    onClick={handleSend}
                    disabled={!message.trim() || isLoading}
                    sx={{
                        bgcolor: '#1976d2',
                        color: 'white',
                        '&:hover': {
                            bgcolor: '#1565c0'
                        },
                        '&.Mui-disabled': {
                            bgcolor: 'rgba(25, 118, 210, 0.3)',
                            color: 'rgba(255, 255, 255, 0.3)'
                        }
                    }}
                >
                    {isLoading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
                </IconButton>
            </Box>
        </Box>
    );
};

export default Chat;
