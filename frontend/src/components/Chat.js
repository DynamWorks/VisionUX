import React, { useState, useRef, useEffect } from 'react';
import { Box, TextField, IconButton, Paper, Typography, CircularProgress, useMediaQuery, useTheme, Button } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import useChat from '../hooks/useChat';
import useStore from '../store';

const Chat = () => {
    const [message, setMessage] = useState('');
    const { messages, isLoading, sendMessage, clearChat, addMessage } = useChat();
    const { isStreaming } = useStore();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const isTablet = useMediaQuery(theme.breakpoints.down('md'));
    const chatBoxRef = useRef(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = async () => {
        if (!message.trim()) return;
        try {
            await sendMessage(message);
            setMessage('');
        } catch (error) {
            console.error('Error sending message:', error);
        }
    };

    const getMessageStyle = (role) => {
        const baseStyle = {
            display: 'inline-block',
            p: isMobile ? 1 : 2,
            borderRadius: 1,
            maxWidth: '100%',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word'
        };

        switch (role) {
            case 'user':
                return { ...baseStyle, bgcolor: '#2c5282' };
            case 'assistant':
                return { ...baseStyle, bgcolor: '#4a5568' };
            case 'system':
                return { ...baseStyle, bgcolor: '#4a5568', width: '100%' };
            case 'error':
                return { ...baseStyle, bgcolor: '#e53e3e' };
            default:
                return { ...baseStyle, bgcolor: '#4a5568' };
        }
    };

    return (
        <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Chat</Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                        startIcon={<RefreshIcon />}
                        onClick={() => {
                            const { currentVideo } = useStore.getState();
                            if (currentVideo) {
                                fetch(`${process.env.REACT_APP_API_URL}/api/v1/chat/history/${currentVideo.name}`)
                                    .then(response => response.json())
                                    .then(data => {
                                        if (data.messages) {
                                            setMessages(data.messages);
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error fetching chat history:', error);
                                    });
                            }
                        }}
                        size="small"
                        sx={{
                            color: 'primary.main',
                            '&:hover': {
                                bgcolor: 'primary.dark',
                                color: 'white'
                            }
                        }}
                    >
                        Refresh
                    </Button>
                    <Button
                        startIcon={<DeleteIcon />}
                        onClick={clearChat}
                        disabled={messages.length === 0}
                        size="small"
                        sx={{
                            color: 'error.main',
                            '&:hover': {
                                bgcolor: 'error.dark',
                                color: 'white'
                            }
                        }}
                    >
                        Clear Chat
                    </Button>
                </Box>
            </Box>
            <Paper
                ref={chatBoxRef}
                sx={{
                    p: 2,
                    height: isMobile ? '200px' : isTablet ? '300px' : 'calc(100vh - 900px)',
                    minHeight: '200px',
                    overflowY: 'auto',
                    bgcolor: '#1a1a1a',
                    color: 'white',
                    '&::-webkit-scrollbar': {
                        width: '8px',
                    },
                    '&::-webkit-scrollbar-track': {
                        bgcolor: 'rgba(255, 255, 255, 0.1)',
                    },
                    '&::-webkit-scrollbar-thumb': {
                        bgcolor: 'rgba(255, 255, 255, 0.2)',
                        borderRadius: '4px',
                    },
                }}
            >
                {messages.map((msg, idx) => (
                    <Box
                        key={idx}
                        sx={{
                            mb: 1,
                            textAlign: msg.role === 'system' ? 'left' : msg.role === 'user' ? 'right' : 'left',
                            maxWidth: msg.role === 'system' ? '100%' : isMobile ? '100%' : '80%',
                            marginLeft: msg.role === 'user' ? 'auto' : 0,
                            marginRight: msg.role === 'user' ? 0 : 'auto'
                        }}
                    >
                        <Typography
                            variant={isMobile ? 'body2' : 'body1'}
                            sx={getMessageStyle(msg.role)}
                        >
                            {typeof msg.content === 'object' ? msg.content.answer || msg.content.error : msg.content}
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
