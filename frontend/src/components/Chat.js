import React, { useState } from 'react';
import { Box, TextField, Button, Paper, Typography } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

const Chat = () => {
    const [message, setMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);

    const handleSend = () => {
        if (message.trim()) {
            // Add user message to chat
            setChatHistory([...chatHistory, { text: message, sender: 'user' }]);
            setMessage('');
            
            // TODO: Send message to backend and handle response
            // For now, just echo
            setTimeout(() => {
                setChatHistory(prev => [...prev, { 
                    text: `Received: ${message}`, 
                    sender: 'assistant'
                }]);
            }, 500);
        }
    };

    return (
        <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Paper 
                sx={{ 
                    p: 2, 
                    height: 'calc(100vh - 300px)',
                    overflowY: 'auto',
                    bgcolor: '#1a1a1a',
                    color: 'white'
                }}
            >
                {chatHistory.map((msg, idx) => (
                    <Box 
                        key={idx} 
                        sx={{ 
                            mb: 1,
                            textAlign: msg.sender === 'user' ? 'right' : 'left'
                        }}
                    >
                        <Typography 
                            variant="body2"
                            sx={{
                                display: 'inline-block',
                                bgcolor: msg.sender === 'user' ? '#2c5282' : '#4a5568',
                                p: 1,
                                borderRadius: 1,
                                maxWidth: '80%'
                            }}
                        >
                            {msg.text}
                        </Typography>
                    </Box>
                ))}
            </Paper>
            <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Type a message..."
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSend();
                        }
                    }}
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
                <Button
                    variant="contained"
                    endIcon={<SendIcon />}
                    onClick={handleSend}
                    disabled={!message.trim()}
                >
                    Send
                </Button>
            </Box>
        </Box>
    );
};

export default Chat;
