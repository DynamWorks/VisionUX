import React, { useState } from 'react';
import { Box, TextField, Button, Paper, Typography, CircularProgress } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

const Chat = ({ messages, isLoading, onSendMessage }) => {
    const [input, setInput] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim()) {
            onSendMessage(input.trim());
            setInput('');
        }
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Paper 
                sx={{ 
                    flex: 1, 
                    mb: 2, 
                    p: 2, 
                    maxHeight: '400px', 
                    overflow: 'auto',
                    bgcolor: '#f5f5f5'
                }}
            >
                {messages.map((msg, idx) => (
                    <Box 
                        key={idx}
                        sx={{
                            mb: 1,
                            textAlign: msg.role === 'user' ? 'right' : 'left'
                        }}
                    >
                        <Typography
                            component="div"
                            sx={{
                                display: 'inline-block',
                                bgcolor: msg.role === 'user' ? '#bd9544' : '#333333',
                                color: '#ffffff',
                                p: 1,
                                borderRadius: 1,
                                maxWidth: '80%'
                            }}
                        >
                            {msg.content}
                        </Typography>
                    </Box>
                ))}
            </Paper>
            <Box 
                component="form" 
                onSubmit={handleSubmit}
                sx={{ 
                    display: 'flex',
                    gap: 1
                }}
            >
                <TextField
                    fullWidth
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question..."
                    disabled={isLoading}
                    sx={{ bgcolor: '#ffffff' }}
                />
                <Button
                    type="submit"
                    variant="contained"
                    disabled={isLoading || !input.trim()}
                    sx={{ 
                        bgcolor: '#bd9544',
                        '&:hover': {
                            bgcolor: '#a17e3a'
                        }
                    }}
                >
                    {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
                </Button>
            </Box>
        </Box>
    );
};

export default Chat;
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
                    maxHeight: '200px',
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
