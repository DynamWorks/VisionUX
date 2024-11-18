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
