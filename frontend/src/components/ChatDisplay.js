import React from 'react';
import { Box, Paper, Typography } from '@mui/material';

const ChatDisplay = ({ messages }) => {
    return (
        <Paper 
            sx={{ 
                p: 2, 
                mt: 2, 
                maxHeight: '300px',
                overflowY: 'auto',
                bgcolor: '#1a1a1a',
                color: 'white'
            }}
        >
            <Typography variant="h6" sx={{ mb: 2 }}>Analysis Results</Typography>
            {messages.map((msg, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                    <Typography 
                        variant="body1" 
                        sx={{ 
                            whiteSpace: 'pre-wrap',
                            color: msg.type === 'error' ? '#ff6b6b' : 'inherit'
                        }}
                    >
                        {msg.content}
                    </Typography>
                    {msg.timestamp && (
                        <Typography variant="caption" color="text.secondary">
                            {new Date(msg.timestamp).toLocaleTimeString()}
                        </Typography>
                    )}
                </Box>
            ))}
        </Paper>
    );
};

export default ChatDisplay;
