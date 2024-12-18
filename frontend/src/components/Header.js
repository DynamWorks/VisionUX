import React from 'react';
import { AppBar, Toolbar, Box, Button } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import logo from '../assets/logo.png';

const Header = ({ onRestartWebSockets }) => {
    return (
        <AppBar position="static" sx={{ bgcolor: '#333333' }}>
            <Toolbar>
                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', py: 2 }}>
                    <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
                        <img 
                            src={logo} 
                            alt="Video Analytics Logo" 
                            style={{ 
                                height: '40px',
                                width: 'auto'
                            }}
                        />
                    </Box>
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<RefreshIcon />}
                        onClick={onRestartWebSockets}
                        sx={{ color: 'white', borderColor: 'white' }}
                    >
                        Restart WebSocket
                    </Button>
                </Box>
            </Toolbar>
        </AppBar>
    );
};

export default Header;
