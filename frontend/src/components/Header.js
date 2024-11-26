import React from 'react';
import { AppBar, Toolbar, Box } from '@mui/material';
import RestartControls from './RestartControls';
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
                    <RestartControls 
                        onRestartWebSockets={onRestartWebSockets}
                    />
                </Box>
            </Toolbar>
        </AppBar>
    );
};

export default Header;
