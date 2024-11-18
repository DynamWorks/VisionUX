import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';

const Header = () => {
    return (
        <AppBar position="static" sx={{ bgcolor: '#333333' }}>
            <Toolbar>
                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', py: 2 }}>
                    <Box sx={{ flexGrow: 1 }}>
                        <Typography 
                            variant="h4" 
                            component="h1" 
                            sx={{ 
                                color: '#bd9544',
                                fontWeight: 'bold'
                            }}
                        >
                            Video Analytics
                        </Typography>
                    </Box>
                    <Button 
                        href="/contact"
                        variant="contained"
                        sx={{
                            bgcolor: '#bd9544',
                            '&:hover': {
                                bgcolor: '#a17e3a'
                            },
                            borderRadius: '30px'
                        }}
                    >
                        Contact
                    </Button>
                </Box>
            </Toolbar>
        </AppBar>
    );
};

export default Header;
