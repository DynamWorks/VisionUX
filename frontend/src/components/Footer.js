import React from 'react';
import { Box, Typography, Link } from '@mui/material';

const Footer = () => {
    const currentYear = new Date().getFullYear();
    
    return (
        <Box 
            component="footer" 
            sx={{ 
                bgcolor: '#333333',
                color: '#bd9544',
                py: 2,
                textAlign: 'center'
            }}
        >
            <Typography variant="body2">
                &copy; {currentYear} Video Analytics | All Rights Reserved |{' '}
                <Link 
                    href="/privacy"
                    sx={{ 
                        color: '#bd9544',
                        textDecoration: 'none',
                        '&:hover': {
                            textDecoration: 'underline'
                        }
                    }}
                >
                    Privacy Policy
                </Link>
            </Typography>
        </Box>
    );
};

export default Footer;
