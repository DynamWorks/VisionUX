import React from 'react';
import { Box, Typography, Link } from '@mui/material';

const Footer = () => {
    const currentYear = new Date().getFullYear();
    
    return (
        <Box 
            component="footer" 
            sx={{ 
                bgcolor: 'background.paper',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                py: 2,
                mt: 'auto',
                textAlign: 'center'
            }}
        >
            <Typography variant="body2" color="text.secondary">
                © {currentYear} DynamWorks LLC | All Rights Reserved |{' '}
                <Link 
                    href="https://www.dynamworks.com/privacy-policy"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{ 
                        color: 'primary.main',
                        textDecoration: 'none',
                        '&:hover': {
                            textDecoration: 'underline'
                        }
                    }}
                >
                    Privacy Policy
                </Link>
            </Typography>
            <Box sx={{ mt: 1 }}>
                <Link
                    href="https://www.dynamworks.com/contact"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
                        color: 'primary.main',
                        textDecoration: 'none',
                        '&:hover': {
                            textDecoration: 'underline'
                        }
                    }}
                >
                    Contact Us
                </Link>
            </Box>
        </Box>
    );
};

export default Footer;
