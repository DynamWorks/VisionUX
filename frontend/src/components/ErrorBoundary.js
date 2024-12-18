import React from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import RefreshIcon from '@mui/icons-material/Refresh';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { 
            hasError: false, 
            error: null,
            errorInfo: null
        };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({
            error,
            errorInfo
        });
        
        // Log error to console with stack trace
        console.error('Error caught by boundary:', error);
        console.error('Component stack:', errorInfo.componentStack);
    }

    handleReset = () => {
        this.setState({ 
            hasError: false, 
            error: null,
            errorInfo: null 
        });
        
        // Attempt to reset the component
        if (this.props.onReset) {
            this.props.onReset();
        }
    };

    handleReload = () => {
        window.location.reload();
    };

    render() {
        if (this.state.hasError) {
            return (
                <Box sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    minHeight: '200px',
                    p: 3
                }}>
                    <Paper 
                        elevation={3}
                        sx={{
                            p: 3,
                            maxWidth: '600px',
                            width: '100%',
                            bgcolor: '#1a1a1a',
                            color: 'white'
                        }}
                    >
                        <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: 2,
                            mb: 2
                        }}>
                            <ErrorOutlineIcon color="error" sx={{ fontSize: 40 }} />
                            <Typography variant="h5" component="h2">
                                Something went wrong
                            </Typography>
                        </Box>

                        <Typography variant="body1" sx={{ mb: 2 }}>
                            An error occurred in the application. You can try resetting the component or reloading the page.
                        </Typography>

                        {this.state.error && (
                            <Paper 
                                sx={{ 
                                    p: 2, 
                                    mb: 2, 
                                    bgcolor: 'rgba(255, 255, 255, 0.05)',
                                    maxHeight: '200px',
                                    overflow: 'auto'
                                }}
                            >
                                <Typography 
                                    variant="body2" 
                                    component="pre"
                                    sx={{ 
                                        whiteSpace: 'pre-wrap',
                                        wordBreak: 'break-word',
                                        m: 0
                                    }}
                                >
                                    {this.state.error.toString()}
                                    {this.state.errorInfo && this.state.errorInfo.componentStack}
                                </Typography>
                            </Paper>
                        )}

                        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                            <Button
                                variant="outlined"
                                startIcon={<RefreshIcon />}
                                onClick={this.handleReset}
                                sx={{ color: 'white', borderColor: 'white' }}
                            >
                                Reset Component
                            </Button>
                            <Button
                                variant="contained"
                                color="primary"
                                onClick={this.handleReload}
                            >
                                Reload Page
                            </Button>
                        </Box>
                    </Paper>
                </Box>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
