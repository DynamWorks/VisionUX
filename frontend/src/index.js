import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './index.css';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <ErrorBoundary>
            <ThemeProvider theme={theme}>
                <CssBaseline />
                <BrowserRouter>
                    <App />
                </BrowserRouter>
            </ThemeProvider>
        </ErrorBoundary>
    </React.StrictMode>
);
