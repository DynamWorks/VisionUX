import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-wasm';

async function init() {
    try {
        await tf.ready();
        console.log('TensorFlow.js backend initialized:', tf.getBackend());

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(
            <React.StrictMode>
                <App />
            </React.StrictMode>
        );
    } catch (error) {
        console.error('Failed to initialize the app:', error);
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(
            <React.StrictMode>
                <App />
            </React.StrictMode>
        );
    }
}

init();
