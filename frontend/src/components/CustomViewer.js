import React, { useRef, useEffect } from 'react';
import { Box } from '@mui/material';

const CustomViewer = ({ websocket }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!websocket) return;

        const handleFrame = async (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'frame_data') {
                    // Wait for binary frame data
                    const frameBlob = await websocket.receive();
                    if (!frameBlob) return;

                    // Convert blob to image
                    const imageUrl = URL.createObjectURL(frameBlob);
                    const img = new Image();
                    img.onload = () => {
                        const canvas = canvasRef.current;
                        if (!canvas) return;

                        const ctx = canvas.getContext('2d');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        URL.revokeObjectURL(imageUrl);
                    };
                    img.src = imageUrl;
                }
            } catch (error) {
                console.error('Error processing frame:', error);
            }
        };

        websocket.addEventListener('message', handleFrame);

        return () => {
            websocket.removeEventListener('message', handleFrame);
        };
    }, [websocket]);

    return (
        <Box sx={{
            width: '100%',
            height: '600px',
            bgcolor: '#1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
        }}>
            <canvas
                ref={canvasRef}
                style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain'
                }}
            />
        </Box>
    );
};

export default CustomViewer;
