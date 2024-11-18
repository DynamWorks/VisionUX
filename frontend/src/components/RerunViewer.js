import React, { useEffect, useState, useRef } from 'react';
import { WebViewer } from '@rerun-io/web-viewer';

const RerunViewer = ({ stream, isStreaming }) => {
    const [rrdUrl, setRrdUrl] = useState(null);
    const wsRef = useRef(null);

    useEffect(() => {
        if (stream && isStreaming) {
            // Connect to WebSocket for stream processing
            const ws = new WebSocket('ws://localhost:8000/stream');
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connected');
                // Start sending video frames
                const videoTrack = stream.getVideoTracks()[0];
                const imageCapture = new ImageCapture(videoTrack);
                
                const sendFrame = async () => {
                    if (ws.readyState === WebSocket.OPEN && isStreaming) {
                        try {
                            const frame = await imageCapture.grabFrame();
                            const blob = await new Promise(resolve => {
                                const canvas = document.createElement('canvas');
                                canvas.width = frame.width;
                                canvas.height = frame.height;
                                const ctx = canvas.getContext('2d');
                                ctx.drawImage(frame, 0, 0);
                                canvas.toBlob(resolve, 'image/jpeg', 0.8);
                            });
                            ws.send(blob);
                            requestAnimationFrame(sendFrame);
                        } catch (error) {
                            console.error('Frame capture error:', error);
                        }
                    }
                };
                sendFrame();
            };

            ws.onmessage = (event) => {
                // Assuming server sends back RRD URL
                const data = JSON.parse(event.data);
                if (data.rrdUrl) {
                    setRrdUrl(data.rrdUrl);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            return () => {
                if (wsRef.current) {
                    wsRef.current.close();
                }
            };
        }
    }, [stream, isStreaming]);

    return (
        <WebViewer
            width="100%"
            height="500px"
            rrd={rrdUrl}
            hide_welcome_screen={!rrdUrl}
            style={{
                borderRadius: '8px',
                overflow: 'hidden'
            }}
            onInit={() => console.log('Rerun viewer initialized')}
            onError={(error) => console.error('Rerun viewer error:', error)}
        />
    );
};

export default RerunViewer;
