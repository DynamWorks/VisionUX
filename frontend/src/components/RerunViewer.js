import React, { useEffect, useRef } from 'react';
import { WebViewer } from '@rerun-io/web-viewer';
import { Box } from '@mui/material';

const RerunViewer = ({ stream, videoFile }) => {
    const viewerRef = useRef(null);
    const rrdRef = useRef(null);

    useEffect(() => {
        if (viewerRef.current) {
            const viewer = new WebViewer({
                container: viewerRef.current,
                recording: rrdRef.current,
                blueprint: {
                    layout: 'horizontal',
                    widgets: [
                        { type: 'viewer3d', path: 'world' },
                        { type: 'image', path: 'camera' }
                    ]
                }
            });

            return () => {
                viewer.dispose();
            };
        }
    }, []);

    useEffect(() => {
        if (stream) {
            // Handle live stream
            const handleFrame = async () => {
                const track = stream.getVideoTracks()[0];
                const imageCapture = new ImageCapture(track);
                const frame = await imageCapture.grabFrame();
                // Send frame to Rerun viewer
                if (rrdRef.current) {
                    rrdRef.current.logImage('camera', frame);
                }
            };

            const interval = setInterval(handleFrame, 1000 / 30); // 30 FPS
            return () => clearInterval(interval);
        }
    }, [stream]);

    return (
        <Box 
            ref={viewerRef}
            sx={{ 
                width: '100%',
                height: '500px',
                bgcolor: '#000',
                borderRadius: '8px',
                overflow: 'hidden'
            }}
        />
    );
};

export default RerunViewer;
