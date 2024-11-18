import React, { useEffect, useRef } from 'react';
import { WebViewer } from '@rerun-io/web-viewer';

const RerunViewer = ({ stream, isStreaming }) => {
    const containerRef = useRef(null);
    const viewerRef = useRef(null);

    useEffect(() => {
        const initViewer = async () => {
            if (containerRef.current && !viewerRef.current) {
                viewerRef.current = new WebViewer();
                await viewerRef.current.start(null, containerRef.current);
            }
        };

        initViewer();

        return () => {
            if (viewerRef.current) {
                viewerRef.current.dispose();
                viewerRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (viewerRef.current && stream && isStreaming) {
            // Handle stream data here
            // You'll need to implement the appropriate data handling
            // based on your stream format
        }
    }, [stream, isStreaming]);

    return (
        <div 
            ref={containerRef}
            style={{
                width: '100%',
                height: '500px',
                borderRadius: '8px',
                overflow: 'hidden'
            }}
        />
    );
};

export default RerunViewer;
