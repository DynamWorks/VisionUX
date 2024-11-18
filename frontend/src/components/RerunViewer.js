import React, { useEffect, useRef } from 'react';
import { Viewer } from '@rerun-io/web-viewer';

const RerunViewer = ({ stream, isStreaming }) => {
    const containerRef = useRef(null);
    const viewerRef = useRef(null);

    useEffect(() => {
        if (containerRef.current && !viewerRef.current) {
            viewerRef.current = new Viewer({
                container: containerRef.current,
                onInit: () => console.log('Rerun viewer initialized'),
                onError: (error) => console.error('Rerun viewer error:', error)
            });
        }

        return () => {
            if (viewerRef.current) {
                viewerRef.current.destroy();
                viewerRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (viewerRef.current && stream && isStreaming) {
            // TODO: Implement stream handling
            // viewerRef.current.addImage({
            //     name: 'camera_feed',
            //     data: stream,
            //     timestamp: Date.now()
            // });
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
