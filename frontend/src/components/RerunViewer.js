import React from 'react';
import { WebViewer } from '@rerun-io/web-viewer-react';

const RerunViewer = ({ stream, isStreaming }) => {
    return (
        <WebViewer
            style={{
                width: '100%',
                height: '500px',
                borderRadius: '8px',
                overflow: 'hidden'
            }}
            recording={stream}
            autoConnect={isStreaming}
        />
    );
};

export default RerunViewer;
