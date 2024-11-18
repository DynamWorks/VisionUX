import React, { useEffect } from 'react';
import { RerunViewer as Viewer } from '@rerun-io/web-viewer-react';

const RerunViewer = ({ stream, isStreaming }) => {
    useEffect(() => {
        if (stream && isStreaming) {
            // TODO: Handle stream data
            console.log('Stream active:', stream);
        }
    }, [stream, isStreaming]);

    return (
        <Viewer
            style={{
                width: '100%',
                height: '500px',
                borderRadius: '8px',
                overflow: 'hidden'
            }}
            onInit={() => console.log('Rerun viewer initialized')}
            onError={(error) => console.error('Rerun viewer error:', error)}
        />
    );
};

export default RerunViewer;
