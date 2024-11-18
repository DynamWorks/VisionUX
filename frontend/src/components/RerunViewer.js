import React, { useEffect, useState } from 'react';
import WebViewer from '@rerun-io/web-viewer-react';

const RerunViewer = ({ stream, isStreaming }) => {
    const [rrdUrl, setRrdUrl] = useState(null);

    useEffect(() => {
        if (stream && isStreaming) {
            // TODO: Handle stream data and set rrdUrl when WebSocket connection is established
            console.log('Stream active:', stream);
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
