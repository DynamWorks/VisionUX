import React from 'react';
import { Box } from '@mui/material';

const RERUN_VERSION = '0.20.0';  // Match your @rerun-io/web-viewer version
const RRD_URL = 'ws://127.0.0.1:4321';  // Use explicit IP address

const RerunViewer = () => {
    const iframeUrl = `https://app.rerun.io/version/${RERUN_VERSION}/index.html?url=${encodeURIComponent(RRD_URL)}`;
    
    return (
        <Box sx={{ 
            width: '100%',
            height: '500px',
            bgcolor: '#000',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <iframe
                src={iframeUrl}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none'
                }}
                title="Rerun Viewer"
            />
        </Box>
    );
};

export default RerunViewer;
