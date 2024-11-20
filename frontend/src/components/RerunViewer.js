import React from 'react';
import { Box } from '@mui/material';

const RerunViewer = () => {
    return (
        <Box sx={{
            width: '100%',
            height: '500px',
            bgcolor: '#000',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <iframe
                src={`http://localhost:9091?url=ws://localhost:9090`}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none'
                }}
                title="Rerun Viewer"
                allow="camera"
                sandbox="allow-scripts allow-same-origin allow-forms"
            />
        </Box>
    );
};

export default RerunViewer;
