import React from 'react';
import { Box } from '@mui/material';

const RerunViewer = () => {
    const rerunWebUrl = process.env.REACT_APP_RERUN_WEB_URL || 'http://localhost:9090';

    return (
        <Box sx={{
            width: '100%',
            height: '600px',
            bgcolor: '#1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden'
        }}>
            <iframe
                src={`${rerunWebUrl}`}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    backgroundColor: '#1a1a1a'
                }}
                title="Rerun Viewer"
                allow="camera"
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-presentation"
            />
        </Box>
    );
};

export default RerunViewer;
