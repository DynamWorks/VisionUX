import React from 'react';
import { Box } from '@mui/material';
import StreamRenderer from './StreamRenderer';
import VideoPlayer from './VideoPlayer';
import useStore from '../store';

const CustomViewer = () => {
    const { currentVideo, inputMode, isStreaming } = useStore();

    return (
        <Box
            sx={{
                width: '100%',
                height: '100%',
                minHeight: '300px',
                maxWidth: 'none', // Remove max-width constraint
                margin: '0 auto',
                bgcolor: '#1a1a1a',
                borderRadius: '8px',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                flex: 1 // Add flex grow
            }}
        >
            {inputMode === 'camera' ? (
                <StreamRenderer
                    source="camera"
                    isStreaming={isStreaming}
                />
            ) : (
                currentVideo ? (
                    <VideoPlayer
                        file={currentVideo}
                    />
                ) : (
                    <StreamRenderer
                        source="video"
                        isStreaming={false}
                    />
                )
            )}
        </Box>
    );
};

export default CustomViewer;
