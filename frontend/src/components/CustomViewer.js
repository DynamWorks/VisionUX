import React from 'react';
import StreamRenderer from './StreamRenderer';
import useStore from '../store';

const CustomViewer = () => {
    const { currentVideo, inputMode, isStreaming } = useStore();

    return (
        <StreamRenderer 
            source={inputMode === 'camera' ? 'camera' : 'video'}
            isStreaming={isStreaming || (inputMode === 'upload' && !!currentVideo)}
        />
    );
};

export default CustomViewer;
