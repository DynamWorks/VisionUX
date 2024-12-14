import React, { useEffect } from 'react';
import useStore from '../store';

const InputSelector = () => {
    const { setInputMode } = useStore();

    // Set input mode to upload on mount
    useEffect(() => {
        setInputMode('upload');
    }, [setInputMode]);

    // Return null since we're hiding the selector
    return null;
};

export default InputSelector;
