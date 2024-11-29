import { useState, useCallback, useEffect } from 'react';
import { websocketService } from '../services/websocket';
import useStore from '../store';

const useChat = () => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const { currentVideo, setAnalysisResults } = useStore();
    
    // Load chat history for current video
    useEffect(() => {
        if (currentVideo) {
            const savedMessages = localStorage.getItem(`chat_${currentVideo.name}`);
            if (savedMessages) {
                setMessages(JSON.parse(savedMessages));
            } else {
                setMessages([]);
            }
        }
    }, [currentVideo]);

    const addMessage = useCallback((role, content) => {
        setMessages(prev => {
            const newMessages = [...prev, { role, content }];
            if (currentVideo) {
                localStorage.setItem(`chat_${currentVideo.name}`, JSON.stringify(newMessages));
            }
            return newMessages;
        });
    }, [currentVideo]);

    const handleSceneAnalysis = useCallback(async (analysisData) => {
        if (!analysisData?.scene_analysis?.description) {
            console.warn('No scene analysis description in response');
            return;
        }

        const description = analysisData.scene_analysis.description;
        const videoFile = analysisData.storage?.video_file || 'unknown video';
        const timestamp = new Date(analysisData.storage?.timestamp * 1000).toLocaleString();
        
        addMessage('system', `Scene Analysis for ${videoFile} at ${timestamp}:\n${description}`);

        if (analysisData.results) {
            setAnalysisResults(analysisData.results);
        }
    }, [addMessage, setAnalysisResults]);

    const sendMessage = useCallback(async (message) => {
        setIsLoading(true);
        try {
            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/v1/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: message,
                    video_path: currentVideo?.name || 'current',
                    use_swarm: true
                })
            });

            if (!response.ok) {
                throw new Error(`Chat request failed: ${response.statusText}`);
            }

            const data = await response.json();

            // Add user message
            addMessage('user', message);

            // Add RAG response if available
            if (data.rag_response) {
                addMessage('assistant', data.rag_response);
            }

            // Add analysis results if available
            if (data.results && Object.keys(data.results).length > 0) {
                handleSceneAnalysis(data);
            }

            // Add scene analysis chat message if available
            if (data.chat_message) {
                addMessage(data.chat_message.role, data.chat_message.content);
            }

            return data;

        } catch (error) {
            console.error('Chat error:', error);
            addMessage('error', `Error: ${error.message}`);
            throw error;
        } finally {
            setIsLoading(false);
        }
    }, [currentVideo, addMessage, handleSceneAnalysis, setAnalysisResults]);

    const clearChat = useCallback(() => {
        setMessages([]);
        setAnalysisResults(null);
        // Clear local storage
        if (currentVideo) {
            localStorage.removeItem(`chat_${currentVideo.name}`);
        }
    }, [setAnalysisResults, currentVideo]);

    return {
        messages,
        isLoading,
        sendMessage,
        handleSceneAnalysis,
        clearChat,
        addMessage
    };
};

export default useChat;
