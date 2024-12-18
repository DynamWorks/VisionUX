import { useState, useCallback, useEffect } from 'react';
import { websocketService } from '../services/websocket';
import useStore from '../store';

const useChat = () => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const { currentVideo, setAnalysisResults, isRagEnabled } = useStore();
    
    // Load chat history from backend when video changes
    useEffect(() => {
        if (currentVideo) {
            // Fetch chat history from backend
            fetch(`${process.env.REACT_APP_API_URL}/api/v1/chat/history/${currentVideo.name}`)
                .then(response => response.json())
                .then(data => {
                    if (data.messages) {
                        setMessages(data.messages);
                    }
                })
                .catch(error => {
                    console.error('Error fetching chat history:', error);
                });
        }
    }, [currentVideo]);

    const addMessage = useCallback((role, content) => {
        setMessages(prev => [...prev, { role, content }]);
    }, []);

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
                    use_swarm: true,
                    use_rag: isRagEnabled
                })
            });

            if (!response.ok) {
                throw new Error(`Chat request failed: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Chat response:', data);

            // Add user message
            addMessage('user', message);

            // Add RAG response and handle tool execution
            if (data.rag_response) {
                addMessage('assistant', data.rag_response);
                
                // Check for tool execution
                if (data.tool) {
                    const { setShowEdgeVisualization, setShowObjectVisualization, currentVideo } = useStore.getState();
                    
                    const { setCurrentVisualization } = useStore.getState();
                    setCurrentVisualization(data.visualization);
                    
                    // Toggle appropriate visualization based on tool
                    if (data.tool === 'edge_detection') {
                        setShowEdgeVisualization(true);
                        setShowObjectVisualization(false);
                        setCurrentVisualization(data.visualization);
                    } else if (data.tool === 'object_detection') {
                        setShowObjectVisualization(true);
                        setShowEdgeVisualization(false);
                        setCurrentVisualization(data.visualization);
                    }

                    // Force video reload
                    const videoElement = document.querySelector('video');
                    if (videoElement) {
                        videoElement.load();
                    }
                }
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
    }, [setAnalysisResults]);

    return {
        messages,
        setMessages,
        isLoading,
        sendMessage,
        handleSceneAnalysis,
        clearChat,
        addMessage
    };
};

export default useChat;
