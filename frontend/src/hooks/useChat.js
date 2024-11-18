import { useState, useCallback } from 'react';
import axios from 'axios';

const useChat = () => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = useCallback(async (message) => {
        setIsLoading(true);
        try {
            const response = await axios.post('http://localhost:8000/api/chat', {
                message,
                context: messages
            });
            
            const newMessages = [
                ...messages,
                { role: 'user', content: message },
                { role: 'assistant', content: response.data.response }
            ];
            setMessages(newMessages);
            return response.data.response;
        } catch (error) {
            console.error('Chat error:', error);
            throw error;
        } finally {
            setIsLoading(false);
        }
    }, [messages]);

    const clearChat = useCallback(() => {
        setMessages([]);
    }, []);

    return {
        messages,
        isLoading,
        sendMessage,
        clearChat
    };
};

export default useChat;
