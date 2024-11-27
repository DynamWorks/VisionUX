import { create } from 'zustand';

const useStore = create((set) => ({
    // Input mode
    inputMode: 'upload', // 'upload' or 'camera'
    setInputMode: (mode) => set({ 
        inputMode: mode,
        // Reset video state when switching modes
        currentVideo: null,
        isStreaming: false
    }),

    // Video state
    uploadedFiles: [], // Initialize as empty array instead of null
    currentVideo: null,
    isStreaming: false,
    streamMetrics: {
        fps: 0,
        frameCount: 0,
        resolution: '',
        timestamp: null
    },

    // Video actions
    setUploadedFiles: (files) => set({ uploadedFiles: files || [] }), // Ensure array
    setCurrentVideo: (video) => set({ currentVideo: video }),
    setIsStreaming: (isStreaming) => set({ isStreaming }),
    setStreamMetrics: (metrics) => set({ streamMetrics: metrics }),

    // Reset state
    resetVideoState: () => set({
        currentVideo: null,
        isStreaming: false,
        streamMetrics: {
            fps: 0,
            frameCount: 0,
            resolution: '',
            timestamp: null
        }
    }),

    // Chat state
    chatHistory: [],
    isChatLoading: false,
    chatError: null,

    // Chat actions
    setChatHistory: (history) => set({ chatHistory: history }),
    addChatMessage: (message) => set(state => ({
        chatHistory: [...state.chatHistory, message]
    })),
    setIsChatLoading: (loading) => set({ isChatLoading: loading }),
    setChatError: (error) => set({ chatError: error }),

    // Reset chat state
    resetChatState: () => set({
        chatHistory: [],
        isChatLoading: false,
        chatError: null
    }),

    // Global reset
    resetState: () => set({
        inputMode: 'upload',
        uploadedFiles: [],
        currentVideo: null,
        isStreaming: false,
        streamMetrics: {
            fps: 0,
            frameCount: 0,
            resolution: '',
            timestamp: null
        },
        chatHistory: [],
        isChatLoading: false,
        chatError: null
    })
}));

export default useStore;
