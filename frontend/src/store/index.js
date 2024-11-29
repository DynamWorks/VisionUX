import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useStore = create(
    persist(
        (set) => ({
    // Input mode
    inputMode: 'upload', // 'upload' or 'camera'
    setInputMode: (mode) => set({
        inputMode: mode,
        // Reset video state when switching modes
        currentVideo: null,
        isStreaming: false
    }),

    // Video state
    uploadedFiles: [],
    currentVideo: null,
    isStreaming: false,
    streamMetrics: {
        fps: 0,
        frameCount: 0,
        resolution: '',
        timestamp: null
    },

    // Analysis state
    analysisResults: null,
    isAnalyzing: false,
    analysisError: null,
    isEdgeDetectionEnabled: false,
    autoAnalysisEnabled: false,
    isRagEnabled: false,

    // Video actions
    setUploadedFiles: (files) => set({ uploadedFiles: Array.isArray(files) ? files : [] }),
    setCurrentVideo: (video) => set({ currentVideo: video }),
    setIsStreaming: (isStreaming) => set({ isStreaming }),
    setStreamMetrics: (metrics) => set({ streamMetrics: metrics }),

    // Analysis actions
    setAnalysisResults: (results) => set({ analysisResults: results }),
    setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
    setAnalysisError: (error) => set({ analysisError: error }),
    setEdgeDetectionEnabled: (enabled) => set({ isEdgeDetectionEnabled: enabled }),
    setAutoAnalysisEnabled: (enabled) => set({ autoAnalysisEnabled: enabled }),
    setRagEnabled: (enabled) => set({ isRagEnabled: enabled }),

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
        analysisResults: null,
        isAnalyzing: false,
        analysisError: null,
        chatHistory: [],
        isChatLoading: false,
        chatError: null
    })
}), {
    name: 'video-analytics-storage',
    partialize: (state) => ({
        currentVideo: state.currentVideo,
        uploadedFiles: state.uploadedFiles
    })
}));

export default useStore;
