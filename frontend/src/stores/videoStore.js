import create from 'zustand';

const useVideoStore = create((set) => ({
    // Stream metrics
    streamMetrics: {
        fps: 0,
        frameCount: 0,
        resolution: '',
        bitrate: 0
    },
    setStreamMetrics: (metrics) => set((state) => ({
        streamMetrics: { ...state.streamMetrics, ...metrics }
    })),

    // Stream status
    isStreaming: false,
    setIsStreaming: (isStreaming) => set({ isStreaming }),
    
    // Error state
    error: null,
    setError: (error) => set({ error }),
    
    // Reset state
    reset: () => set({
        streamMetrics: {
            fps: 0,
            frameCount: 0,
            resolution: '',
            bitrate: 0
        },
        isStreaming: false,
        error: null
    })
}));

export default useVideoStore;
