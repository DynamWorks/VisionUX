import create from 'zustand';

const useStore = create((set) => ({
  // WebSocket state
  ws: null,
  setWs: (ws) => set({ ws }),
  
  // Video state
  isStreaming: false,
  setIsStreaming: (isStreaming) => set({ isStreaming }),
  currentVideo: null,
  setCurrentVideo: (currentVideo) => set({ currentVideo }),
  
  // File state
  uploadedFiles: [],
  setUploadedFiles: (uploadedFiles) => set({ uploadedFiles }),
  
  // UI state
  inputType: 'camera',
  setInputType: (inputType) => set({ inputType }),
  
  // Camera state
  selectedDevice: null,
  setSelectedDevice: (selectedDevice) => set({ selectedDevice }),
  devices: [],
  setDevices: (devices) => set({ devices }),
  
  // Analysis state
  analysisResults: null,
  setAnalysisResults: (analysisResults) => set({ analysisResults }),
}));

export default useStore;
