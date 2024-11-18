import React, { useEffect, useRef, useCallback } from 'react';
import { WebViewer } from '@rerun-io/web-viewer';

const RerunViewer = ({ stream, isStreaming, videoFile }) => {
    const wsRef = useRef(null);
    const viewerRef = useRef(null);
    const frameRef = useRef(0);

    const initWebSocket = useCallback(() => {
        const ws = new WebSocket(`ws://${process.env.REACT_APP_API_URL.replace('http://', '')}/stream`);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('WebSocket connected');
            if (viewerRef.current) {
                viewerRef.current.clear();
                frameRef.current = 0;
            }
        };

        ws.onmessage = (event) => {
            if (event.data instanceof Blob) {
                const reader = new FileReader();
                reader.onload = () => {
                    if (viewerRef.current) {
                        const frameData = new Uint8Array(reader.result);
                        viewerRef.current.logImage("camera/frames", {
                            data: frameData,
                            frameNr: frameRef.current++,
                            timestamp: performance.now() * 1e6 // Convert to nanoseconds
                        });
                    }
                };
                reader.readAsArrayBuffer(event.data);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        return ws;
    }, []);

    const processVideoFrame = useCallback(async (imageCapture) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !isStreaming) {
            return;
        }

        try {
            const frame = await imageCapture.grabFrame();
            const blob = await new Promise(resolve => {
                const canvas = document.createElement('canvas');
                canvas.width = frame.width;
                canvas.height = frame.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(frame, 0, 0);
                canvas.toBlob(resolve, 'image/jpeg', 0.8);
            });
            wsRef.current.send(blob);
            requestAnimationFrame(() => processVideoFrame(imageCapture));
        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }, [isStreaming]);

    useEffect(() => {
        if (videoFile) {
            const ws = initWebSocket();
            ws.onopen = async () => {
                console.log('WebSocket connected for video upload');
                const arrayBuffer = await videoFile.arrayBuffer();
                ws.send(new Blob([arrayBuffer], { type: videoFile.type }));
            };
        } else if (stream && isStreaming) {
            const ws = initWebSocket();
            const videoTrack = stream.getVideoTracks()[0];
            const imageCapture = new ImageCapture(videoTrack);
            processVideoFrame(imageCapture);
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [stream, isStreaming, videoFile, initWebSocket, processVideoFrame]);

    return (
        <WebViewer
            ref={viewerRef}
            width="100%"
            height="500px"
            blueprint={{
                name: "Camera Feed",
                components: {
                    "camera/frames": {
                        type: "image",
                        shape: [720, 1280, 3],
                        stream: true
                    }
                }
            }}
            style={{
                borderRadius: '8px',
                overflow: 'hidden'
            }}
            onInit={() => {
                console.log('Rerun viewer initialized');
                viewerRef.current?.clear();
                frameRef.current = 0;
            }}
            onError={(error) => {
                console.error('Rerun viewer error:', error);
                // Attempt recovery
                if (viewerRef.current) {
                    viewerRef.current.clear();
                    frameRef.current = 0;
                }
            }}
        />
    );
};

export default RerunViewer;
