import { io } from 'socket.io-client';

class WebSocketService {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.listeners = new Map();
        this._connected = false;
        this.pingInterval = null;
        this.connectionCheckInterval = null;
        this.lastPongTime = Date.now();
        this.frameBuffer = [];
        this.maxBufferSize = 5;
        this.frameCount = 0;
        this.fpsCalculator = {
            frames: 0,
            lastCheck: Date.now(),
            currentFps: 0
        };
    }

    connect() {
        if (this.socket?.connected) {
            console.log('WebSocket already connected');
            this._connected = true;
            return;
        }

        const wsUrl = process.env.REACT_APP_WS_URL || 'http://localhost:8000';
        console.log('Connecting to WebSocket:', wsUrl);

        this.socket = io(wsUrl, {
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: this.maxReconnectAttempts,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            timeout: 20000,
            autoConnect: true,
            path: '/socket.io/',
            withCredentials: true,
            pingTimeout: 60000,
            pingInterval: 25000,
            extraHeaders: {
                'User-Agent': navigator.userAgent
            }
        });

        this.setupEventHandlers();
        this.startConnectionCheck();
    }

    setupEventHandlers() {
        if (!this.socket) return;

        // Connection events
        this.socket.on('connect', () => {
            console.log('WebSocket Connected');
            this._connected = true;
            this.reconnectAttempts = 0;
            this.startPingInterval();

            // Send initial client info
            this.socket.emit('client_info', {
                timestamp: Date.now(),
                userAgent: navigator.userAgent,
                screenResolution: `${window.screen.width}x${window.screen.height}`,
                connectionType: navigator.connection?.type || 'unknown'
            });

            this.notifyListeners('connect');
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this._connected = false;
            this.clearPingInterval();
            this.handleReconnect();
            this.notifyListeners('connect_error', error);
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            this._connected = false;
            this.clearPingInterval();
            this.notifyListeners('disconnect', reason);

            if (reason === 'io server disconnect') {
                setTimeout(() => {
                    this.socket.connect();
                }, 1000);
            }
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            if (!this.socket.connected) {
                this._connected = false;
                this.handleReconnect();
            }
            this.notifyListeners('error', {
                message: error.message || 'Socket error',
                timestamp: Date.now()
            });
        });

        // Stream events
        this.socket.on('frame', (frameData) => {
            this.handleFrame(frameData);
        });

        this.socket.on('stream_started', (data) => {
            console.log('Stream started:', data);
            this.notifyListeners('stream_started', data);
        });

        this.socket.on('stream_stopped', (data) => {
            console.log('Stream stopped:', data);
            this.notifyListeners('stream_stopped', data);
        });

        this.socket.on('stream_error', (error) => {
            console.error('Stream error:', error);
            this.notifyListeners('stream_error', error);
        });

        this.socket.on('frame_metadata', (metadata) => {
            this.updateStreamMetrics(metadata);
            this.notifyListeners('frame_metadata', metadata);
        });

        // Keep-alive mechanism
        this.socket.on('pong', () => {
            this.lastPongTime = Date.now();
            this.notifyListeners('pong');
        });
    }

    handleFrame(frameData) {
        try {
            if (!frameData) {
                console.warn('Empty frame data received');
                return;
            }

            let frame;
            // Handle different frame data formats
            if (frameData instanceof ArrayBuffer) {
                frame = new Blob([frameData], { type: 'image/jpeg' });
            } else if (frameData instanceof Uint8Array) {
                frame = new Blob([frameData.buffer], { type: 'image/jpeg' });
            } else if (frameData instanceof Blob) {
                frame = frameData;
            } else {
                console.warn('Invalid frame data format:', typeof frameData);
                return;
            }

            // Update frame count and FPS
            this.frameCount++;
            this.updateFps();

            // Notify listeners
            this.notifyListeners('frame', frame);

        } catch (error) {
            console.error('Frame handling error:', error);
            this.notifyListeners('frame_error', error);
        }
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
            console.log(`Reconnecting... Attempt ${this.reconnectAttempts} in ${delay}ms`);

            setTimeout(() => {
                if (this.socket) {
                    this.socket.connect();
                }
            }, delay);
        } else {
            console.error('Max reconnection attempts reached');
            this.notifyListeners('max_reconnects_reached');
            this.cleanup();
        }
    }

    startConnectionCheck() {
        this.clearConnectionCheck();
        this.connectionCheckInterval = setInterval(() => {
            const isReallyConnected = this.socket?.connected && this._connected;
            if (this._connected !== isReallyConnected) {
                this._connected = isReallyConnected;
                this.notifyListeners('connection_change', isReallyConnected);
            }
        }, 1000);
    }

    startPingInterval() {
        this.clearPingInterval();
        this.lastPongTime = Date.now();

        this.pingInterval = setInterval(() => {
            if (this.socket?.connected) {
                this.socket.emit('ping');

                const timeSinceLastPong = Date.now() - this.lastPongTime;
                if (timeSinceLastPong > 120000) { // 2 minutes
                    console.warn('No pong received in 120s, reconnecting...');
                    this._connected = false;
                    this.clearPingInterval();
                    this.socket.disconnect();
                    setTimeout(() => this.socket.connect(), 1000);
                }
            }
        }, 15000); // Every 15 seconds
    }

    clearPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    clearConnectionCheck() {
        if (this.connectionCheckInterval) {
            clearInterval(this.connectionCheckInterval);
            this.connectionCheckInterval = null;
        }
    }

    updateFps() {
        const now = Date.now();
        const elapsed = now - this.fpsCalculator.lastCheck;

        this.fpsCalculator.frames++;

        if (elapsed >= 1000) {
            this.fpsCalculator.currentFps = Math.round((this.fpsCalculator.frames * 1000) / elapsed);
            this.fpsCalculator.frames = 0;
            this.fpsCalculator.lastCheck = now;

            this.notifyListeners('fps_update', this.fpsCalculator.currentFps);
        }
    }

    updateStreamMetrics(metadata) {
        const metrics = {
            fps: this.fpsCalculator.currentFps,
            frameCount: this.frameCount,
            timestamp: Date.now(),
            ...metadata
        };

        this.notifyListeners('stream_metrics', metrics);
    }

    // Event handling methods
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).delete(callback);
        }
    }

    emit(event, data) {
        if (!this.socket?.connected) {
            console.warn('Socket not connected, cannot emit:', event);
            return false;
        }

        try {
            this.socket.emit(event, data);
            return true;
        } catch (error) {
            console.error('Error emitting event:', error);
            return false;
        }
    }

    notifyListeners(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in listener for ${event}:`, error);
                }
            });
        }
    }

    // Cleanup and status methods
    cleanup() {
        this.clearPingInterval();
        this.clearConnectionCheck();

        // Clear frame buffer
        this.frameBuffer = [];

        // Reset metrics
        this.frameCount = 0;
        this.fpsCalculator = {
            frames: 0,
            lastCheck: Date.now(),
            currentFps: 0
        };
    }

    disconnect() {
        this.cleanup();
        if (this.socket) {
            this._connected = false;
            this.socket.disconnect();
            this.socket = null;
        }
    }

    isConnected() {
        return this.socket?.connected && this._connected;
    }

    getConnectionStatus() {
        return {
            connected: this.isConnected(),
            socket: !!this.socket,
            attempts: this.reconnectAttempts,
            lastPong: this.lastPongTime,
            fps: this.fpsCalculator.currentFps,
            frameCount: this.frameCount
        };
    }

    getFrameBuffer() {
        return [...this.frameBuffer];
    }
}

// Create singleton instance
export const websocketService = new WebSocketService();
