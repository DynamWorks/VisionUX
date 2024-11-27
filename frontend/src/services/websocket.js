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
    }

    connect() {
        if (this.socket?.connected) {
            console.log('WebSocket already connected');
            this._connected = true;
            return;
        }

        // Get WebSocket URL from environment with fallback
        const wsUrl = process.env.REACT_APP_WS_URL || 'http://localhost:8000';
        console.log('Connecting to WebSocket:', wsUrl);
        
        // Enhanced connection options
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
        
        // Add video stream handler
        this.socket.on('video_frame', (frameData) => {
            if (this.listeners.has('video_frame')) {
                this.listeners.get('video_frame').forEach(callback => {
                    const blob = new Blob([frameData], { type: 'image/jpeg' });
                    callback(URL.createObjectURL(blob));
                });
            }
        });

        this.socket.on('connect', () => {
            console.log('WebSocket Connected');
            this._connected = true;
            this.reconnectAttempts = 0;
            this.startPingInterval();
            
            // Send initial connection data
            this.socket.emit('client_info', {
                timestamp: Date.now(),
                userAgent: navigator.userAgent,
                screenResolution: `${window.screen.width}x${window.screen.height}`,
                connectionType: navigator.connection?.type || 'unknown'
            });

            // Notify all listeners
            if (this.listeners.has('connect')) {
                this.listeners.get('connect').forEach(callback => callback());
            }
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this._connected = false;
            this.clearPingInterval();
            this.handleReconnect();
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            this._connected = false;
            this.clearPingInterval();
            
            if (reason === 'io server disconnect') {
                setTimeout(() => {
                    console.log('Attempting reconnection...');
                    this.socket.connect();
                }, 1000);
            }
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            if (!this.socket.connected) {
                this.handleReconnect();
            }
        });

        this.socket.on('pong', () => {
            this.lastPongTime = Date.now();
        });

        // Handle connection established confirmation
        this.socket.on('connection_established', (data) => {
            console.log('Connection established:', data);
            this._connected = true;
            this.startPingInterval();
        });
    }

    startConnectionCheck() {
        this.clearConnectionCheck();
        this.connectionCheckInterval = setInterval(() => {
            const isReallyConnected = this.socket?.connected && this._connected;
            if (this._connected !== isReallyConnected) {
                this._connected = isReallyConnected;
                if (this.listeners.has('connection_change')) {
                    this.listeners.get('connection_change').forEach(callback => 
                        callback(isReallyConnected)
                    );
                }
            }
        }, 1000);
    }

    clearConnectionCheck() {
        if (this.connectionCheckInterval) {
            clearInterval(this.connectionCheckInterval);
            this.connectionCheckInterval = null;
        }
    }

    startPingInterval() {
        this.clearPingInterval();
        this.lastPongTime = Date.now();

        this.pingInterval = setInterval(() => {
            if (this.socket?.connected) {
                this.socket.emit('ping');
                
                const timeSinceLastPong = Date.now() - this.lastPongTime;
                if (timeSinceLastPong > 120000) { // Increase to 2 minutes
                    console.warn('No pong received in 120s, reconnecting...');
                    this._connected = false;
                    this.clearPingInterval();
                    this.socket.disconnect();
                    setTimeout(() => this.socket.connect(), 1000);
                }
            }
        }, 15000);
    }

    clearPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
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
            this.clearPingInterval();
            this.clearConnectionCheck();
        }
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
        if (this.socket) {
            this.socket.on(event, callback);
        }
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).delete(callback);
        }
        if (this.socket) {
            this.socket.off(event, callback);
        }
    }

    emit(event, data) {
        if (this.socket?.connected) {
            this.socket.emit(event, data);
            return true;
        } else {
            console.warn('Socket not connected, cannot emit:', event);
            return false;
        }
    }

    disconnect() {
        this.clearPingInterval();
        this.clearConnectionCheck();
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
            lastPong: this.lastPongTime
        };
    }
}

export const websocketService = new WebSocketService();
