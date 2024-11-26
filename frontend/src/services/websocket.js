import { io } from 'socket.io-client';

class WebSocketService {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.listeners = new Map();
    }

    connect(url) {
        if (this.socket?.connected) {
            console.log('WebSocket already connected');
            return;
        }

        // Construct WebSocket URL with fallbacks
        const wsUrl = this._constructWebSocketUrl(url);
        console.log('Connecting to WebSocket:', wsUrl);
        
        // Enhanced connection options
        this.socket = io(wsUrl, {
            transports: ['websocket'],  // WebSocket only for video streaming
            reconnection: true,
            reconnectionAttempts: this.maxReconnectAttempts,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            timeout: 10000, // Reduced timeout for faster failure detection
            binary: true,   // Enable binary frame support
            autoConnect: true,
            forceNew: true,
            path: '/socket.io/',
            rejectUnauthorized: false,
            pingTimeout: 30000,
            pingInterval: 10000,
            upgrade: true,
            rememberUpgrade: true,
            perMessageDeflate: true,
            extraHeaders: {
                'User-Agent': navigator.userAgent
            },
            query: {
                clientInfo: JSON.stringify({
                    userAgent: navigator.userAgent,
                    timestamp: Date.now(),
                    screenResolution: `${window.screen.width}x${window.screen.height}`
                })
            }
        });

        this.setupEventHandlers();
    }

    _constructWebSocketUrl(url) {
        // If full URL provided, use it
        if (url) {
            return url;
        }

        // Get WebSocket URL with fallbacks
        const wsPort = process.env.REACT_APP_WS_PORT || '8001';
        const wsHost = process.env.REACT_APP_WS_HOST || window.location.hostname;
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        // Remove any trailing slashes from host
        const cleanHost = wsHost.replace(/\/$/, '');
        
        // Construct full URL
        return `${wsProtocol}//${cleanHost}:${wsPort}`;
    }

    setupEventHandlers() {
        this.socket.on('connect', () => {
            console.log('WebSocket Connected');
            this.reconnectAttempts = 0;
            this.socket.emit('get_uploaded_files');
            
            // Send initial connection data
            this.socket.emit('client_info', {
                timestamp: Date.now(),
                userAgent: navigator.userAgent,
                screenResolution: `${window.screen.width}x${window.screen.height}`,
                connectionType: navigator.connection?.type || 'unknown'
            });
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.handleReconnect();
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            if (reason === 'io server disconnect' || reason === 'transport close') {
                setTimeout(() => {
                    console.log('Attempting reconnection...');
                    this.socket.connect();
                }, 1000);
            }
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            // Attempt recovery on error
            if (!this.socket.connected) {
                this.handleReconnect();
            }
        });

        this.socket.on('reconnect_attempt', (attemptNumber) => {
            console.log(`Reconnection attempt ${attemptNumber}`);
        });

        this.socket.on('reconnect_failed', () => {
            console.error('Failed to reconnect after all attempts');
        });

        // Enhanced ping/pong with timeout detection
        let lastPongTime = Date.now();
        
        this.socket.on('pong', () => {
            lastPongTime = Date.now();
        });

        setInterval(() => {
            if (this.socket?.connected) {
                this.socket.emit('ping');
                
                // Check if we haven't received a pong in too long
                if (Date.now() - lastPongTime > 30000) {
                    console.warn('No pong received in 30s, reconnecting...');
                    this.socket.disconnect();
                    this.socket.connect();
                }
            }
        }, 10000);
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
            console.log(`Reconnecting... Attempt ${this.reconnectAttempts} in ${delay}ms`);
            setTimeout(() => this.socket.connect(), delay);
        } else {
            console.error('Max reconnection attempts reached');
        }
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
        this.socket.on(event, callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).delete(callback);
        }
        this.socket.off(event, callback);
    }

    emit(event, data) {
        if (this.socket?.connected) {
            this.socket.emit(event, data);
        } else {
            console.warn('Socket not connected, cannot emit:', event);
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }
}

export const websocketService = new WebSocketService();
