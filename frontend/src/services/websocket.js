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

        console.log('Connecting to WebSocket:', url);
        
        this.socket = io(url, {
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: this.maxReconnectAttempts,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            timeout: 10000,
            autoConnect: true,
            query: {
                clientInfo: JSON.stringify({
                    userAgent: navigator.userAgent,
                    timestamp: Date.now()
                })
            }
        });

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.socket.on('connect', () => {
            console.log('WebSocket Connected');
            this.reconnectAttempts = 0;
            this.socket.emit('get_uploaded_files');
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.handleReconnect();
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            if (reason === 'io server disconnect') {
                this.socket.connect();
            }
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
        });

        // Setup ping/pong
        setInterval(() => {
            if (this.socket?.connected) {
                this.socket.emit('ping');
            }
        }, 5000);
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
