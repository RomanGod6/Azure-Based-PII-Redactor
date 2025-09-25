// WebSocket service for real-time communication with backend
class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private messageHandlers: { [key: string]: (data: any) => void } = {};
  private currentSessionId: string | null = null;

  private connect() {
    try {
      // Always connect to backend on localhost:8080 regardless of frontend host
      const wsUrl = 'ws://localhost:8080/ws';
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = this.onOpen.bind(this);
      this.ws.onmessage = this.onMessage.bind(this);
      this.ws.onclose = this.onClose.bind(this);
      this.ws.onerror = this.onError.bind(this);
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  private onOpen() {
    console.log('🔌 WebSocket connected');
    this.reconnectAttempts = 0;
    
    // Send ping to maintain connection
    this.startHeartbeat();
  }

  private onMessage(event: MessageEvent) {
    try {
      const message = JSON.parse(event.data);
      console.log('📨 WebSocket message received:', message);
      
      // Handle different message types
      const handler = this.messageHandlers[message.type];
      if (handler) {
        handler(message);
      } else {
        console.warn('No handler for message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private onClose(event: CloseEvent) {
    console.log('🔌 WebSocket disconnected:', event.code, event.reason);
    this.ws = null;
    
    if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  private onError(error: Event) {
    console.error('🔌 WebSocket error:', error);
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
      console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.reconnectAttempts++;
        this.connect();
      }, delay);
    } else {
      console.error('❌ Max reconnection attempts reached');
    }
  }

  private startHeartbeat() {
    const heartbeat = () => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({
          type: 'ping',
          timestamp: new Date().toISOString()
        });
        setTimeout(heartbeat, 30000); // Ping every 30 seconds
      }
    };
    heartbeat();
  }

  public send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message:', message);
    }
  }

  public subscribe(messageType: string, handler: (data: any) => void) {
    this.messageHandlers[messageType] = handler;
  }

  public unsubscribe(messageType: string) {
    delete this.messageHandlers[messageType];
  }

  public subscribeToSession(sessionId: string) {
    this.currentSessionId = sessionId;
    this.send({
      type: 'subscribe',
      data: sessionId,
      timestamp: new Date().toISOString()
    });
  }

  public getActiveSessions() {
    this.send({
      type: 'get_active_sessions',
      timestamp: new Date().toISOString()
    });
  }

  public getSessionDetails(sessionId: string) {
    this.send({
      type: 'get_session_details',
      data: sessionId,
      timestamp: new Date().toISOString()
    });
  }

  public cancelProcessing(sessionId: string) {
    this.send({
      type: 'cancel',
      session_id: sessionId,
      timestamp: new Date().toISOString()
    });
  }

  public disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  public getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }
}

// File processing specific WebSocket service
class FileProcessingWebSocketService {
  private ws: WebSocket | null = null;
  private onProgressUpdate: ((progress: any) => void) | null = null;
  private onComplete: ((result: any) => void) | null = null;
  private onError: ((error: any) => void) | null = null;
  private onRateLimit: ((info: any) => void) | null = null;
  private sessionId: string | null = null;

  public async processFile(
    fileName: string, 
    fileContent: string, 
    fileType: string, 
    redactOptions: any
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      // Set a timeout for the connection
      const connectionTimeout = setTimeout(() => {
        if (this.ws) {
          this.ws.close();
        }
        reject(new Error('WebSocket connection timeout'));
      }, 10000); // 10 second timeout

      try {
        // Always connect to backend on localhost:8080 regardless of frontend host
        const wsUrl = 'ws://localhost:8080/ws/process';
        
        console.log('🔌 Attempting WebSocket connection to:', wsUrl);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log('🔌 File processing WebSocket connected');
          
          // Send file processing request
          const request = {
            filename: fileName,
            file_content: fileContent,
            file_type: fileType,
            redact_options: redactOptions
          };
          
          console.log('📤 Sending file processing request:', { 
            filename: fileName, 
            file_type: fileType,
            content_length: fileContent.length 
          });
          this.ws!.send(JSON.stringify(request));
        };
        
        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            
            switch (message.type) {
              case 'session_created':
                clearTimeout(connectionTimeout);
                this.sessionId = message.session_id;
                console.log('📝 Processing session created:', this.sessionId);
                if (this.sessionId) {
                  resolve(this.sessionId);
                } else {
                  reject(new Error('Session ID not received'));
                }
                break;
                
              case 'progress_update':
                if (this.onProgressUpdate) {
                  this.onProgressUpdate(message.data);
                }
                break;
                
              case 'session_completed':
                if (this.onComplete) {
                  this.onComplete(message.data);
                }
                this.disconnect();
                break;
                
              case 'error':
                clearTimeout(connectionTimeout);
                if (this.onError) {
                  this.onError(message.data);
                }
                this.disconnect();
                reject(new Error(message.data.message || 'Processing failed'));
                break;
                
              case 'rate_limit':
                if (this.onRateLimit) {
                  this.onRateLimit(message.data);
                }
                break;
                
              default:
                console.log('Unknown message type:', message.type);
            }
          } catch (error) {
            console.error('Failed to parse file processing message:', error);
          }
        };
        
        this.ws.onclose = () => {
          clearTimeout(connectionTimeout);
          console.log('🔌 File processing WebSocket disconnected');
        };
        
        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error('🔌 File processing WebSocket error:', error);
          reject(new Error('WebSocket connection failed'));
        };
        
      } catch (error) {
        clearTimeout(connectionTimeout);
        reject(error);
      }
    });
  }

  public setProgressHandler(handler: (progress: any) => void) {
    this.onProgressUpdate = handler;
  }

  public setCompleteHandler(handler: (result: any) => void) {
    this.onComplete = handler;
  }

  public setErrorHandler(handler: (error: any) => void) {
    this.onError = handler;
  }

  public setRateLimitHandler(handler: (info: any) => void) {
    this.onRateLimit = handler;
  }

  public cancel() {
    if (this.ws && this.sessionId) {
      this.ws.send(JSON.stringify({
        type: 'cancel',
        session_id: this.sessionId,
        timestamp: new Date().toISOString()
      }));
    }
    this.disconnect();
  }

  public disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.sessionId = null;
  }

  public getSessionId(): string | null {
    return this.sessionId;
  }
}

// Export singleton instances
export const webSocketService = new WebSocketService();
export const fileProcessingWebSocket = new FileProcessingWebSocketService();

export default webSocketService;