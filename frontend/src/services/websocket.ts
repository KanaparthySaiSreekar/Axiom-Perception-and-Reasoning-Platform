/**
 * WebSocket Client for Real-time Data Streaming
 */

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export type WebSocketCallback<T> = (data: T) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 5000;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private callbacks: Map<string, WebSocketCallback<any>> = new Map();
  private isConnecting: boolean = false;

  constructor(
    private endpoint: string,
    private autoReconnect: boolean = true
  ) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        resolve();
        return;
      }

      this.isConnecting = true;

      try {
        this.ws = new WebSocket(`${WS_URL}${this.endpoint}`);

        this.ws.onopen = () => {
          console.log(`WebSocket connected: ${this.endpoint}`);
          this.isConnecting = false;
          this.clearReconnectTimer();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error(`WebSocket error on ${this.endpoint}:`, error);
          this.isConnecting = false;
          reject(error);
        };

        this.ws.onclose = () => {
          console.log(`WebSocket closed: ${this.endpoint}`);
          this.isConnecting = false;
          if (this.autoReconnect) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.autoReconnect = false;
    this.clearReconnectTimer();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  on<T>(event: string, callback: WebSocketCallback<T>): void {
    this.callbacks.set(event, callback);
  }

  off(event: string): void {
    this.callbacks.delete(event);
  }

  send(data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected, cannot send data');
    }
  }

  private handleMessage(data: any): void {
    // Call all registered callbacks
    this.callbacks.forEach((callback) => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in WebSocket callback:', error);
      }
    });
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      return;
    }

    console.log(`Scheduling reconnect for ${this.endpoint} in ${this.reconnectInterval}ms`);

    this.reconnectTimer = setTimeout(() => {
      console.log(`Attempting to reconnect ${this.endpoint}...`);
      this.reconnectTimer = null;
      this.connect().catch((error) => {
        console.error('Reconnection failed:', error);
      });
    }, this.reconnectInterval);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// WebSocket managers for different data streams
export class CameraStreamClient extends WebSocketClient {
  constructor(cameraId: number) {
    super(`/ws/video/${cameraId}`);
  }
}

export class PerceptionStreamClient extends WebSocketClient {
  constructor() {
    super('/ws/perception');
  }
}

export class TelemetryStreamClient extends WebSocketClient {
  constructor() {
    super('/ws/telemetry');
  }
}
