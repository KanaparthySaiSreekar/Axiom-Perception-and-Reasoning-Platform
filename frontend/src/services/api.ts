/**
 * API Client for Axiom Platform
 */
import axios, { AxiosInstance } from 'axios';
import type {
  AuthToken,
  User,
  CameraInfo,
  PerceptionOutput,
  RobotTelemetry,
  RobotAction,
  StructuredCommand,
  ModelInfo,
  ModelPerformance,
  LayerMetrics,
  HealthStatus,
} from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          this.clearToken();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );

    // Load token from localStorage
    if (typeof window !== 'undefined') {
      const savedToken = localStorage.getItem('auth_token');
      if (savedToken) {
        this.token = savedToken;
      }
    }
  }

  setToken(token: string) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token);
    }
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }

  // Authentication
  async login(username: string, password: string): Promise<AuthToken> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await this.client.post<AuthToken>('/api/v1/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    this.setToken(response.data.access_token);
    return response.data;
  }

  async logout(): Promise<void> {
    await this.client.post('/api/v1/auth/logout');
    this.clearToken();
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/api/v1/auth/me');
    return response.data;
  }

  // Health & Status
  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/api/v1/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/api/v1/health/detailed');
    return response.data;
  }

  // Cameras
  async getCameras(): Promise<CameraInfo[]> {
    const response = await this.client.get<CameraInfo[]>('/api/v1/cameras');
    return response.data;
  }

  async getCamera(cameraId: number): Promise<CameraInfo> {
    const response = await this.client.get<CameraInfo>(`/api/v1/cameras/${cameraId}`);
    return response.data;
  }

  // Perception
  async getPerceptionStatus() {
    const response = await this.client.get('/api/v1/perception/status');
    return response.data;
  }

  async startPerception() {
    const response = await this.client.post('/api/v1/perception/start');
    return response.data;
  }

  async stopPerception() {
    const response = await this.client.post('/api/v1/perception/stop');
    return response.data;
  }

  // Robot Control
  async executeAction(action: RobotAction) {
    const response = await this.client.post('/api/v1/robot/action', action);
    return response.data;
  }

  async emergencyStop() {
    const response = await this.client.post('/api/v1/robot/emergency_stop');
    return response.data;
  }

  async resumeRobot() {
    const response = await this.client.post('/api/v1/robot/resume');
    return response.data;
  }

  async getRobotTelemetry(): Promise<RobotTelemetry> {
    const response = await this.client.get<RobotTelemetry>('/api/v1/robot/telemetry');
    return response.data;
  }

  async processNaturalLanguageCommand(command: string): Promise<StructuredCommand> {
    const response = await this.client.post<StructuredCommand>(
      '/api/v1/robot/llm/command',
      { command, reasoning: '' }
    );
    return response.data;
  }

  // Models
  async getModels(): Promise<ModelInfo[]> {
    const response = await this.client.get<ModelInfo[]>('/api/v1/models');
    return response.data;
  }

  async getModelPerformance(): Promise<ModelPerformance[]> {
    const response = await this.client.get<ModelPerformance[]>('/api/v1/models/performance');
    return response.data;
  }

  async getLayerMetrics(): Promise<LayerMetrics[]> {
    const response = await this.client.get<LayerMetrics[]>('/api/v1/models/layers');
    return response.data;
  }
}

// Singleton instance
export const apiClient = new ApiClient();
