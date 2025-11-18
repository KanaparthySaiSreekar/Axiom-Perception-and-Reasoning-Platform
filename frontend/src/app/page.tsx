'use client';

import { useEffect, useState } from 'react';
import { CameraGrid } from '@/components/CameraGrid';
import { NaturalLanguageConsole } from '@/components/NaturalLanguageConsole';
import { RobotControlPanel } from '@/components/RobotControlPanel';
import { SystemDiagnostics } from '@/components/SystemDiagnostics';
import { PerceptionOverlay } from '@/components/PerceptionOverlay';
import { apiClient } from '@/services/api';
import type { HealthStatus } from '@/types';

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check health on mount
    apiClient
      .getHealth()
      .then(setHealth)
      .catch((err) => {
        console.error('Failed to fetch health:', err);
        setError('Failed to connect to backend');
      });

    // Poll health every 30 seconds
    const interval = setInterval(() => {
      apiClient.getHealth().then(setHealth).catch(console.error);
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-destructive mb-4">
            Connection Error
          </h1>
          <p className="text-muted-foreground">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold">Axiom Platform</h1>
            <div className="flex items-center gap-2">
              <div
                className={`status-indicator ${
                  health?.status === 'healthy' ? 'active' : 'error'
                }`}
              />
              <span className="text-sm text-muted-foreground">
                {health?.status || 'connecting...'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-sm text-muted-foreground">
              Uptime: {health ? Math.floor(health.uptime_seconds / 60) : 0}m
            </div>
            <div className="text-sm text-muted-foreground">
              {health?.version || '1.0.0'}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Camera Grid & Perception */}
          <div className="col-span-8 space-y-6">
            {/* Multi-Camera Grid */}
            <div className="bg-card rounded-lg border border-border p-4">
              <h2 className="text-lg font-semibold mb-4">Multi-Camera View</h2>
              <CameraGrid />
            </div>

            {/* Perception Overlay Visualization */}
            <div className="bg-card rounded-lg border border-border p-4">
              <h2 className="text-lg font-semibold mb-4">
                Perception Analysis
              </h2>
              <PerceptionOverlay />
            </div>

            {/* Natural Language Console */}
            <div className="bg-card rounded-lg border border-border p-4">
              <h2 className="text-lg font-semibold mb-4">
                AI Command Console
              </h2>
              <NaturalLanguageConsole />
            </div>
          </div>

          {/* Right Column - Control & Diagnostics */}
          <div className="col-span-4 space-y-6">
            {/* Robot Control Panel */}
            <div className="bg-card rounded-lg border border-border p-4">
              <h2 className="text-lg font-semibold mb-4">Robot Control</h2>
              <RobotControlPanel />
            </div>

            {/* System Diagnostics */}
            <div className="bg-card rounded-lg border border-border p-4">
              <h2 className="text-lg font-semibold mb-4">
                System Diagnostics
              </h2>
              <SystemDiagnostics />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
