'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/services/api';
import type { ModelPerformance, LayerMetrics } from '@/types';

export function SystemDiagnostics() {
  const [modelPerf, setModelPerf] = useState<ModelPerformance[]>([]);
  const [layerMetrics, setLayerMetrics] = useState<LayerMetrics[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const [perf, layers] = await Promise.all([
          apiClient.getModelPerformance(),
          apiClient.getLayerMetrics(),
        ]);

        setModelPerf(perf);
        setLayerMetrics(layers);
      } catch (error) {
        console.error('Failed to load metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    loadMetrics();

    // Refresh every 10 seconds
    const interval = setInterval(loadMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="text-sm text-muted-foreground">Loading diagnostics...</div>;
  }

  const totalLatency = layerMetrics.reduce((sum, layer) => sum + layer.latency_p50_ms, 0);
  const latencyBudget = 90; // ms
  const latencyUtilization = (totalLatency / latencyBudget) * 100;

  return (
    <div className="space-y-4 max-h-96 overflow-y-auto">
      {/* Pipeline Latency */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-muted-foreground">
            Pipeline Latency
          </h3>
          <span className="text-xs text-muted-foreground">
            {totalLatency.toFixed(1)}ms / {latencyBudget}ms
          </span>
        </div>

        <div className="h-4 bg-secondary rounded-full overflow-hidden">
          <div
            className={`h-full ${
              latencyUtilization > 100
                ? 'bg-red-500'
                : latencyUtilization > 80
                ? 'bg-yellow-500'
                : 'bg-green-500'
            }`}
            style={{ width: `${Math.min(latencyUtilization, 100)}%` }}
          />
        </div>

        <div className="text-xs text-muted-foreground">
          {latencyUtilization.toFixed(1)}% utilization
        </div>
      </div>

      {/* Layer Breakdown */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-muted-foreground">
          Layer Performance
        </h3>

        <div className="space-y-1">
          {layerMetrics.map((layer) => (
            <div key={layer.layer_name} className="text-xs">
              <div className="flex items-center justify-between mb-1">
                <span className="truncate">{layer.layer_name}</span>
                <span className="font-mono">
                  {layer.latency_p50_ms.toFixed(1)}ms
                </span>
              </div>
              <div className="flex gap-2 text-[10px] text-muted-foreground">
                <span>p95: {layer.latency_p95_ms.toFixed(1)}ms</span>
                <span>p99: {layer.latency_p99_ms.toFixed(1)}ms</span>
                <span>acc: {(layer.accuracy * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Performance */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-muted-foreground">
          Model Performance
        </h3>

        {modelPerf.map((model) => (
          <div key={model.model_name} className="p-2 bg-secondary rounded text-xs">
            <div className="font-semibold mb-1">{model.model_name}</div>
            <div className="grid grid-cols-2 gap-1 text-[10px]">
              <span>Accuracy: {(model.accuracy * 100).toFixed(1)}%</span>
              <span>FPS: {model.throughput_fps.toFixed(1)}</span>
              <span>Latency: {model.latency_ms.toFixed(1)}ms</span>
              <span>GPU: {model.gpu_memory_mb.toFixed(0)}MB</span>
            </div>
            {model.accuracy_drift > 0.05 && (
              <div className="text-yellow-500 mt-1">
                ⚠️ Accuracy drift detected: {(model.accuracy_drift * 100).toFixed(1)}%
              </div>
            )}
          </div>
        ))}
      </div>

      {/* System Resources */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-muted-foreground">
          System Resources
        </h3>

        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span>GPU Memory:</span>
            <span className="font-mono">
              {modelPerf.reduce((sum, m) => sum + m.gpu_memory_mb, 0).toFixed(0)} MB
            </span>
          </div>
          <div className="flex justify-between">
            <span>Pipeline FPS:</span>
            <span className="font-mono">
              {modelPerf[0]?.throughput_fps.toFixed(1) || '0.0'} FPS
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
