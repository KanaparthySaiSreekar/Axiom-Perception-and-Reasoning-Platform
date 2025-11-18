'use client';

import { useEffect, useState } from 'react';
import { PerceptionStreamClient } from '@/services/websocket';
import type { PerceptionOutput } from '@/types';

export function PerceptionOverlay() {
  const [perception, setPerception] = useState<PerceptionOutput | null>(null);
  const [wsClient] = useState(() => new PerceptionStreamClient());

  useEffect(() => {
    wsClient.connect().then(() => {
      wsClient.on<PerceptionOutput>('perception', setPerception);
    });

    return () => {
      wsClient.disconnect();
    };
  }, [wsClient]);

  if (!perception) {
    return (
      <div className="h-64 flex items-center justify-center bg-muted rounded-lg">
        <div className="text-center text-muted-foreground">
          <div className="text-4xl mb-2">👁️</div>
          <div>Waiting for perception data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Detection Overview */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 bg-secondary rounded-lg">
          <div className="text-2xl font-bold">{perception.detections.length}</div>
          <div className="text-sm text-muted-foreground">Objects Detected</div>
        </div>

        <div className="p-4 bg-secondary rounded-lg">
          <div className="text-2xl font-bold">
            {perception.liquid_level.detected ? (
              <span className="text-green-400">
                {perception.liquid_level.level_mm.toFixed(0)}mm
              </span>
            ) : (
              <span className="text-gray-500">—</span>
            )}
          </div>
          <div className="text-sm text-muted-foreground">Liquid Level</div>
        </div>

        <div className="p-4 bg-secondary rounded-lg">
          <div className="text-2xl font-bold">
            {perception.trajectory_prediction.available ? (
              <span className="text-blue-400">
                {perception.trajectory_prediction.horizon_sec}s
              </span>
            ) : (
              <span className="text-gray-500">—</span>
            )}
          </div>
          <div className="text-sm text-muted-foreground">Prediction Horizon</div>
        </div>
      </div>

      {/* Detections List */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-muted-foreground">
          Detected Objects
        </h3>

        {perception.detections.length > 0 ? (
          <div className="space-y-2">
            {perception.detections.map((det, idx) => (
              <div
                key={idx}
                className="p-3 bg-secondary rounded-lg flex items-center justify-between"
              >
                <div>
                  <div className="font-semibold text-sm">{det.class}</div>
                  <div className="text-xs text-muted-foreground">
                    Position: ({det.center[0].toFixed(0)}, {det.center[1].toFixed(0)})
                    {det.track_id !== undefined && ` • Track ID: ${det.track_id}`}
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-sm font-mono">
                    {(det.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">confidence</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-muted-foreground p-4 bg-secondary rounded-lg text-center">
            No objects detected
          </div>
        )}
      </div>

      {/* Segmentation Info */}
      {perception.segmentation.available && (
        <div className="p-3 bg-secondary rounded-lg">
          <div className="text-sm font-semibold mb-2">Segmentation</div>
          <div className="flex gap-2 flex-wrap">
            {perception.segmentation.classes.map((cls) => (
              <span
                key={cls}
                className="px-2 py-1 bg-primary/20 text-primary rounded text-xs"
              >
                {cls}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Frame Info */}
      <div className="text-xs text-muted-foreground">
        Frame #{perception.frame_id} • {perception.timestamp}
      </div>
    </div>
  );
}
