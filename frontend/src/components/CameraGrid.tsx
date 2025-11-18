'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/services/api';
import { CameraStreamClient } from '@/services/websocket';
import type { CameraInfo } from '@/types';

export function CameraGrid() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load camera list
    apiClient
      .getCameras()
      .then((data) => {
        setCameras(data);
        if (data.length > 0) {
          setSelectedCamera(data[0].id);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-4 h-96">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="bg-muted animate-pulse rounded-lg flex items-center justify-center"
          >
            <span className="text-muted-foreground">Loading camera {i}...</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Camera Grid */}
      <div className="grid grid-cols-2 gap-4 h-96">
        {cameras.map((camera) => (
          <div
            key={camera.id}
            className={`relative rounded-lg overflow-hidden border-2 cursor-pointer transition-all ${
              selectedCamera === camera.id
                ? 'border-primary'
                : 'border-border hover:border-primary/50'
            }`}
            onClick={() => setSelectedCamera(camera.id)}
          >
            <CameraView camera={camera} />

            {/* Camera Info Overlay */}
            <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm rounded px-2 py-1 text-xs">
              <div className="text-white font-semibold">{camera.name}</div>
              <div className="text-gray-300">
                {camera.resolution[0]}x{camera.resolution[1]} @ {camera.fps}fps
              </div>
              <div className="flex items-center gap-1">
                <div
                  className={`status-indicator ${
                    camera.status === 'streaming' ? 'active' : 'inactive'
                  }`}
                />
                <span className="text-gray-300">{camera.status}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Camera Controls */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          <button className="px-3 py-1 bg-secondary text-secondary-foreground rounded text-sm">
            Calibrate Cameras
          </button>
          <button className="px-3 py-1 bg-secondary text-secondary-foreground rounded text-sm">
            Sync Adjustment
          </button>
        </div>

        <div className="text-sm text-muted-foreground">
          {cameras.filter((c) => c.status === 'streaming').length} / {cameras.length}{' '}
          cameras active
        </div>
      </div>
    </div>
  );
}

function CameraView({ camera }: { camera: CameraInfo }) {
  const [connected, setConnected] = useState(false);
  const [wsClient] = useState(() => new CameraStreamClient(camera.id));

  useEffect(() => {
    // Connect to camera stream
    wsClient
      .connect()
      .then(() => setConnected(true))
      .catch((err) => console.error(`Failed to connect to camera ${camera.id}:`, err));

    wsClient.on('data', (data) => {
      // Handle camera frame data
      console.log(`Camera ${camera.id} frame:`, data);
    });

    return () => {
      wsClient.disconnect();
    };
  }, [camera.id, wsClient]);

  return (
    <div className="video-stream h-full bg-black flex items-center justify-center">
      {connected ? (
        <div className="text-center text-white">
          <div className="text-6xl mb-2">📹</div>
          <div className="text-sm">Camera {camera.id} Streaming</div>
          <div className="text-xs text-gray-400 mt-1">
            {camera.resolution[0]}x{camera.resolution[1]} @ {camera.fps}fps
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-500">
          <div className="text-4xl mb-2">⚠️</div>
          <div className="text-sm">Connecting...</div>
        </div>
      )}
    </div>
  );
}
