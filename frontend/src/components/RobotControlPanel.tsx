'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/services/api';
import { TelemetryStreamClient } from '@/services/websocket';
import type { RobotTelemetry } from '@/types';

export function RobotControlPanel() {
  const [telemetry, setTelemetry] = useState<RobotTelemetry | null>(null);
  const [isEmergencyStopped, setIsEmergencyStopped] = useState(false);
  const [wsClient] = useState(() => new TelemetryStreamClient());

  useEffect(() => {
    // Connect to telemetry stream
    wsClient.connect().then(() => {
      wsClient.on<RobotTelemetry>('telemetry', setTelemetry);
    });

    return () => {
      wsClient.disconnect();
    };
  }, [wsClient]);

  const handleEmergencyStop = async () => {
    try {
      await apiClient.emergencyStop();
      setIsEmergencyStopped(true);
    } catch (error) {
      console.error('Emergency stop failed:', error);
    }
  };

  const handleResume = async () => {
    try {
      await apiClient.resumeRobot();
      setIsEmergencyStopped(false);
    } catch (error) {
      console.error('Resume failed:', error);
    }
  };

  const handleQuickAction = async (action: string) => {
    try {
      await apiClient.executeAction({
        action_type: action,
        parameters: {},
      });
    } catch (error) {
      console.error(`Action ${action} failed:`, error);
    }
  };

  return (
    <div className="space-y-4">
      {/* Emergency Stop */}
      <div className="p-4 bg-red-950/20 border border-red-600 rounded-lg">
        {isEmergencyStopped ? (
          <div className="space-y-2">
            <div className="text-red-400 font-semibold flex items-center gap-2">
              <div className="status-indicator error" />
              EMERGENCY STOP ACTIVE
            </div>
            <button
              onClick={handleResume}
              className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded font-medium"
            >
              Resume Operation
            </button>
          </div>
        ) : (
          <button
            onClick={handleEmergencyStop}
            className="w-full px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded font-bold"
          >
            🛑 EMERGENCY STOP
          </button>
        )}
      </div>

      {/* Robot Status */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-muted-foreground">Status</h3>
        <div className="p-3 bg-secondary rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm">Robot State:</span>
            <span className="text-sm font-semibold">
              {telemetry?.status || 'Unknown'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">Gripper:</span>
            <span className="text-sm font-semibold">
              {telemetry?.gripper_state || 'Unknown'}
            </span>
          </div>
        </div>
      </div>

      {/* Joint Positions */}
      {telemetry && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground">
            Joint Positions
          </h3>
          <div className="space-y-1">
            {telemetry.joint_positions.map((pos, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <span className="text-xs w-16">Joint {idx + 1}:</span>
                <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary"
                    style={{
                      width: `${Math.abs((pos / Math.PI) * 100)}%`,
                    }}
                  />
                </div>
                <span className="text-xs w-16 text-right">
                  {pos.toFixed(2)} rad
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* End Effector Pose */}
      {telemetry && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground">
            End Effector
          </h3>
          <div className="p-3 bg-secondary rounded-lg text-xs space-y-1">
            <div>
              Position:{' '}
              {telemetry.end_effector_pose.position.map((v) => v.toFixed(3)).join(', ')}
            </div>
            <div>
              Orientation:{' '}
              {telemetry.end_effector_pose.orientation.map((v) => v.toFixed(3)).join(', ')}
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-muted-foreground">
          Quick Actions
        </h3>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => handleQuickAction('home')}
            disabled={isEmergencyStopped}
            className="px-3 py-2 bg-primary hover:bg-primary/90 disabled:bg-gray-700 text-primary-foreground rounded text-sm"
          >
            Home
          </button>
          <button
            onClick={() => handleQuickAction('pick')}
            disabled={isEmergencyStopped}
            className="px-3 py-2 bg-primary hover:bg-primary/90 disabled:bg-gray-700 text-primary-foreground rounded text-sm"
          >
            Pick
          </button>
          <button
            onClick={() => handleQuickAction('place')}
            disabled={isEmergencyStopped}
            className="px-3 py-2 bg-primary hover:bg-primary/90 disabled:bg-gray-700 text-primary-foreground rounded text-sm"
          >
            Place
          </button>
          <button
            onClick={() => handleQuickAction('pour')}
            disabled={isEmergencyStopped}
            className="px-3 py-2 bg-primary hover:bg-primary/90 disabled:bg-gray-700 text-primary-foreground rounded text-sm"
          >
            Pour
          </button>
        </div>
      </div>
    </div>
  );
}
