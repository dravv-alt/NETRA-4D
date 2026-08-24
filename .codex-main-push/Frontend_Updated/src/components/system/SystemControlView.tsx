import React, { useState } from 'react';
import { CameraFeedInfo, SystemHealthData } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { Camera, Cpu, Sliders, ShieldCheck, MapPin, Server, Activity } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Tabs } from '../ui/tabs';
import { Button } from '../ui/button';
import { ParallaxCard } from '../ui/parallax-card';

interface SystemControlViewProps {
  cameras: CameraFeedInfo[];
  health: SystemHealthData;
}

export const SystemControlView: React.FC<SystemControlViewProps> = ({
  cameras,
  health,
}) => {
  const [activeTab, setActiveTab] = useState<'cameras' | 'context' | 'providers'>('cameras');

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      {/* Top Navigation Tabs */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200 flex-wrap gap-4">
        <div>
          <div className="font-bold text-lg text-neutral-950 flex items-center gap-3">
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
              <Cpu size={18} />
            </div>
            <span>Inference Infrastructure & Camera Nodes</span>
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Manage multi-camera RTSP ingestion, spatial context rules, and GPU tensor core health.
          </div>
        </div>

        <Tabs
          items={[
            { id: 'cameras', label: `Camera Nodes (${cameras.length})`, icon: <Camera size={14} /> },
            { id: 'context', label: 'Spatial Context & Rules', icon: <MapPin size={14} /> },
            { id: 'providers', label: 'AI Providers & GPU Telemetry', icon: <Server size={14} /> },
          ]}
          activeId={activeTab}
          onChange={(id) => setActiveTab(id as any)}
          size="sm"
        />
      </div>

      {/* TAB 1: CAMERAS */}
      {activeTab === 'cameras' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {cameras.map((cam) => (
            <ParallaxCard
              key={cam.id}
              tiltAngle={2}
              className="p-7 space-y-5 bg-white border border-neutral-200 shadow-xs rounded-3xl"
            >
              <div className="flex items-center justify-between pb-3 border-b border-neutral-200">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-2xl bg-neutral-900 flex items-center justify-center text-white font-bold shadow-xs">
                    <Camera size={16} />
                  </div>
                  <div>
                    <span className="font-bold text-base text-neutral-950">{cam.name}</span>
                    <div className="text-xs text-neutral-500">{cam.location}</div>
                  </div>
                </div>
                <StatusBadge status="HEALTHY" label={cam.status} size="sm" />
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs bg-neutral-50 p-4 rounded-2xl border border-neutral-200">
                <div>
                  <span className="text-neutral-500">Resolution: </span>
                  <strong className="text-neutral-950 font-bold">{cam.resolution}</strong>
                </div>
                <div>
                  <span className="text-neutral-500">Frame Rate: </span>
                  <strong className="text-emerald-700 tabular-nums font-bold">{cam.fps} FPS</strong>
                </div>
                <div>
                  <span className="text-neutral-500">Assigned Desks: </span>
                  <strong className="text-neutral-950 font-bold">{cam.assignedSeats.length} Terminals</strong>
                </div>
                <div>
                  <span className="text-neutral-500">Node Status: </span>
                  <strong className="text-emerald-700 font-bold">RTSP Active</strong>
                </div>
              </div>

              {/* Quality Indicators */}
              <div className="grid grid-cols-3 gap-3 text-xs pt-1">
                <div className="p-3.5 rounded-2xl bg-neutral-50 border border-neutral-200">
                  <div className="text-neutral-500 text-2xs uppercase font-semibold">Tracking</div>
                  <div className="font-bold text-emerald-700 mt-1">{cam.trackingQuality}</div>
                </div>
                <div className="p-3.5 rounded-2xl bg-neutral-50 border border-neutral-200">
                  <div className="text-neutral-500 text-2xs uppercase font-semibold">Pose Lattice</div>
                  <div className="font-bold text-emerald-700 mt-1">{cam.poseQuality}</div>
                </div>
                <div className="p-3.5 rounded-2xl bg-neutral-50 border border-neutral-200">
                  <div className="text-neutral-500 text-2xs uppercase font-semibold">Objects</div>
                  <div className="font-bold text-emerald-700 mt-1">{cam.objectVisibility}</div>
                </div>
              </div>
            </ParallaxCard>
          ))}
        </div>
      )}

      {/* TAB 2: SPATIAL CONTEXT EDITOR */}
      {activeTab === 'context' && (
        <ParallaxCard tiltAngle={2} className="p-8 space-y-6 bg-white border border-neutral-200 shadow-xs rounded-3xl">
          <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
            <div>
              <div className="font-bold text-base text-neutral-950">
                Exam Hall Spatial Context Rules & Anti-False Positive Zones
              </div>
              <div className="text-xs text-neutral-500 mt-1">
                Polygons define wall countdown timers and proctor walk corridors to eliminate false positives.
              </div>
            </div>
            <Button
              variant="default"
              size="sm"
              onClick={() => playHapticSound('snap')}
              className="rounded-xl font-semibold shadow-xs"
            >
              + Add Context Polygon
            </Button>
          </div>

          <div className="relative aspect-21/9 bg-neutral-50 rounded-2xl border border-neutral-200 flex items-center justify-center overflow-hidden p-6 shadow-2xs">
            <div className="w-full h-full border border-dashed border-neutral-300 rounded-2xl flex flex-col justify-between p-4 relative bg-white/70">
              <div className="w-2/3 mx-auto py-2.5 bg-sky-50 border border-sky-200 rounded-xl text-center text-xs text-sky-900 font-bold tracking-wider shadow-2xs">
                [ZONE 01] NORTH WALL TIMER & BOARD (GAZE SUPPRESSION CONE)
              </div>

              <div className="grid grid-cols-2 gap-4 my-auto px-4">
                <div className="p-4 bg-neutral-50 rounded-xl text-center text-xs border border-neutral-200 shadow-2xs">
                  <div className="text-neutral-500 uppercase font-bold">ROW A (NORTH DESK)</div>
                  <div className="text-neutral-950 font-bold mt-1">TERMINALS A01 - A10 (STUDENTS FACING NORTH)</div>
                </div>
                <div className="p-4 bg-neutral-50 rounded-xl text-center text-xs border border-neutral-200 shadow-2xs">
                  <div className="text-neutral-500 uppercase font-bold">ROW B (SOUTH DESK)</div>
                  <div className="text-neutral-950 font-bold mt-1">TERMINALS B01 - B10 (STUDENTS FACING SOUTH)</div>
                </div>
              </div>

              <div className="w-1/2 mx-auto py-2 bg-stone-50 border border-neutral-300 rounded-xl text-center text-xs text-neutral-800 font-bold tracking-wider shadow-2xs">
                [ZONE 02] CENTRAL PROCTOR WALKWAY (INTERACTION CORRIDOR)
              </div>
            </div>
          </div>

          <div className="text-xs text-neutral-500 flex items-center justify-between pt-3 border-t border-neutral-200">
            <span>Active Rules: <strong className="text-neutral-900 font-semibold">4 Rules Enabled</strong> (Wall Timer Glance, Proctor Walkway Baseline, Wired Mouse Discrimination, Screen Plane Focus)</span>
            <span className="text-emerald-700 font-bold flex items-center gap-1.5">
              <ShieldCheck size={15} />
              Spatial Graph: SYNCHRONIZED
            </span>
          </div>
        </ParallaxCard>
      )}

      {/* TAB 3: PROVIDERS & GPU */}
      {activeTab === 'providers' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ParallaxCard tiltAngle={2} className="p-8 space-y-6 bg-white border border-neutral-200 shadow-xs rounded-3xl">
            <div className="font-bold text-base text-neutral-950 pb-3 border-b border-neutral-200 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Server size={16} className="text-neutral-900" />
                <span>NVIDIA T4 Tensor Core GPU Telemetry</span>
              </div>
              <StatusBadge status="HEALTHY" label="OPTIMAL" size="sm" />
            </div>
            <div className="space-y-4 text-xs">
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">GPU Utilization:</span>
                <strong className="text-neutral-950 tabular-nums font-bold text-sm">{health.t4GpuUtilPct}% (Nominal Load)</strong>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">VRAM Ingestion:</span>
                <strong className="text-neutral-950 tabular-nums font-bold text-sm">{health.vramUsedGb} GB / {health.vramTotalGb} GB (37%)</strong>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">Operating Temperature:</span>
                <strong className="text-neutral-950 tabular-nums font-bold text-sm">{health.t4GpuTempC}°C</strong>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">Pipeline DAG Latency:</span>
                <strong className="text-emerald-700 tabular-nums font-bold text-sm">{health.pipelineLatencyMs}ms</strong>
              </div>
            </div>
          </ParallaxCard>

          <ParallaxCard tiltAngle={2} className="p-8 space-y-6 bg-white border border-neutral-200 shadow-xs rounded-3xl">
            <div className="font-bold text-base text-neutral-950 pb-3 border-b border-neutral-200 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Sliders size={16} className="text-neutral-900" />
                <span>Multi-Modal Evidence Weighting DAG</span>
              </div>
              <span className="text-2xs font-semibold px-2.5 py-0.5 rounded-full bg-neutral-100 border border-neutral-200 text-neutral-700">Weights</span>
            </div>
            <div className="space-y-4 text-xs">
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">Pose & Gaze Angle Weight:</span>
                <strong className="text-neutral-900 font-bold text-sm">0.35 (35%)</strong>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">Object Detector Weight:</span>
                <strong className="text-neutral-900 font-bold text-sm">0.30 (30%)</strong>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">Temporal Persistence Weight:</span>
                <strong className="text-neutral-900 font-bold text-sm">0.25 (25%)</strong>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-600 font-medium">Spatial Context Suppression:</span>
                <strong className="text-emerald-700 font-bold text-sm">Active (-0.40 Risk Offset)</strong>
              </div>
            </div>
          </ParallaxCard>
        </div>
      )}
    </div>
  );
};
