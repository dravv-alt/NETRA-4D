import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SystemHealthData } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { X, Activity, Cpu, HardDrive, ShieldCheck, CheckCircle2, Server, Clock } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';

interface SystemHealthModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: SystemHealthData;
}

export const SystemHealthModal: React.FC<SystemHealthModalProps> = ({
  isOpen,
  onClose,
  health,
}) => {
  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Dimmed Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onClick={() => {
            playHapticSound('click');
            onClose();
          }}
          className="fixed inset-0 bg-black/40 backdrop-blur-xs"
        />

        {/* Modal Window */}
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 8 }}
          transition={{ type: 'spring', damping: 26, stiffness: 350 }}
          className="relative z-50 w-full max-w-2xl bg-white border border-neutral-200 rounded-2xl p-6 font-mono text-xs shadow-2xl space-y-5 text-neutral-900"
        >
          {/* Header */}
          <div className="flex items-center justify-between pb-3 border-b border-neutral-200">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-neutral-100 border border-neutral-200 flex items-center justify-center text-neutral-900">
                <Activity size={18} />
              </div>
              <div>
                <h3 className="font-bold text-sm text-neutral-950 uppercase tracking-wider flex items-center gap-2">
                  <span>Controlled Runtime Diagnostics</span>
                  <StatusBadge status="AVAILABLE" label="DEMO" size="sm" />
                </h3>
                <p className="text-2xs text-neutral-500 mt-0.5">
                  Interface sample values; validated GPU provenance is shown in the Validated Run page
                </p>
              </div>
            </div>

            <button
              onClick={() => {
                playHapticSound('click');
                onClose();
              }}
              className="p-1.5 rounded-lg hover:bg-neutral-100 text-neutral-400 hover:text-neutral-900 transition-colors cursor-pointer"
            >
              <X size={18} />
            </button>
          </div>

          {/* Core Hardware Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-neutral-50 p-3 rounded-xl border border-neutral-200 space-y-1">
              <div className="text-2xs text-neutral-500 uppercase flex items-center gap-1">
                <Cpu size={12} className="text-neutral-700" />
                <span>GPU Load</span>
              </div>
              <div className="font-bold text-lg text-neutral-950 tabular-nums">
                {health.t4GpuUtilPct}%
              </div>
              <div className="text-3xs text-neutral-600 font-bold">Nominal</div>
            </div>

            <div className="bg-neutral-50 p-3 rounded-xl border border-neutral-200 space-y-1">
              <div className="text-2xs text-neutral-500 uppercase flex items-center gap-1">
                <HardDrive size={12} className="text-neutral-700" />
                <span>VRAM Load</span>
              </div>
              <div className="font-bold text-lg text-neutral-950 tabular-nums">
                {health.vramUsedGb} <span className="text-xs text-neutral-500 font-normal">/ {health.vramTotalGb}GB</span>
              </div>
              <div className="text-3xs text-neutral-500">43% Used</div>
            </div>

            <div className="bg-neutral-50 p-3 rounded-xl border border-neutral-200 space-y-1">
              <div className="text-2xs text-neutral-500 uppercase flex items-center gap-1">
                <Activity size={12} className="text-neutral-700" />
                <span>Temp</span>
              </div>
              <div className="font-bold text-lg text-neutral-950 tabular-nums">
                {health.t4GpuTempC}°C
              </div>
              <div className="text-3xs text-neutral-600 font-bold">Optimal</div>
            </div>

            <div className="bg-neutral-50 p-3 rounded-xl border border-neutral-200 space-y-1">
              <div className="text-2xs text-neutral-500 uppercase flex items-center gap-1">
                <Clock size={12} className="text-neutral-700" />
                <span>Latency</span>
              </div>
              <div className="font-bold text-lg text-neutral-950 tabular-nums">
                {health.pipelineLatencyMs} ms
              </div>
              <div className="text-3xs text-neutral-600 font-bold">&lt; 50ms</div>
            </div>
          </div>

          {/* Model Engines Status Card */}
          <div className="p-4 rounded-xl border border-neutral-200 space-y-2.5 bg-white">
            <div className="font-bold text-2xs uppercase text-neutral-700 tracking-wider flex items-center justify-between pb-1.5 border-b border-neutral-200">
              <span>Active Inference Models & Fallback Pipelines</span>
              <Server size={13} className="text-neutral-900" />
            </div>

            <div className="space-y-2 text-2xs">
              <div className="flex items-center justify-between p-2 rounded-lg bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-800 font-medium">YOLOv8-Pose (Full-Frame Edge Detector)</span>
                <span className="text-neutral-900 font-bold flex items-center gap-1">
                  <CheckCircle2 size={12} /> PRIMARY • 18.3 FPS
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-800 font-medium">RTMPose (High-Precision 2D Keypoint Refiner)</span>
                <span className="text-neutral-900 font-bold flex items-center gap-1">
                  <CheckCircle2 size={12} /> REFINEMENT • 8.6 FPS
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-800 font-medium">Multi-HMR (Selective 3D Mesh Reconstruction)</span>
                <span className="text-neutral-900 font-bold flex items-center gap-1">
                  <CheckCircle2 size={12} /> ON-DEMAND (Flag Triggered)
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-neutral-50 border border-neutral-200">
                <span className="text-neutral-800 font-medium">Temporal Evidence Weighting & Context Suppressor</span>
                <span className="text-neutral-900 font-bold flex items-center gap-1">
                  <ShieldCheck size={12} /> ACTIVE (60% FP Reduction)
                </span>
              </div>
            </div>
          </div>

          {/* Footer Action */}
          <div className="flex justify-end pt-2">
            <Button
              variant="default"
              size="sm"
              onClick={() => {
                playHapticSound('click');
                onClose();
              }}
              className="rounded-xl px-5"
            >
              Close Diagnostics
            </Button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
