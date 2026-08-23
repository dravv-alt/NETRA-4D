import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Eye, Shield, Activity, Sliders, Filter, Sparkles, AlertTriangle } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';

export const GazeAttentionHeatmap: React.FC = () => {
  const [mode, setMode] = useState<'heat' | 'convergence' | 'suppression'>('heat');
  const [sensitivity, setSensitivity] = useState<number>(75);

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-violet-50 text-violet-700 border border-violet-100">
              <Eye size={18} />
            </div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Spatial Attention Heatmap & Gaze Convergence
            </h3>
            <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-violet-50 text-violet-800 border border-violet-200">
              Spatial Graph Network
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">
            Hall-wide spatial gaze probability distribution, cross-terminal mutual attention convergence, and contextual glance suppression.
          </p>
        </div>

        {/* Mode Selector */}
        <div className="flex items-center gap-2">
          {(['heat', 'convergence', 'suppression'] as const).map((m) => (
            <button
              key={m}
              onClick={() => {
                playHapticSound('click');
                setMode(m);
              }}
              className={`px-3.5 py-2 rounded-xl text-xs font-mono font-semibold transition-all cursor-pointer ${
                mode === m
                  ? 'bg-neutral-900 text-white shadow-xs'
                  : 'bg-neutral-100 text-neutral-600 hover:text-neutral-900 hover:bg-neutral-200'
              }`}
            >
              {m === 'heat' && '🔥 Gaze Density Heatmap'}
              {m === 'convergence' && '⚡ Peer Convergence Mesh'}
              {m === 'suppression' && '🛡️ Spatial Context Mask'}
            </button>
          ))}
        </div>
      </div>

      {/* Main Heatmap Visualizer Canvas */}
      <div className="relative aspect-21/9 min-h-[340px] bg-radial from-neutral-900 via-[#181716] to-[#0F0E0D] rounded-3xl border border-neutral-900 p-6 flex flex-col justify-between overflow-hidden shadow-inner select-none">
        {/* Top Status Bar */}
        <div className="flex items-center justify-between text-2xs font-mono text-neutral-400 z-10">
          <div className="flex items-center gap-3">
            <span className="text-violet-400 font-bold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
              HEAT DENSITY KERNEL: GAUSSIAN 2D
            </span>
            <span>•</span>
            <span>Laptop Focus Cohort: 89.2%</span>
          </div>

          <div className="flex items-center gap-2">
            <span>Filter Sensitivity:</span>
            <input
              type="range"
              min="20"
              max="100"
              value={sensitivity}
              onChange={(e) => setSensitivity(Number(e.target.value))}
              className="w-24 accent-neutral-200 cursor-pointer"
            />
            <span className="font-bold text-white">{sensitivity}%</span>
          </div>
        </div>

        {/* Heat Map Overlay Projection Grid */}
        <div className="my-auto w-full max-w-4xl mx-auto relative py-6">
          {/* North Desk Focus Line */}
          <div className="flex items-center justify-around relative">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="relative flex flex-col items-center group">
                <div
                  className={`w-12 h-12 rounded-full blur-md transition-all ${
                    i === 5
                      ? 'bg-amber-500/80 scale-150 animate-pulse'
                      : 'bg-emerald-500/40 group-hover:scale-125'
                  }`}
                />
                <span className="relative z-10 font-mono text-3xs text-neutral-300 font-bold mt-1">
                  A0{i + 1 === 10 ? '10' : i + 1}
                </span>
              </div>
            ))}
          </div>

          {/* Cross Peer Convergence Line (Between Seat B08 & B09) */}
          {mode === 'convergence' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute left-[70%] top-[40%] px-4 py-1.5 rounded-full bg-rose-500/30 border border-rose-500/60 text-rose-200 font-mono text-3xs flex items-center gap-2 animate-pulse"
            >
              <AlertTriangle size={12} className="text-rose-400" />
              <span>MUTUAL GAZE CONVERGENCE (SEAT B08 ↔ B09) • 4.8s</span>
            </motion.div>
          )}

          {/* South Desk Focus Line */}
          <div className="flex items-center justify-around relative mt-10">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="relative flex flex-col items-center group">
                <div
                  className={`w-12 h-12 rounded-full blur-md transition-all ${
                    i === 7 || i === 8
                      ? 'bg-rose-500/80 scale-140 animate-pulse'
                      : i === 3
                      ? 'bg-amber-500/70 scale-130'
                      : 'bg-emerald-500/40 group-hover:scale-125'
                  }`}
                />
                <span className="relative z-10 font-mono text-3xs text-neutral-300 font-bold mt-1">
                  B0{i + 1 === 10 ? '10' : i + 1}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Heatmap Legend */}
        <div className="z-10 flex items-center justify-between text-2xs font-mono text-neutral-400 border-t border-white/10 pt-3">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              Nominal Screen Focus (&lt;15° Yaw)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
              Lateral Gaze (&gt;25° Yaw)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
              Mutual Peer Convergence (&gt;3s)
            </span>
          </div>

          <span className="text-neutral-500">Auto-Filtered: Front Wall Timer Polygon (Suppressed)</span>
        </div>
      </div>
    </div>
  );
};
