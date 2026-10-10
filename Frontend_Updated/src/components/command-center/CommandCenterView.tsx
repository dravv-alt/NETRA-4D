import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SeatNodeData, TriagedAIEvent, SystemHealthData, CameraFeedInfo } from '../../types';
import { ExamHallMap } from './ExamHallMap';
import { CommandCenterInspectorPanel } from './CommandCenterInspectorPanel';
import { IncidentTriageWorkflowMaster } from '../telemetry/IncidentTriageWorkflowMaster';
import { ParallaxContainer } from '../ui/parallax-container';
import {
  ShieldCheck,
  Activity,
  Users,
  AlertTriangle,
  CheckCircle2,
  Lock,
  Sparkles,
  Radio,
  Clock,
  Compass
} from 'lucide-react';
import { StatusBadge } from '../ui/StatusBadge';
import { playHapticSound } from '../../lib/utils';

interface CommandCenterViewProps {
  seats: SeatNodeData[];
  events: TriagedAIEvent[];
  health: SystemHealthData;
  onSelectSeat: (seatId: string) => void;
  onReviewEvent: (event: TriagedAIEvent) => void;
  selectedSeatId?: string | null;
  onUpdateSeats?: (newSeats: SeatNodeData[]) => void;
  cameras?: CameraFeedInfo[];
  onViewCandidateDossier?: (seatId: string) => void;
}

export const CommandCenterView: React.FC<CommandCenterViewProps> = ({
  seats,
  events,
  health,
  onSelectSeat,
  onReviewEvent,
  selectedSeatId,
  onUpdateSeats,
  cameras,
  onViewCandidateDossier,
}) => {
  const selectedSeat = seats.find((s) => s.id === selectedSeatId) || null;
  const needsReviewCount = events.filter((e) => e.severity === 'review').length;
  const escalatedCount = events.filter((e) => e.severity === 'escalated').length;
  const normalCount = seats.filter((s) => s.state === 'normal').length;

  // Rolling 42-frame live telemetry capsules
  const [capsules, setCapsules] = useState<('normal' | 'review' | 'escalated')[]>(() => {
    const arr: ('normal' | 'review' | 'escalated')[] = [];
    for (let i = 0; i < 42; i++) {
      if (i === 12 || i === 28) arr.push('review');
      else if (i === 38) arr.push('escalated');
      else arr.push('normal');
    }
    return arr;
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setCapsules((prev) => {
        const rand = Math.random();
        const nextState = rand > 0.94 ? 'escalated' : rand > 0.82 ? 'review' : 'normal';
        return [...prev.slice(1), nextState];
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <ParallaxContainer depth={20} className="space-y-4 sm:space-y-8 max-w-7xl mx-auto font-sans">
      {/* ========================================================================= */}
      {/* MASTER SUITE 1: UNIFIED LIVE SURVEILLANCE COCKPIT (ALL IN ONE MASTER DECK) */}
      {/* ========================================================================= */}
      <div className="p-3.5 sm:p-8 sm:p-9 rounded-2xl sm:rounded-[2.5rem] bg-white border border-neutral-200/90 shadow-[0_24px_50px_rgba(0,0,0,0.04)] space-y-4 sm:space-y-7">
        {/* 1. Master Header: Ingestion Stream Status & Live 42-Capsule Rolling Strip */}
        <div className="p-3.5 sm:p-6 rounded-2xl sm:rounded-3xl bg-[#181716] text-white flex flex-col justify-between space-y-3 sm:space-y-4 shadow-sm select-none">
          <div className="flex items-center justify-between flex-wrap gap-2.5 sm:gap-3">
            <div className="flex items-center gap-2.5 sm:gap-3">
              <div className="p-1.5 sm:p-2 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                <Radio size={15} className="animate-pulse" />
              </div>
              <div>
                <div className="font-bold text-sm sm:text-base text-white flex items-center gap-2 sm:gap-2.5 flex-wrap tracking-tight">
                  <span>Surveillance Stream Continuity</span>
                  <span className="text-2xs font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    99.8% INGESTION
                  </span>
                </div>
                <div className="text-xs text-neutral-400 font-sans mt-0.5">
                  CBT Session 01 • 4 Synchronized RTSP Feeds & Multi-Modal Ingestion DAG
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-3 text-xs font-mono">
              <span className="text-neutral-400 font-sans">Latency:</span>
              <span className="px-2.5 py-0.5 sm:px-3 sm:py-1 rounded-lg sm:rounded-xl bg-white/10 text-white font-bold border border-white/10">
                {health.pipelineLatencyMs}ms
              </span>
            </div>
          </div>

          {/* Rolling Capsule Bar */}
          <div className="space-y-1 pt-1.5 border-t border-white/10">
            <div className="flex items-center justify-between text-2xs font-sans text-neutral-400 px-1">
              <span>-42s Rolling Telemetry Buffer</span>
              <span className="text-emerald-400 flex items-center gap-1 font-medium font-sans">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Live Ingestion (T+0s)
              </span>
            </div>
            <div className="h-3 sm:h-3.5 flex items-center gap-0.5 sm:gap-1.5 bg-black/40 p-0.5 sm:p-1 rounded-full overflow-hidden border border-white/5">
              {capsules.map((state, idx) => (
                <div
                  key={idx}
                  className={`flex-1 h-full rounded-full transition-all duration-300 ${
                    state === 'escalated'
                      ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.7)]'
                      : state === 'review'
                      ? 'bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.5)]'
                      : 'bg-emerald-400/90'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* 2. Integrated 4-KPI Metric Row (Level 2 Insets with #F5F4F0) */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 sm:gap-4">
          <div className="p-3.5 sm:p-5 rounded-xl sm:rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
                Active Terminals
              </div>
              <div className="font-bold text-2xl sm:text-3xl text-neutral-950 font-mono tabular-nums tracking-tight">
                {seats.length}
              </div>
              <div className="text-xs text-neutral-500 font-sans">20 / 20 Ingested</div>
            </div>
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-white flex items-center justify-center text-neutral-700 shadow-2xs border border-neutral-200/80">
              <Users size={16} />
            </div>
          </div>

          <div className="p-3.5 sm:p-5 rounded-xl sm:rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
                Nominal Focus
              </div>
              <div className="font-bold text-2xl sm:text-3xl text-emerald-700 font-mono tabular-nums tracking-tight">
                {normalCount}
              </div>
              <div className="text-xs text-emerald-800 font-sans">
                {((normalCount / seats.length) * 100).toFixed(0)}% of Cohort
              </div>
            </div>
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center shadow-2xs border border-emerald-200">
              <CheckCircle2 size={16} />
            </div>
          </div>

          <div className="p-3.5 sm:p-5 rounded-xl sm:rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
                Needs Review
              </div>
              <div className="font-bold text-2xl sm:text-3xl text-amber-700 font-mono tabular-nums tracking-tight">
                {needsReviewCount}
              </div>
              <div className="text-xs text-amber-800 font-sans">Elevated Anomaly Cues</div>
            </div>
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center shadow-2xs border border-amber-200">
              <AlertTriangle size={16} />
            </div>
          </div>

          <div className="p-3.5 sm:p-5 rounded-xl sm:rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
                Inference DAG
              </div>
              <div className="font-bold text-2xl sm:text-3xl text-neutral-950 font-mono tabular-nums tracking-tight">
                {health.pipelineLatencyMs}ms
              </div>
              <div className="text-xs text-neutral-500 font-sans">{health.t4GpuUtilPct}% T4 Load</div>
            </div>
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-sky-50 text-sky-700 flex items-center justify-center shadow-2xs border border-sky-200">
              <Activity size={16} />
            </div>
          </div>
        </div>

        {/* 3. Main Workspace: Spatial Lab Terminal Grid (8 cols) + Contextual Inspector (4 cols) */}
        <div className="grid grid-cols-12 gap-3.5 sm:gap-6 items-start">
          <div className="col-span-12 lg:col-span-8">
            <ExamHallMap
              seats={seats}
              onSelectSeat={onSelectSeat}
              selectedSeatId={selectedSeatId}
              onUpdateSeats={onUpdateSeats}
              cameras={cameras}
            />
          </div>

          <div className="col-span-12 lg:col-span-4">
            <CommandCenterInspectorPanel
              selectedSeat={selectedSeat}
              onClearSelectedSeat={() => onSelectSeat('')}
              events={events}
              onReviewEvent={onReviewEvent}
              onViewCandidateDossier={onViewCandidateDossier}
            />
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MASTER SUITE 2: ADJUDICATION WORKFLOW & INTEGRATED COHORT DYNAMICS */}
      {/* ========================================================================= */}
      <IncidentTriageWorkflowMaster />
    </ParallaxContainer>
  );
};
