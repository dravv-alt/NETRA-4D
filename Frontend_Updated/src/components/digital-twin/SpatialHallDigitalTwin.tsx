import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Compass,
  Camera,
  Layers,
  Activity,
  Maximize2,
  Eye,
  Laptop,
  User,
  Shield,
  RotateCw,
  Sparkles,
  ArrowRight,
  Video,
  Zap
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';
import { SeatNodeData } from '../../types';

interface SpatialHallDigitalTwinProps {
  seats: SeatNodeData[];
  onSelectSeat?: (seatId: string) => void;
  selectedSeatId?: string | null;
}

export const SpatialHallDigitalTwin: React.FC<SpatialHallDigitalTwinProps> = ({
  seats,
  onSelectSeat,
  selectedSeatId,
}) => {
  const [viewMode, setViewMode] = useState<'assignments' | 'attention' | 'interactions' | 'evidence'>('assignments');
  const [activeSeat, setActiveSeat] = useState<string | null>(selectedSeatId || 'A06');

  useEffect(() => {
    if (selectedSeatId) setActiveSeat(selectedSeatId);
  }, [selectedSeatId]);

  const rowA = seats.filter((s) => s.gridPosition.row === 'A');
  const rowB = seats.filter((s) => s.gridPosition.row === 'B');

  const selectedSeatData = seats.find((s) => s.id === activeSeat) || seats[5];

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
              <Compass size={18} />
            </div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Candidate–PC Context & Interaction Map
            </h3>
            <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
              CBT TEMPORAL CONTEXT
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">
            Candidate assignments, associated screens, head-direction proxies, pairwise interactions, and human-review evidence across the examination lab.
          </p>
        </div>

        {/* View Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          {(['assignments', 'attention', 'interactions', 'evidence'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => {
                playHapticSound('click');
                setViewMode(mode);
              }}
              className={`px-3.5 py-2 rounded-xl text-xs font-mono font-semibold transition-all cursor-pointer ${
                viewMode === mode
                  ? 'bg-neutral-900 text-white shadow-xs'
                  : 'bg-neutral-100 text-neutral-600 hover:text-neutral-900 hover:bg-neutral-200'
              }`}
            >
              {mode === 'assignments' && 'Candidate ↔ PC'}
              {mode === 'attention' && 'Head Direction'}
              {mode === 'interactions' && 'Human Interaction'}
              {mode === 'evidence' && 'Evidence State'}
            </button>
          ))}
        </div>
      </div>

      {/* Candidate-workstation context viewport */}
      <div className="relative aspect-21/9 min-h-[360px] bg-[#1A1918] rounded-3xl border border-neutral-900 overflow-hidden shadow-inner flex flex-col justify-between p-6 select-none">
        {/* Top HUD Overlay */}
        <div className="flex items-center justify-between z-10 text-2xs font-mono text-neutral-400">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-emerald-400 font-bold">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              TEMPORAL CONTEXT ENGINE ACTIVE
            </span>
            <span>•</span>
            <span>Assigned seats and screens</span>
            <span>•</span>
            <span>Human review required</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-lg bg-white/10 text-white border border-white/15">
              CANDIDATE • PC • GAZE PROXY • INTERACTION
            </span>
          </div>
        </div>

        {/* Contextual examination-lab map */}
        <div className="my-auto w-full max-w-5xl mx-auto py-2">
          {/* North Desk (Row A - Facing North ⬆) */}
          <div className="space-y-2">
            <div className="text-3xs font-mono font-bold uppercase tracking-widest text-sky-400/80 text-center">
              ▲ NORTH WALL DESK — ROW A (STUDENTS FACING NORTH LAPTOPS) ▲
            </div>

            <div className="grid grid-cols-10 gap-2.5">
              {rowA.map((seat) => {
                const isSelected = activeSeat === seat.id;
                const isReview = seat.state === 'review';
                const isEscalated = seat.state === 'escalated';

                return (
                  <motion.div
                    key={seat.id}
                    onClick={() => {
                      playHapticSound('snap');
                      setActiveSeat(seat.id);
                      onSelectSeat?.(seat.id);
                    }}
                    whileHover={{ scale: 1.08, y: -4 }}
                    className={`relative p-2.5 rounded-xl border transition-all cursor-pointer text-center font-mono ${
                      isSelected
                        ? 'bg-white text-neutral-950 border-white shadow-lg ring-2 ring-sky-400'
                        : isEscalated
                        ? 'bg-rose-950/80 border-rose-500 text-rose-200'
                        : isReview
                        ? 'bg-amber-950/80 border-amber-500 text-amber-200'
                        : 'bg-neutral-900/90 border-neutral-700 text-neutral-300 hover:border-neutral-500'
                    }`}
                  >
                    {/* Laptop Screen Indicator */}
                    <div className="w-full h-1.5 rounded-full bg-sky-400/80 mb-1.5" />

                    <div className="font-bold text-xs">{seat.id}</div>
                    <div className="text-3xs text-neutral-400">#{seat.candidateId}</div>

                    {/* Dynamic Vector Ray Visualizer */}
                    {viewMode === 'attention' && (
                      <div className="mt-1.5 flex justify-center">
                        <div
                          className={`w-1 h-3.5 rounded-full ${
                            isEscalated
                              ? 'bg-rose-500 animate-pulse'
                              : isReview
                              ? 'bg-amber-400 rotate-25'
                              : 'bg-emerald-400'
                          }`}
                          title={`Ray Vector: ${seat.telemetry.headEvidence}`}
                        />
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Central Proctor Walkway (Aisle) */}
          <div className="my-5 py-2.5 rounded-xl border border-dashed border-neutral-700 bg-neutral-900/40 text-center font-mono text-3xs text-neutral-400 tracking-wider flex items-center justify-center gap-3">
            <Shield size={13} className="text-neutral-500" />
            <span>CENTRAL PROCTOR WALKWAY (STUDENTS SIT BACK-TO-BACK)</span>
            <Shield size={13} className="text-neutral-500" />
          </div>

          {/* South Desk (Row B - Facing South ⬇) */}
          <div className="space-y-2">
            <div className="grid grid-cols-10 gap-2.5">
              {rowB.map((seat) => {
                const isSelected = activeSeat === seat.id;
                const isReview = seat.state === 'review';
                const isEscalated = seat.state === 'escalated';

                return (
                  <motion.div
                    key={seat.id}
                    onClick={() => {
                      playHapticSound('snap');
                      setActiveSeat(seat.id);
                      onSelectSeat?.(seat.id);
                    }}
                    whileHover={{ scale: 1.08, y: 4 }}
                    className={`relative p-2.5 rounded-xl border transition-all cursor-pointer text-center font-mono ${
                      isSelected
                        ? 'bg-white text-neutral-950 border-white shadow-lg ring-2 ring-sky-400'
                        : isEscalated
                        ? 'bg-rose-950/80 border-rose-500 text-rose-200'
                        : isReview
                        ? 'bg-amber-950/80 border-amber-500 text-amber-200'
                        : 'bg-neutral-900/90 border-neutral-700 text-neutral-300 hover:border-neutral-500'
                    }`}
                  >
                    {/* Dynamic Vector Ray Visualizer */}
                    {viewMode === 'attention' && (
                      <div className="mb-1.5 flex justify-center">
                        <div
                          className={`w-1 h-3.5 rounded-full ${
                            isEscalated
                              ? 'bg-rose-500 animate-pulse'
                              : isReview
                              ? 'bg-amber-400 -rotate-25'
                              : 'bg-emerald-400'
                          }`}
                          title={`Ray Vector: ${seat.telemetry.headEvidence}`}
                        />
                      </div>
                    )}

                    <div className="font-bold text-xs">{seat.id}</div>
                    <div className="text-3xs text-neutral-400">#{seat.candidateId}</div>

                    {/* Laptop Screen Indicator */}
                    <div className="w-full h-1.5 rounded-full bg-sky-400/80 mt-1.5" />
                  </motion.div>
                );
              })}
            </div>

            <div className="text-3xs font-mono font-bold uppercase tracking-widest text-sky-400/80 text-center">
              ▼ SOUTH WALL DESK — ROW B (STUDENTS FACING SOUTH LAPTOPS) ▼
            </div>
          </div>
        </div>

        {/* Bottom Digital Twin Telemetry Strip */}
        <div className="z-10 p-3 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md flex items-center justify-between text-xs font-mono text-white flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <span className="font-bold text-sky-300">Selected Node: Terminal {selectedSeatData.id} (Candidate #{selectedSeatData.candidateId})</span>
            <span className="text-neutral-400">•</span>
            <span>Track: <strong className="text-emerald-400">{selectedSeatData.telemetry.trackContinuity}</strong></span>
            <span>•</span>
            <span>Hands: <strong className="text-amber-300">{selectedSeatData.telemetry.handsEvidence}</strong></span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="default"
              size="xs"
              onClick={() => {
                playHapticSound('snap');
                onSelectSeat?.(selectedSeatData.id);
              }}
              className="bg-white text-neutral-950 hover:bg-neutral-100 rounded-lg text-2xs font-bold"
            >
              <span>Inspect Telemetry</span>
              <ArrowRight size={11} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
