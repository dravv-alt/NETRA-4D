import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  ShieldCheck,
  User,
  Radio,
  Bell,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Sparkles,
  Volume2,
  Eye
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';
import { SeatNodeData } from '../../types';

interface InvigilatorAislePatrolAssistantProps {
  seats?: SeatNodeData[];
  onSelectCandidate?: (seatId: string) => void;
}

export const InvigilatorAislePatrolAssistant: React.FC<InvigilatorAislePatrolAssistantProps> = ({
  seats = [],
  onSelectCandidate,
}) => {
  const [proctorPosition, setProctorPosition] = useState<number>(6); // Station 6 (between A06 and B06)
  const [activeActionLog, setActiveActionLog] = useState<string | null>(null);

  const nearestNorthSeat = `A0${proctorPosition === 10 ? '10' : proctorPosition}`;
  const nearestSouthSeat = `B0${proctorPosition === 10 ? '10' : proctorPosition}`;

  const handleAction = (actionName: string) => {
    playHapticSound('snap');
    setActiveActionLog(actionName);
    setTimeout(() => setActiveActionLog(null), 3500);
  };

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-800 border border-emerald-200">
              <ShieldCheck size={18} />
            </div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Invigilator Central Aisle Patrol Assistant
            </h3>
            <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
              PROCTOR RADAR 1.0
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">
            Real-time walkway corridor tracking, physical proximity radar, and silent candidate verification prompts.
          </p>
        </div>

        {/* Station Scrubber */}
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-neutral-500">Aisle Station:</span>
          <span className="px-3 py-1 rounded-xl bg-neutral-900 text-white font-mono text-xs font-bold shadow-xs">
            STATION {proctorPosition} / 10
          </span>
        </div>
      </div>

      {/* Corridor Visualizer */}
      <div className="p-6 rounded-3xl bg-neutral-50/80 border border-neutral-200 space-y-6 select-none">
        {/* Row A (North Facing) */}
        <div className="flex items-center justify-between gap-2">
          {Array.from({ length: 10 }).map((_, idx) => {
            const num = idx + 1;
            const isNear = num === proctorPosition;
            const seatId = `A0${num === 10 ? '10' : num}`;

            return (
              <div
                key={seatId}
                onClick={() => onSelectCandidate?.(seatId)}
                className={`flex-1 p-2.5 rounded-2xl border text-center transition-all cursor-pointer ${
                  isNear
                    ? 'bg-amber-50 border-amber-300 shadow-xs ring-2 ring-amber-400/40'
                    : 'bg-white border-neutral-200/80 hover:border-neutral-300'
                }`}
              >
                <div className="w-full h-1 bg-sky-300 rounded-full mb-1" />
                <span className="font-mono text-xs font-bold text-neutral-950 block">{seatId}</span>
                <span className="text-3xs text-neutral-400 font-mono">#{num}</span>
              </div>
            );
          })}
        </div>

        {/* Central Walkway Proctor Slider */}
        <div className="relative py-4 px-3 rounded-2xl bg-white border border-neutral-300 shadow-inner flex items-center">
          <div className="absolute left-6 right-6 h-1.5 bg-neutral-200 rounded-full" />
          <input
            type="range"
            min="1"
            max="10"
            step="1"
            value={proctorPosition}
            onChange={(e) => {
              playHapticSound('click');
              setProctorPosition(Number(e.target.value));
            }}
            className="w-full relative z-10 accent-neutral-900 cursor-pointer h-6 opacity-90"
          />
        </div>

        {/* Row B (South Facing) */}
        <div className="flex items-center justify-between gap-2">
          {Array.from({ length: 10 }).map((_, idx) => {
            const num = idx + 1;
            const isNear = num === proctorPosition;
            const seatId = `B0${num === 10 ? '10' : num}`;

            return (
              <div
                key={seatId}
                onClick={() => onSelectCandidate?.(seatId)}
                className={`flex-1 p-2.5 rounded-2xl border text-center transition-all cursor-pointer ${
                  isNear
                    ? 'bg-amber-50 border-amber-300 shadow-xs ring-2 ring-amber-400/40'
                    : 'bg-white border-neutral-200/80 hover:border-neutral-300'
                }`}
              >
                <span className="font-mono text-xs font-bold text-neutral-950 block">{seatId}</span>
                <span className="text-3xs text-neutral-400 font-mono">#{num + 10}</span>
                <div className="w-full h-1 bg-sky-300 rounded-full mt-1" />
              </div>
            );
          })}
        </div>
      </div>

      {/* Proximity Command Deck */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-amber-50/70 border border-amber-200 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-mono text-2xs uppercase text-amber-900 font-bold">Nearest Active Nodes</span>
            <Radio size={14} className="text-amber-700 animate-pulse" />
          </div>
          <div className="font-heading font-extrabold text-lg text-neutral-950">
            {nearestNorthSeat} & {nearestSouthSeat}
          </div>
          <p className="text-xs text-amber-900/80">
            Within 1.8m physical proctor observation field.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-neutral-200 space-y-3 flex flex-col justify-between">
          <span className="font-mono text-2xs uppercase text-neutral-400 font-bold">Physical Verification</span>
          <Button
            variant="default"
            size="sm"
            onClick={() => handleAction(`Physical desk audit recorded at Station ${proctorPosition} (${nearestNorthSeat} & ${nearestSouthSeat})`)}
            className="w-full rounded-xl text-xs font-semibold shadow-xs"
          >
            <CheckCircle2 size={13} className="mr-1.5" />
            <span>Log Physical Desk Audit</span>
          </Button>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-neutral-200 space-y-3 flex flex-col justify-between">
          <span className="font-mono text-2xs uppercase text-neutral-400 font-bold">Silent Proctor Prompt</span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleAction(`Silent posture reminder sent to Terminal ${nearestNorthSeat}`)}
            className="w-full rounded-xl text-xs font-semibold"
          >
            <Volume2 size={13} className="mr-1.5" />
            <span>Send Silent Prompt</span>
          </Button>
        </div>
      </div>

      {/* Action Notification */}
      <AnimatePresence>
        {activeActionLog && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            className="p-3.5 rounded-2xl bg-neutral-900 text-white font-mono text-xs flex items-center justify-between shadow-md"
          >
            <span className="flex items-center gap-2">
              <CheckCircle2 size={14} className="text-emerald-400" />
              {activeActionLog}
            </span>
            <span className="text-2xs text-neutral-400">Timestamp: Logged in Audit Trail</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
