import React from 'react';
import { motion } from 'motion/react';
import { Smartphone, ShieldAlert, ArrowRight, Video } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { TriagedAIEvent } from '../../types';

interface CriticalAnomalyHeroCardProps {
  onReviewEvent?: () => void;
  activeEvent?: TriagedAIEvent;
}

export const CriticalAnomalyHeroCard: React.FC<CriticalAnomalyHeroCardProps> = ({
  onReviewEvent,
  activeEvent,
}) => {
  return (
    <div className="p-7 rounded-[2rem] bg-[#242323] text-[#FAF7F3] border border-white/10 shadow-[0_24px_48px_rgba(0,0,0,0.14)] flex flex-col justify-between space-y-6 h-full">
      <div>
        {/* Top Kicker & Badge */}
        <div className="flex items-center justify-between pb-3 border-b border-white/10">
          <div className="text-3xs font-mono font-bold uppercase tracking-widest text-neutral-400">
            ACTIVE HIGH-PRIORITY ANOMALY INTERCEPT
          </div>
          <span className="px-3 py-1 rounded-full bg-rose-500/20 text-rose-300 font-mono text-2xs font-bold border border-rose-500/30 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-pulse" />
            SEAT A06 • REVIEW
          </span>
        </div>

        {/* Large Display Value */}
        <div className="mt-4">
          <div className="font-heading font-extrabold text-2xl sm:text-3xl lg:text-4xl text-white tracking-tight">
            Candidate #6 / 15:42:18 IST
          </div>
        </div>
      </div>

      {/* 2 Sub-panels at Bottom */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1">
          <div className="text-3xs font-mono font-bold uppercase text-neutral-400">
            DETECTED ANOMALY
          </div>
          <div className="text-sm font-heading font-bold text-white leading-snug">
            Possible Phone Interaction (Lap Area)
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1">
          <div className="text-3xs font-mono font-bold uppercase text-neutral-400">
            PROTOCOL ACTION
          </div>
          <div className="text-sm font-heading font-bold text-white leading-snug">
            EV-2038 / Forensic Replay & Confirmation
          </div>
        </div>
      </div>
    </div>
  );
};
