import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { ShieldCheck, Activity, Camera } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';

export const SurveillanceSignalUptimeBar: React.FC = () => {
  // Generate 45 initial capsule states (mostly green, few amber/rose)
  const [capsules, setCapsules] = useState<{ id: number; status: 'green' | 'amber' | 'rose' }[]>(() => {
    const initial = [];
    for (let i = 0; i < 45; i++) {
      let status: 'green' | 'amber' | 'rose' = 'green';
      if (i === 12 || i === 28 || i === 39) status = 'amber';
      if (i === 41) status = 'rose';
      initial.push({ id: i, status });
    }
    return initial;
  });

  // 1Hz live rolling ingestion pulse
  useEffect(() => {
    const interval = setInterval(() => {
      setCapsules((prev) => {
        const rand = Math.random();
        let newStatus: 'green' | 'amber' | 'rose' = 'green';
        if (rand > 0.88) newStatus = 'amber';
        if (rand > 0.96) newStatus = 'rose';

        const updated = [...prev.slice(1), { id: Date.now(), status: newStatus }];
        return updated;
      });
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  const getCapsuleColor = (status: 'green' | 'amber' | 'rose') => {
    switch (status) {
      case 'rose':
        return 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.4)]';
      case 'amber':
        return 'bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.3)]';
      case 'green':
      default:
        return 'bg-emerald-400 hover:bg-emerald-300';
    }
  };

  return (
    <div className="p-7 rounded-[2rem] bg-[#1E1D1D] text-[#FAF7F3] border border-white/10 shadow-[0_24px_48px_rgba(0,0,0,0.12)] space-y-6">
      {/* Top Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="font-heading font-extrabold text-xl text-white tracking-tight">
              Surveillance Stream Continuity
            </span>
          </div>
          <div className="text-xs font-sans text-neutral-400 mt-1">
            CBT Session 01 • 4 Synchronized RTSP Feeds & Multi-Modal Ingestion DAG
          </div>
        </div>

        <div className="text-right">
          <span className="font-heading font-extrabold text-3xl sm:text-4xl text-white tracking-tight font-mono">
            99.8%
          </span>
        </div>
      </div>

      {/* 45 Horizontal Pill Capsules */}
      <div className="space-y-2">
        <div className="flex items-center gap-1.5 sm:gap-2">
          {capsules.map((cap, idx) => (
            <motion.div
              key={cap.id}
              layout
              transition={{ type: 'spring', damping: 25, stiffness: 400 }}
              className={`h-5 flex-1 rounded-full cursor-pointer transition-all duration-200 ${getCapsuleColor(cap.status)}`}
              title={`Frame -${45 - idx}s: ${cap.status.toUpperCase()}`}
            />
          ))}
        </div>

        {/* Bottom Legend Labels */}
        <div className="flex items-center justify-between text-2xs font-mono text-neutral-400 pt-1">
          <span>-45s Ingestion Buffer</span>
          <span>Live Synchronized (T+0s)</span>
        </div>
      </div>
    </div>
  );
};
