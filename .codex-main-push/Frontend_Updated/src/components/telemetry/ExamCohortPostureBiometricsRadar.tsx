import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Activity, Shield, Sparkles, User, CheckCircle2, AlertTriangle } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';

export const ExamCohortPostureBiometricsRadar: React.FC = () => {
  const [selectedCandidate, setSelectedCandidate] = useState<'A06' | 'B04' | 'A01'>('A06');

  const candidatesData = {
    A06: {
      id: 'A06',
      candidate: 'Candidate #6',
      status: 'review',
      metrics: {
        gazeCentration: 42, // Low screen focus
        yawSymmetry: 38,    // High lateral bias
        wristStability: 32, // Off keyboard
        spineSlump: 85,     // Forward hunch
        fidgetRate: 68,     // High micro-jerk
      },
      finding: 'Heavy downward lap bias with off-keyboard wrist transition (0.74 phone cue).',
    },
    B04: {
      id: 'B04',
      candidate: 'Candidate #14',
      status: 'review',
      metrics: {
        gazeCentration: 58,
        yawSymmetry: 44,
        wristStability: 82,
        spineSlump: 70,
        fidgetRate: 74,
      },
      finding: 'Repeated rotational turns back toward central aisle (+38° yaw bursts).',
    },
    A01: {
      id: 'A01',
      candidate: 'Candidate #1',
      status: 'normal',
      metrics: {
        gazeCentration: 94,
        yawSymmetry: 92,
        wristStability: 96,
        spineSlump: 88,
        fidgetRate: 22,
      },
      finding: 'Nominal screen focus and consistent keyboard posture.',
    },
  };

  const active = candidatesData[selectedCandidate];

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-violet-50 text-violet-700 border border-violet-100">
              <Activity size={18} />
            </div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Cohort Biometric Posture Radar & Kinematics
            </h3>
            <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-violet-50 text-violet-800 border border-violet-200">
              POSE KINEMATICS
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">
            Multi-axis spatial posture symmetry comparison against the room-wide cohort baseline.
          </p>
        </div>

        {/* Candidate Switcher */}
        <div className="flex items-center gap-2">
          {(['A06', 'B04', 'A01'] as const).map((seat) => (
            <button
              key={seat}
              onClick={() => {
                playHapticSound('click');
                setSelectedCandidate(seat);
              }}
              className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold transition-all cursor-pointer ${
                selectedCandidate === seat
                  ? 'bg-neutral-900 text-white shadow-xs'
                  : 'bg-neutral-100 text-neutral-600 hover:text-neutral-900'
              }`}
            >
              Terminal {seat}
            </button>
          ))}
        </div>
      </div>

      {/* Radar Matrix & Metric Bars Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
        {/* Visual Biometric Polygon */}
        <div className="md:col-span-6 p-6 rounded-3xl bg-neutral-950 text-white border border-neutral-900 flex flex-col justify-between space-y-4">
          <div className="flex items-center justify-between text-2xs font-mono text-neutral-400">
            <span className="text-violet-300 font-bold">5-AXIS KINEMATIC POLYGON</span>
            <span>Room Baseline: Green (90%)</span>
          </div>

          <div className="relative aspect-square max-w-[280px] mx-auto flex items-center justify-center my-2 select-none">
            {/* Background Axis Circles */}
            <div className="absolute inset-4 rounded-full border border-dashed border-white/15" />
            <div className="absolute inset-12 rounded-full border border-dashed border-white/10" />
            <div className="absolute inset-20 rounded-full border border-dashed border-white/5" />

            {/* Baseline Polygon (Soft Mint) */}
            <div className="absolute inset-8 rounded-full border-2 border-emerald-400/40 bg-emerald-500/10 rotate-12" />

            {/* Candidate Polygon (Soft Rose/Amber) */}
            <div
              className={`absolute rounded-2xl border-2 transition-all duration-300 ${
                active.status === 'review'
                  ? 'inset-12 border-rose-400 bg-rose-500/20 rotate-45'
                  : 'inset-6 border-emerald-400 bg-emerald-500/20'
              }`}
            />

            <span className="relative z-10 font-mono font-bold text-xs text-white bg-black/60 px-2 py-1 rounded-md">
              {active.id} vs. Cohort
            </span>
          </div>

          <div className="text-3xs font-mono text-neutral-400 text-center">
            Normalized 0–100 Biometric Posture Index
          </div>
        </div>

        {/* 5 Biometric Metrics Breakdown */}
        <div className="md:col-span-6 space-y-4 text-xs font-sans">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-200">
            <div>
              <span className="font-heading font-bold text-base text-neutral-950">{active.candidate} ({active.id})</span>
              <div className="text-2xs text-neutral-500 font-mono">Posture Health Signature</div>
            </div>
            <StatusBadge status={active.status as any} size="sm" />
          </div>

          <div className="space-y-3">
            {[
              { label: 'Screen Gaze Centration', val: active.metrics.gazeCentration, base: 92 },
              { label: 'Head Yaw Symmetry', val: active.metrics.yawSymmetry, base: 90 },
              { label: 'Wrist Keyboard Stability', val: active.metrics.wristStability, base: 95 },
              { label: 'Spine Slump Continuity', val: active.metrics.spineSlump, base: 88 },
              { label: 'Micro-Fidget Index (Lower is stable)', val: active.metrics.fidgetRate, base: 20 },
            ].map((m, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-2xs font-medium">
                  <span className="text-neutral-700">{m.label}</span>
                  <span className="font-mono font-bold text-neutral-950">{m.val}% (Cohort: {m.base}%)</span>
                </div>
                <div className="h-2 rounded-full bg-neutral-100 overflow-hidden relative">
                  <div
                    style={{ width: `${m.val}%` }}
                    className={`h-full rounded-full transition-all duration-300 ${
                      m.val < 50
                        ? 'bg-rose-400'
                        : m.val < 75
                        ? 'bg-amber-400'
                        : 'bg-emerald-400'
                    }`}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="p-3.5 rounded-2xl bg-neutral-50 border border-neutral-200 text-xs text-neutral-700 leading-relaxed font-medium">
            💡 {active.finding}
          </div>
        </div>
      </div>
    </div>
  );
};
