import React from 'react';
import { motion } from 'motion/react';
import { CheckCircle2 } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';

export const InvestigationLifecycleStepper: React.FC = () => {
  const steps = [
    { id: 1, label: 'Anomaly Flagged', active: true },
    { id: 2, label: 'Context Verified', active: false },
    { id: 3, label: 'Forensic Replay', active: false },
    { id: 4, label: 'Proctor Intercept', active: false },
    { id: 5, label: 'Audit Sealed', active: false },
    { id: 6, label: 'Disciplinary Export', active: false },
    { id: 7, label: 'Case Resolved', active: false },
  ];

  return (
    <div className="p-7 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-heading font-extrabold text-lg text-neutral-950 tracking-tight">
            Investigation lifecycle
          </h3>
          <p className="text-xs text-neutral-500 font-sans mt-0.5">
            Current state: AI Anomaly Flagged
          </p>
        </div>

        <span className="px-3 py-1 rounded-full bg-neutral-100 text-neutral-900 font-mono text-xs font-bold border border-neutral-200 uppercase">
          TRIAGED
        </span>
      </div>

      {/* Step Pills Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {steps.map((step) => (
          <div
            key={step.id}
            className={`p-3 rounded-2xl flex items-center gap-2.5 transition-all text-xs font-sans font-semibold ${
              step.active
                ? 'bg-[#242323] text-white shadow-xs border border-neutral-900'
                : 'bg-neutral-50 text-neutral-500 border border-neutral-200/80 hover:bg-neutral-100 hover:text-neutral-900'
            }`}
          >
            <span
              className={`w-5 h-5 rounded-full flex items-center justify-center font-mono text-2xs font-bold shrink-0 ${
                step.active ? 'bg-white text-neutral-950' : 'bg-neutral-200 text-neutral-600'
              }`}
            >
              {step.id}
            </span>
            <span className="truncate">{step.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
