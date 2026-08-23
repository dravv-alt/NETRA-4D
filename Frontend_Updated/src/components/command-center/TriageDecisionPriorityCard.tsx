import React from 'react';
import { ShieldAlert, CheckCircle2, Clock } from 'lucide-react';

export const TriageDecisionPriorityCard: React.FC = () => {
  return (
    <div className="p-7 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] flex flex-col justify-between space-y-5 h-full">
      <div>
        <h3 className="font-heading font-extrabold text-lg text-neutral-950 tracking-tight">
          Triage decision priority
        </h3>
        <p className="text-xs text-neutral-500 font-sans mt-0.5">
          Multi-modal decision context.
        </p>
      </div>

      <div className="space-y-3.5 divide-y divide-neutral-100 text-xs">
        <div className="flex items-center justify-between pt-1">
          <span className="text-neutral-500 font-medium">Priority level</span>
          <span className="flex items-center gap-1.5 font-bold text-neutral-950 font-sans">
            <span className="w-2 h-2 rounded-full bg-rose-500 shrink-0" />
            High (Escalated)
          </span>
        </div>

        <div className="flex items-center justify-between pt-3">
          <span className="text-neutral-500 font-medium">Invigilator triage</span>
          <span className="flex items-center gap-1.5 font-bold text-neutral-950 font-sans">
            <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0" />
            In Review (Seat A06)
          </span>
        </div>

        <div className="flex items-center justify-between pt-3">
          <span className="text-neutral-500 font-medium">Context rule</span>
          <span className="flex items-center gap-1.5 font-bold text-neutral-950 font-sans">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
            Board timer glance active
          </span>
        </div>
      </div>
    </div>
  );
};
