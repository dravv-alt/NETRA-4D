import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Shield, Clock, CheckCircle2, AlertTriangle, ArrowRight, Activity, FileCheck, User, PieChart, TrendingUp, ShieldCheck } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';

interface IncidentTriageWorkflowMasterProps {
  onConfirmViolation?: () => void;
  onDismissAnomaly?: () => void;
}

export const IncidentTriageWorkflowMaster: React.FC<IncidentTriageWorkflowMasterProps> = ({
  onConfirmViolation,
  onDismissAnomaly,
}) => {
  const [selectedScenario, setSelectedScenario] = useState<1 | 2>(1);
  const [activeStep, setActiveStep] = useState<number>(3); // 1 to 4

  const behaviorCategories = [
    { label: 'Normal Screen Focus', pct: 80, color: 'bg-emerald-400', count: '16 Terminals' },
    { label: 'Routine Attention / Glance', pct: 10, color: 'bg-violet-400', count: '2 Terminals' },
    { label: 'Needs Invigilator Review', pct: 5, color: 'bg-amber-400', count: '1 Terminal' },
    { label: 'Critical Anomaly Cue', pct: 5, color: 'bg-rose-500', count: '1 Terminal' },
  ];

  const timeBuckets = [
    { time: '15:00', events: 1 },
    { time: '15:10', events: 2 },
    { time: '15:20', events: 0 },
    { time: '15:30', events: 3 },
    { time: '15:40', events: 2 },
  ];

  return (
    <div className="p-8 sm:p-9 rounded-[2.5rem] bg-white border border-neutral-200/90 shadow-[0_24px_50px_rgba(0,0,0,0.04)] space-y-8 font-sans">
      {/* ========================================================================= */}
      {/* SECTION 1: SURVEILLANCE INCIDENT ADJUDICATION CALCULATION */}
      {/* ========================================================================= */}
      <div className="space-y-6">
        {/* 1. Header: Title + Description + Top-Right SLA */}
        <div className="flex items-start justify-between flex-wrap gap-4 pb-1">
          <div className="space-y-1 max-w-2xl">
            <h3 className="font-heading font-extrabold text-2xl text-neutral-950 tracking-tight">
              Surveillance Incident Adjudication Calculation
            </h3>
            <p className="text-xs text-neutral-500 leading-relaxed font-sans">
              Select the observation hypothesis; the multi-modal evidence breakdown, spatial confidence, and legal dossier seal update deterministically.
            </p>
          </div>

          <div className="text-right">
            <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
              TOTAL TRIAGE RESOLUTION TIME
            </div>
            <div className="font-bold text-3xl sm:text-4xl text-neutral-950 font-mono tracking-tight tabular-nums mt-0.5">
              24s <span className="text-xs font-normal text-neutral-400 font-sans">/ Target: 30s</span>
            </div>
          </div>
        </div>

        {/* 2. Dual Scenario Selection Cards (Level 2 Insets) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Scenario 1: Physical Phone in Lap */}
          <div
            onClick={() => {
              playHapticSound('snap');
              setSelectedScenario(1);
            }}
            className={`p-5 rounded-2xl transition-all duration-200 cursor-pointer flex flex-col justify-between space-y-3 ${
              selectedScenario === 1
                ? 'bg-[#181716] text-[#FAF7F3] shadow-md ring-2 ring-neutral-900'
                : 'bg-[#F8F9FA] hover:bg-neutral-100 text-neutral-900 border border-neutral-200/80'
            }`}
          >
            <div className="space-y-1">
              <div className={`text-2xs font-semibold uppercase tracking-wider font-sans ${
                selectedScenario === 1 ? 'text-amber-400' : 'text-neutral-500'
              }`}>
                SCENARIO 1 / UNAUTHORIZED DEVICE IN LAP
              </div>
              <div className="font-bold text-sm sm:text-base leading-snug font-sans">
                Candidate #6 (Seat A06) — Phone screen illumination & downward lap vector.
              </div>
            </div>
            <div className={`text-xs font-sans pt-2 border-t ${
              selectedScenario === 1 ? 'border-white/10 text-neutral-400' : 'border-neutral-200 text-neutral-500'
            }`}>
              Confidence: 0.84 Peak • CAM-01 & CAM-02 Cross-Fused
            </div>
          </div>

          {/* Scenario 2: Benign Counter-Evidence */}
          <div
            onClick={() => {
              playHapticSound('snap');
              setSelectedScenario(2);
            }}
            className={`p-5 rounded-2xl transition-all duration-200 cursor-pointer flex flex-col justify-between space-y-3 ${
              selectedScenario === 2
                ? 'bg-[#181716] text-[#FAF7F3] shadow-md ring-2 ring-neutral-900'
                : 'bg-[#F8F9FA] hover:bg-neutral-100 text-neutral-900 border border-neutral-200/80'
            }`}
          >
            <div className="space-y-1">
              <div className={`text-2xs font-semibold uppercase tracking-wider font-sans ${
                selectedScenario === 2 ? 'text-emerald-400' : 'text-neutral-500'
              }`}>
                SCENARIO 2 / COUNTER-EVIDENCE & BENIGN VARIATION
              </div>
              <div className="font-bold text-sm sm:text-base leading-snug font-sans">
                Candidate #3 (Seat A03) — Wall countdown timer glance during section transition.
              </div>
            </div>
            <div className={`text-xs font-sans pt-2 border-t ${
              selectedScenario === 2 ? 'border-white/10 text-neutral-400' : 'border-neutral-200 text-neutral-500'
            }`}>
              Confidence: 0.22 (Suppressed by Room Geometry Mask)
            </div>
          </div>
        </div>

        {/* 3. Multi-Stage Progress Bar */}
        <div className="space-y-1.5">
          <div className="h-3 rounded-full bg-neutral-100 overflow-hidden relative border border-neutral-200/60">
            <div
              style={{ width: selectedScenario === 1 ? '75%' : '100%' }}
              className={`h-full rounded-full transition-all duration-500 ${
                selectedScenario === 1 ? 'bg-[#181716]' : 'bg-emerald-600'
              }`}
            />
          </div>
        </div>

        {/* 4. Three Inset Data Blocks (Level 2 Insets) */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
          <div className="md:col-span-4 p-4 sm:p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-1">
            <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
              Observation Timestamp
            </div>
            <div className="font-mono font-bold text-sm sm:text-base text-neutral-950 tabular-nums">
              {selectedScenario === 1 ? '15:33:42 IST / CBT-A01' : '15:28:10 IST / CBT-A01'}
            </div>
          </div>

          <div className="md:col-span-5 p-4 sm:p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-1">
            <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
              Telemetry Model Selected
            </div>
            <div className="font-sans font-semibold text-xs sm:text-sm text-neutral-950 truncate">
              {selectedScenario === 1
                ? 'Multi-HMR 3D Pose + Lap Bounding Box'
                : 'Spatial Wall Angle Homography Suppression'}
            </div>
          </div>

          <div className="md:col-span-3 p-4 sm:p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-1">
            <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500 font-sans">
              Aggregated Risk
            </div>
            <div className={`font-mono font-bold text-lg sm:text-xl tabular-nums ${
              selectedScenario === 1 ? 'text-rose-700' : 'text-emerald-700'
            }`}>
              {selectedScenario === 1 ? '84% (Critical)' : '14% (Nominal)'}
            </div>
          </div>
        </div>

        {/* 5. Four-Step Progression Breakdown Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
          {[
            { num: '1', title: 'RTSP Ingestion', time: '12ms', desc: '4 HD CCTV synchronized' },
            { num: '2', title: 'Pose Kinematics', time: '18ms', desc: '20 skeleton nodes' },
            { num: '3', title: 'Invigilator Aisle', time: '24s', desc: 'Proctor proximity alert' },
            { num: '4', title: 'SHA-256 Seal', time: 'Instant', desc: 'Immutable disciplinary hash' },
          ].map((step, idx) => {
            const isCurrent = idx + 1 === activeStep;
            const isDone = idx + 1 < activeStep;

            return (
              <div
                key={idx}
                onClick={() => {
                  playHapticSound('click');
                  setActiveStep(idx + 1);
                }}
                className={`p-4.5 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between min-h-[96px] ${
                  isCurrent
                    ? 'bg-neutral-100/90 border-neutral-400 shadow-xs'
                    : isDone
                    ? 'bg-emerald-50/50 border-emerald-200'
                    : 'bg-white border-neutral-200/80 hover:border-neutral-300'
                }`}
              >
                <div className="flex items-center justify-between text-2xs">
                  <span className="font-mono font-bold text-neutral-400">Step {step.num}</span>
                  <span className="font-mono font-bold text-neutral-900 bg-white px-2 py-0.5 rounded-md border border-neutral-200 shadow-2xs">
                    {step.time}
                  </span>
                </div>
                <div className="mt-2">
                  <div className="font-heading font-bold text-xs text-neutral-950">{step.title}</div>
                  <div className="text-3xs text-neutral-500 font-sans mt-0.5">{step.desc}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: COHORT DYNAMICS & SIGNAL FREQUENCY (INTEGRATED SUB-DECK) */}
      {/* ========================================================================= */}
      <div className="pt-7 border-t border-neutral-200/80 space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h4 className="font-heading font-extrabold text-lg text-neutral-950 tracking-tight">
              Cohort Behavioral Distribution & Anomaly Frequency
            </h4>
            <p className="text-xs text-neutral-500 mt-0.5">
              Continuous 20-terminal posture synthesis and 45-minute incident rate tracking.
            </p>
          </div>
          <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-neutral-100 text-neutral-700 border border-neutral-200">
            20 Terminals Monitored
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Sub-Card 1: Behavioral Distribution Bar & Breakdown */}
          <div className="p-6 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-4">
            <div className="flex items-center justify-between text-xs font-semibold text-neutral-950">
              <div className="flex items-center gap-2">
                <PieChart size={15} className="text-neutral-700" />
                <span>Cohort Posture Classification</span>
              </div>
              <span className="text-3xs font-mono text-neutral-500">100% Normalized</span>
            </div>

            {/* Stacked Bar */}
            <div className="h-3.5 w-full rounded-full overflow-hidden flex bg-white border border-neutral-200 p-0.5 shadow-inner">
              <div style={{ width: '80%' }} className="bg-emerald-400 h-full rounded-l-full" title="Normal: 80%" />
              <div style={{ width: '10%' }} className="bg-violet-400 h-full" title="Observed: 10%" />
              <div style={{ width: '5%' }} className="bg-amber-400 h-full" title="Needs Review: 5%" />
              <div style={{ width: '5%' }} className="bg-rose-500 h-full rounded-r-full" title="Critical: 5%" />
            </div>

            {/* Breakdown Legend */}
            <div className="grid grid-cols-2 gap-2 text-xs font-sans pt-1">
              {behaviorCategories.map((cat, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 rounded-xl bg-white border border-neutral-200/60 shadow-2xs">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${cat.color}`} />
                    <span className="text-neutral-700 text-2xs font-medium">{cat.label}</span>
                  </div>
                  <strong className="text-neutral-950 font-mono text-2xs">{cat.pct}%</strong>
                </div>
              ))}
            </div>
          </div>

          {/* Sub-Card 2: Activity Timeline Histogram */}
          <div className="p-6 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-3 flex flex-col justify-between">
            <div className="flex items-center justify-between text-xs font-semibold text-neutral-950">
              <div className="flex items-center gap-2">
                <TrendingUp size={15} className="text-neutral-700" />
                <span>Rolling 45-Min Incident Histogram</span>
              </div>
              <span className="text-3xs font-mono text-emerald-800 font-bold px-2 py-0.5 bg-emerald-50 rounded-md border border-emerald-200">
                Mean: 0.11 sig/min
              </span>
            </div>

            {/* Histogram Bars */}
            <div className="h-28 flex items-end justify-between gap-3 px-3 pt-2 pb-1 bg-white rounded-xl border border-neutral-200 shadow-inner">
              {timeBuckets.map((bucket, idx) => {
                const height = Math.max(18, bucket.events * 28);
                return (
                  <div key={idx} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end group">
                    <span className="text-3xs font-mono text-neutral-500 font-bold">
                      {bucket.events}
                    </span>
                    <div
                      style={{ height: `${height}%` }}
                      className={`w-full rounded-t-md transition-all duration-300 ${
                        bucket.events > 2
                          ? 'bg-rose-400'
                          : bucket.events === 2
                          ? 'bg-amber-400'
                          : 'bg-emerald-300'
                      }`}
                    />
                    <span className="text-3xs font-mono text-neutral-400">{bucket.time}</span>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center justify-between text-3xs font-mono text-neutral-500 pt-1">
              <span>Time Horizon: 15:00 - 15:40 IST</span>
              <span className="text-emerald-700 font-bold flex items-center gap-1">
                <ShieldCheck size={12} /> Grid Healthy
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
