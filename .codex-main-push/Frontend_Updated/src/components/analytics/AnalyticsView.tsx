import React from 'react';
import { BarChart3, PieChart, Activity, ShieldCheck, TrendingUp, Zap, Camera, Compass, Volume2, Search, Cpu, Sparkles } from 'lucide-react';
import { MetricValue } from '../ui/MetricValue';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { LiveTelemetryTemporalGraphs } from '../telemetry/LiveTelemetryTemporalGraphs';
import { SpatialHallDigitalTwin } from '../digital-twin/SpatialHallDigitalTwin';
import { GazeAttentionHeatmap } from './GazeAttentionHeatmap';
import { SurveillanceAIQueryConsole } from '../telemetry/SurveillanceAIQueryConsole';
import { InvigilatorAislePatrolAssistant } from '../telemetry/InvigilatorAislePatrolAssistant';
import { ExamCohortPostureBiometricsRadar } from '../telemetry/ExamCohortPostureBiometricsRadar';
import { LiveAIAmbientSoundscapeMonitor } from '../telemetry/LiveAIAmbientSoundscapeMonitor';
import { IllustratedIcon } from '../ui/IllustratedIcon';
import { StatusBadge } from '../ui/StatusBadge';
import { SEATS_DATA } from '../../data/demoData';
import { SeatNodeData } from '../../types';

interface AnalyticsViewProps {
  seats?: SeatNodeData[];
  onSelectCandidate?: (seatId: string) => void;
}

export const AnalyticsView: React.FC<AnalyticsViewProps> = ({
  seats = SEATS_DATA,
  onSelectCandidate,
}) => {
  const outcomes = [
    { label: 'Marked Normal (False Positive)', count: 6, pct: 60, color: 'emerald' as const },
    { label: 'Confirmed Violations', count: 1, pct: 10, color: 'rose' as const },
    { label: 'Needs Follow-up (Observed)', count: 2, pct: 20, color: 'amber' as const },
    { label: 'Pending Human Decision', count: 1, pct: 10, color: 'lavender' as const },
  ];

  const cameraQuality = [
    { id: 'CAM-01', name: 'Lab North Left', tracking: 'OPTIMAL', pose: 'OPTIMAL', object: 'OPTIMAL', overall: 'OPTIMAL' },
    { id: 'CAM-02', name: 'Lab North Right', tracking: 'OPTIMAL', pose: 'OPTIMAL', object: 'MODERATE', overall: 'OPTIMAL' },
    { id: 'CAM-03', name: 'Lab South Left', tracking: 'OPTIMAL', pose: 'MODERATE', object: 'OPTIMAL', overall: 'OPTIMAL' },
    { id: 'CAM-04', name: 'Lab South Right', tracking: 'OPTIMAL', pose: 'OPTIMAL', object: 'MODERATE', overall: 'OPTIMAL' },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      {/* 1. STANDARDIZED PAGE HEADER */}
      <div className="pb-6 border-b border-neutral-200/80">
        <div className="flex items-center gap-2.5 text-xs font-mono uppercase tracking-widest text-neutral-500 mb-2">
          <IllustratedIcon icon={Activity} variant="sky" size="sm" />
          <span>REAL-TIME MULTI-MODAL TELEMETRY • SESSION CBT-A01</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl sm:text-4xl font-extrabold font-heading text-neutral-950 tracking-tight">
              Cohort Telemetry & Kinematic Intelligence
            </h1>
            <p className="text-neutral-600 text-sm mt-1.5 max-w-2xl leading-relaxed">
              Longitudinal behavioral trails, 3D LiDAR digital twin, spatial gaze attention convergence, proctor patrol radar, and ambient acoustic monitoring.
            </p>
          </div>
          <span className="text-xs font-mono font-bold text-neutral-900 px-3.5 py-1.5 rounded-full bg-white border border-neutral-200 shadow-2xs">
            180 MIN SESSION
          </span>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MASTER SUITE 1: AI SURVEILLANCE QUERY ENGINE & INTEGRATED KPI INSETS */}
      {/* ========================================================================= */}
      <div className="p-8 sm:p-9 rounded-[2.5rem] bg-white border border-neutral-200/90 shadow-[0_24px_50px_rgba(0,0,0,0.04)] space-y-7">
        <SurveillanceAIQueryConsole onSelectCandidate={onSelectCandidate} />

        {/* 4 Inset KPI Cards with Illustrated Icons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-1">
          <div className="p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-3xs font-mono font-bold uppercase tracking-wider text-neutral-500">
                Flagged Signals
              </div>
              <div className="font-heading font-extrabold text-2xl sm:text-3xl text-neutral-950 font-mono">
                10
              </div>
              <div className="text-3xs text-neutral-500 font-mono">0.055 events / cand-hr</div>
            </div>
            <IllustratedIcon icon={Zap} variant="neutral" size="md" />
          </div>

          <div className="p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-3xs font-mono font-bold uppercase tracking-wider text-neutral-500">
                Risk Suppression
              </div>
              <div className="font-heading font-extrabold text-2xl sm:text-3xl text-emerald-700 font-mono">
                60%
              </div>
              <div className="text-3xs text-emerald-800 font-mono">6 spatial mask filtered</div>
            </div>
            <IllustratedIcon icon={ShieldCheck} variant="emerald" size="md" />
          </div>

          <div className="p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-3xs font-mono font-bold uppercase tracking-wider text-neutral-500">
                Average Triage
              </div>
              <div className="font-heading font-extrabold text-2xl sm:text-3xl text-neutral-950 font-mono">
                24s
              </div>
              <div className="text-3xs text-neutral-500 font-mono">Under 30s SLA target</div>
            </div>
            <IllustratedIcon icon={Activity} variant="amber" size="md" />
          </div>

          <div className="p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-3xs font-mono font-bold uppercase tracking-wider text-neutral-500">
                Confirmed Rate
              </div>
              <div className="font-heading font-extrabold text-2xl sm:text-3xl text-rose-700 font-mono">
                10%
              </div>
              <div className="text-3xs text-rose-800 font-mono">1 confirmed case (A06)</div>
            </div>
            <IllustratedIcon icon={Sparkles} variant="rose" size="md" />
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MASTER SUITE 2: SPATIAL 3D DIGITAL TWIN & AISLE PATROL RADAR SUITE */}
      {/* ========================================================================= */}
      <div className="space-y-8">
        <SpatialHallDigitalTwin seats={seats} onSelectSeat={onSelectCandidate} />
        <InvigilatorAislePatrolAssistant seats={seats} onSelectCandidate={onSelectCandidate} />
      </div>

      {/* ========================================================================= */}
      {/* MASTER SUITE 3: MULTI-MODAL BIOMETRIC & ACOUSTIC TELEMETRY DECK */}
      {/* ========================================================================= */}
      <div className="space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <ExamCohortPostureBiometricsRadar />
          <LiveAIAmbientSoundscapeMonitor />
        </div>

        <GazeAttentionHeatmap />
        <LiveTelemetryTemporalGraphs />
      </div>

      {/* ========================================================================= */}
      {/* MASTER SUITE 4: REVIEW DISPOSITION & CAMERA QUALITY MATRIX */}
      {/* ========================================================================= */}
      <div className="p-8 sm:p-9 rounded-[2.5rem] bg-white border border-neutral-200/90 shadow-[0_24px_50px_rgba(0,0,0,0.04)] space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-200/80">
          <div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Ingestion Quality & Human Review Disposition
            </h3>
            <p className="text-xs text-neutral-500 mt-0.5">
              Multi-camera RTSP ingestion health and invigilator decision breakdown.
            </p>
          </div>
          <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
            4 / 4 FEEDS OPTIMAL
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Review Outcome Distribution */}
          <div className="p-6 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-4">
            <div className="font-heading font-bold text-sm text-neutral-950 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <PieChart size={16} className="text-neutral-700" />
                <span>Review Outcome Disposition</span>
              </div>
              <span className="text-3xs font-mono text-neutral-500 font-bold">10 Actions Logged</span>
            </div>

            <div className="space-y-3">
              {outcomes.map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-xs font-sans">
                    <span className="text-neutral-700 font-medium">{item.label}</span>
                    <span className="text-neutral-950 font-bold font-mono text-2xs">{item.count} ({item.pct}%)</span>
                  </div>
                  <ConfidenceBar
                    value={item.pct / 100}
                    showPercent={false}
                    color={item.color}
                    size="sm"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Camera Quality & Vision Ingestion Health */}
          <div className="p-6 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-3">
            <div className="font-heading font-bold text-sm text-neutral-950 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Camera size={16} className="text-neutral-700" />
                <span>Camera Vision Quality Matrix</span>
              </div>
              <span className="text-3xs font-mono text-emerald-800 font-bold px-2 py-0.5 bg-white rounded-md border border-neutral-200 shadow-2xs">
                All Optimal
              </span>
            </div>

            <div className="space-y-2.5 text-xs font-mono">
              {cameraQuality.map((cam) => (
                <div
                  key={cam.id}
                  className="p-3 rounded-xl border border-neutral-200 bg-white shadow-2xs flex items-center justify-between"
                >
                  <div>
                    <div className="font-bold text-neutral-950">{cam.id}</div>
                    <div className="text-3xs text-neutral-500 font-sans">{cam.name}</div>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="text-3xs bg-neutral-50 px-2 py-0.5 rounded-md border border-neutral-200 text-neutral-600 font-mono">
                      Pose: <strong>{cam.pose}</strong>
                    </div>
                    <StatusBadge status={cam.overall as any} size="sm" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
