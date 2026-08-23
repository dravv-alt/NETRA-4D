import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { TriagedAIEvent } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import {
  Video,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  UserCheck,
  Layers,
  Compass,
  ArrowLeft,
  ChevronDown,
  Sparkles,
  LayoutGrid
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { MultiCamForensicMatrix } from './MultiCamForensicMatrix';
import { ForensicAdjudicationMatrixMaster } from './ForensicAdjudicationMatrixMaster';

interface EvidenceReviewViewProps {
  event: TriagedAIEvent;
  onDecision: (
    eventId: string,
    decision: 'marked_normal' | 'needs_follow_up' | 'confirmed_violation' | 'escalated',
    note?: string
  ) => void;
  onBackToQueue?: () => void;
}

export const EvidenceReviewView: React.FC<EvidenceReviewViewProps> = ({
  event,
  onDecision,
  onBackToQueue,
}) => {
  const [hoveredTime, setHoveredTime] = useState<string | null>(null);
  const [reviewerNote, setReviewerNote] = useState<string>(event.humanDecision?.notes || '');
  const [currentDecision, setCurrentDecision] = useState<string | undefined>(event.humanDecision?.status);
  const [expandedStep, setExpandedStep] = useState<string | null>('ev-1');
  const [viewTab, setViewTab] = useState<'single' | 'multicam'>('single');

  const handleApplyDecision = (status: 'marked_normal' | 'needs_follow_up' | 'confirmed_violation' | 'escalated') => {
    playHapticSound('snap');
    setCurrentDecision(status);
    onDecision(event.id, status, reviewerNote);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      {/* 1. STANDARDIZED PAGE HEADER */}
      <div className="pb-6 border-b border-neutral-200/80">
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-neutral-500 mb-2">
          <ShieldCheck size={13} className="text-neutral-700" />
          <span>FORENSIC INCIDENT INVESTIGATION • EVIDENCE STACK</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            {onBackToQueue && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  playHapticSound('click');
                  onBackToQueue();
                }}
                className="gap-2 rounded-xl shadow-2xs font-semibold"
              >
                <ArrowLeft size={14} />
                <span>Back to Queue</span>
              </Button>
            )}
            <div>
              <h1 className="text-3xl sm:text-4xl font-extrabold font-heading text-neutral-950 tracking-tight">
                Review Event {event.id}
              </h1>
              <p className="text-neutral-600 text-sm mt-1.5 max-w-2xl leading-relaxed">
                Candidate #{event.candidateId} • Terminal {event.seatId} • Flagged at {event.timestamp} IST • Risk Priority: <strong className="text-neutral-950 font-mono">{event.reviewPriority}</strong>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                playHapticSound('click');
                setViewTab('single');
              }}
              className={`px-3.5 py-2 rounded-xl text-xs font-mono font-semibold transition-all cursor-pointer ${
                viewTab === 'single'
                  ? 'bg-neutral-900 text-white shadow-xs'
                  : 'bg-white border border-neutral-200 text-neutral-600 hover:bg-neutral-50'
              }`}
            >
              Primary Camera Angle
            </button>
            <button
              onClick={() => {
                playHapticSound('click');
                setViewTab('multicam');
              }}
              className={`px-3.5 py-2 rounded-xl text-xs font-mono font-semibold transition-all cursor-pointer ${
                viewTab === 'multicam'
                  ? 'bg-neutral-900 text-white shadow-xs'
                  : 'bg-white border border-neutral-200 text-neutral-600 hover:bg-neutral-50'
              }`}
            >
              4-Camera Forensic Matrix
            </button>
          </div>
        </div>
      </div>

      {/* 2. 4-CAMERA MATRIX (IF SELECTED) */}
      {viewTab === 'multicam' && <MultiCamForensicMatrix />}

      {/* 3. FOUR-QUADRANT MASTER GROUPING SYSTEM (REPLICATING REFERENCE ARCHITECTURE) */}
      <ForensicAdjudicationMatrixMaster
        onConfirmViolation={() => handleApplyDecision('confirmed_violation')}
        onDismissAnomaly={() => handleApplyDecision('marked_normal')}
        onReturnToCockpit={onBackToQueue}
      />

      {/* 4. MAIN 2-COLUMN SPLIT VIEW: REPLAY BUFFER & KINEMATIC TIMELINE */}
      <div className="grid grid-cols-12 gap-8 items-start">
        {/* LEFT 7 COLS: Video Player & Timeline */}
        <div className="col-span-12 lg:col-span-7 space-y-6">
          {/* Synchronized Replay */}
          {viewTab === 'single' && (
            <div className="bg-white rounded-[2rem] border border-neutral-200/80 overflow-hidden shadow-[0_20px_40px_rgba(0,0,0,0.035)]">
              <div className="relative aspect-video bg-neutral-950 flex items-center justify-center">
                <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center">
                  <div className="w-16 h-16 rounded-2xl bg-neutral-900 flex items-center justify-center text-white mb-4 border border-neutral-800 shadow-md">
                    <Video size={28} />
                  </div>
                  <div className="font-heading font-extrabold text-lg text-white tracking-tight">
                    Synchronized Anomaly Replay Buffer
                  </div>
                  <div className="text-xs text-neutral-400 mt-2 max-w-md leading-relaxed font-sans">
                    Showing -15.0s ... Flag Frame ({event.timestamp}) ... +15.0s multi-camera temporal buffer.
                  </div>
                  <div className="mt-4 flex items-center gap-3 text-xs font-mono">
                    <span className="px-3.5 py-1.5 rounded-full bg-neutral-900 text-neutral-200 font-bold border border-neutral-800">
                      Seat {event.seatId} • Candidate #{event.candidateId}
                    </span>
                    <span className="px-3.5 py-1.5 rounded-full bg-neutral-900 text-neutral-200 font-bold border border-neutral-800">
                      Confidence: {(event.aiConfidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="absolute top-4 left-4 bg-black/80 px-3.5 py-1.5 rounded-full text-xs text-white font-mono font-bold border border-white/20">
                  {event.cameraName} • {event.timestamp}.4 IST
                </div>
              </div>

              {/* Video Controls Bar */}
              <div className="p-4 bg-white border-t border-neutral-200 flex items-center justify-between text-xs font-mono">
                <span className="text-neutral-500">Replay Span: -15.0s to +15.0s (30.0s total)</span>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="xs" onClick={() => playHapticSound('click')} className="rounded-lg">
                    Step -1 Frame
                  </Button>
                  <Button variant="outline" size="xs" onClick={() => playHapticSound('click')} className="rounded-lg">
                    Step +1 Frame
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* MULTI-TRACK TEMPORAL TIMELINE */}
          <div className="p-8 space-y-6 bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] rounded-[2rem]">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-200/80">
              <div className="font-heading font-bold text-base text-neutral-950 flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
                  <Layers size={16} />
                </div>
                <span>Multi-Modal Temporal Kinematics</span>
              </div>
              <span className="text-xs font-mono text-neutral-500 font-bold">100ms Ingestion DAG</span>
            </div>

            {/* 5-Track Visualization */}
            <div className="space-y-3.5 text-xs">
              {/* Track 1: Head Gaze */}
              <div className="flex items-center gap-4">
                <span className="w-28 text-neutral-600 font-medium">Head Gaze</span>
                <div
                  onMouseEnter={() => {
                    playHapticSound('click');
                    setHoveredTime('14:32:19.4 | Head yaw: -28° toward lap | Continuity: Optimal');
                  }}
                  className="flex-1 h-3.5 bg-neutral-100 rounded-full border border-neutral-200 overflow-hidden relative cursor-pointer"
                >
                  <div className="absolute left-[30%] w-[25%] h-full bg-amber-400 rounded-full" />
                  <div className="absolute left-[65%] w-[18%] h-full bg-amber-400 rounded-full" />
                </div>
              </div>

              {/* Track 2: Hands on Laptop */}
              <div className="flex items-center gap-4">
                <span className="w-28 text-neutral-600 font-medium">Laptop Hands</span>
                <div
                  onMouseEnter={() => {
                    playHapticSound('click');
                    setHoveredTime('14:32:20.1 | Right wrist off-keyboard | Duration: 6.2s');
                  }}
                  className="flex-1 h-3.5 bg-neutral-100 rounded-full border border-neutral-200 overflow-hidden relative cursor-pointer"
                >
                  <div className="absolute left-[20%] w-[45%] h-full bg-rose-400 rounded-full" />
                </div>
              </div>

              {/* Track 3: Body Pose & Spine */}
              <div className="flex items-center gap-4">
                <span className="w-28 text-neutral-600 font-medium">Body Spine</span>
                <div
                  onMouseEnter={() => {
                    playHapticSound('click');
                    setHoveredTime('14:32:18.0 | Posture: Nominal upright | Slight forward lean');
                  }}
                  className="flex-1 h-3.5 bg-neutral-100 rounded-full border border-neutral-200 overflow-hidden relative cursor-pointer"
                >
                  <div className="absolute left-[0%] w-[100%] h-full bg-emerald-300 rounded-full opacity-60" />
                </div>
              </div>

              {/* Track 4: Object Detect */}
              <div className="flex items-center gap-4">
                <span className="w-28 text-neutral-600 font-medium">Object Detect</span>
                <div
                  onMouseEnter={() => {
                    playHapticSound('click');
                    setHoveredTime('14:32:21.0 | Phone signature in lap area | Confidence: 74%');
                  }}
                  className="flex-1 h-3.5 bg-neutral-100 rounded-full border border-neutral-200 overflow-hidden relative cursor-pointer"
                >
                  <div className="absolute left-[35%] w-[15%] h-full bg-rose-500 rounded-full" />
                </div>
              </div>

              {/* Track 5: Spatial Context Mask */}
              <div className="flex items-center gap-4">
                <span className="w-28 text-neutral-600 font-medium">Spatial Mask</span>
                <div
                  onMouseEnter={() => {
                    playHapticSound('click');
                    setHoveredTime('14:32:15.0 | Desk polygon: Clear | Adjacent distance: 1.2m');
                  }}
                  className="flex-1 h-3.5 bg-neutral-100 rounded-full border border-neutral-200 overflow-hidden relative cursor-pointer"
                >
                  <div className="absolute left-[0%] w-[100%] h-full bg-sky-300 rounded-full opacity-50" />
                </div>
              </div>
            </div>

            {/* Hover HUD */}
            <div className="p-3 bg-neutral-50 rounded-2xl border border-neutral-200 text-2xs font-mono text-neutral-700 min-h-[38px] flex items-center justify-center">
              {hoveredTime ? hoveredTime : 'Hover over any kinematic track to view sub-second vector telemetry.'}
            </div>
          </div>
        </div>

        {/* RIGHT 5 COLS: Explainable Evidence Stack */}
        <div className="col-span-12 lg:col-span-5 space-y-6">
          <div className="p-7 space-y-5 bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] rounded-[2rem]">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-200/80">
              <div className="font-heading font-bold text-base text-neutral-950">
                Explainable Multi-Modal Evidence Stack
              </div>
              <span className="text-2xs font-mono font-bold text-neutral-500">
                {event.evidenceStack.length} Triggers
              </span>
            </div>

            <div className="space-y-3">
              {event.evidenceStack.map((step) => {
                const isExpanded = expandedStep === `ev-${step.stepNumber}`;
                return (
                  <div
                    key={step.stepNumber}
                    className="p-4 rounded-2xl border border-neutral-200/80 bg-neutral-50/60 space-y-2.5 transition-all"
                  >
                    <div
                      onClick={() => {
                        playHapticSound('click');
                        setExpandedStep(isExpanded ? null : `ev-${step.stepNumber}`);
                      }}
                      className="flex items-center justify-between cursor-pointer"
                    >
                      <div className="flex items-center gap-2.5">
                        <span className="w-6 h-6 rounded-full bg-neutral-900 text-white font-mono text-2xs font-bold flex items-center justify-center">
                          {step.stepNumber}
                        </span>
                        <span className="font-heading font-bold text-xs text-neutral-950">
                          {step.title}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-2xs font-mono font-bold text-neutral-600">
                          {(step.confidence * 100).toFixed(0)}%
                        </span>
                        <ChevronDown size={14} className={`text-neutral-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                      </div>
                    </div>

                    {isExpanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="pt-2 border-t border-neutral-200/60 text-xs text-neutral-600 space-y-2 leading-relaxed font-sans"
                      >
                        <p>{step.description}</p>
                        <div className="flex items-center justify-between text-3xs font-mono text-neutral-400">
                          <span>Category: {step.category}</span>
                          <span>Duration: {step.durationSec}s</span>
                        </div>
                      </motion.div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
