import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SeatNodeData, TriagedAIEvent } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import {
  ArrowRight,
  Clock,
  CheckCircle2,
  X,
  Radio,
  User,
  Shield,
  Smartphone,
  Eye,
  Hand,
  Activity,
  Maximize2,
  AlertTriangle,
  Compass,
  Sparkles,
  Layers,
  ChevronRight,
  Camera,
  Send,
  Zap,
  Video
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { IllustratedIcon } from '../ui/IllustratedIcon';

interface CommandCenterInspectorPanelProps {
  selectedSeat: SeatNodeData | null;
  onClearSelectedSeat: () => void;
  events: TriagedAIEvent[];
  onReviewEvent: (event: TriagedAIEvent) => void;
  onViewCandidateDossier?: (seatId: string) => void;
}

export const CommandCenterInspectorPanel: React.FC<CommandCenterInspectorPanelProps> = ({
  selectedSeat,
  onClearSelectedSeat,
  events,
  onReviewEvent,
  onViewCandidateDossier,
}) => {
  const [filter, setFilter] = useState<'ALL' | 'escalated' | 'review' | 'observe'>('ALL');
  const [dispatched, setDispatched] = useState<boolean>(false);

  // Filtered Events
  const filteredEvents = events.filter((e) => {
    if (filter === 'ALL') return true;
    return e.severity === filter;
  });

  const selectedSeatEvent = selectedSeat
    ? events.find((e) => e.seatId === selectedSeat.id)
    : null;

  const isCritical = selectedSeat?.state === 'escalated' || selectedSeat?.state === 'review';

  return (
    <div className={`p-6 sm:p-7 rounded-3xl border transition-all duration-300 font-sans h-full flex flex-col justify-between space-y-5 ${
      selectedSeat && isCritical
        ? 'bg-[#181716] text-[#FAF7F3] border-neutral-800 shadow-md'
        : 'bg-neutral-50/70 text-neutral-900 border-neutral-200/80'
    }`}>
      {/* ========================================================================= */}
      {/* CASE A: A SPECIFIC CANDIDATE / SEAT IS SELECTED ON THE MAP */}
      {/* ========================================================================= */}
      {selectedSeat ? (
        <div className="space-y-5 flex-1 flex flex-col justify-between">
          {/* Header */}
          <div className={`flex items-center justify-between pb-3.5 border-b ${isCritical ? 'border-white/10' : 'border-neutral-200'}`}>
            <div className="flex items-center gap-3">
              <div className={`w-11 h-11 rounded-2xl font-mono font-extrabold text-base flex items-center justify-center shadow-sm shrink-0 ${
                isCritical ? 'bg-white text-neutral-950' : 'bg-neutral-900 text-white'
              }`}>
                {selectedSeat.id}
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-heading font-extrabold text-base">
                    Candidate #{selectedSeat.candidateId}
                  </span>
                  <StatusBadge status={selectedSeat.state} size="sm" />
                </div>
                <div className={`text-3xs mt-0.5 font-mono ${isCritical ? 'text-neutral-400' : 'text-neutral-500'}`}>
                  {selectedSeat.gridPosition.row === 'A' ? 'North Desk' : 'South Desk'} • Station {selectedSeat.id}
                </div>
              </div>
            </div>

            <button
              onClick={() => {
                playHapticSound('click');
                onClearSelectedSeat();
              }}
              className={`p-2 rounded-xl transition-colors cursor-pointer shrink-0 ${
                isCritical ? 'text-neutral-400 hover:text-white hover:bg-white/10' : 'text-neutral-400 hover:text-neutral-900 hover:bg-white'
              }`}
              title="Return to Live AI Stream"
            >
              <X size={16} />
            </button>
          </div>

          {/* Real-time Kinematic Telemetry Stack */}
          <div className="space-y-4">
            <div className="flex items-center justify-between text-3xs font-mono">
              <span className={`font-bold uppercase tracking-wider ${isCritical ? 'text-amber-300' : 'text-neutral-500'}`}>
                {isCritical ? 'CRITICAL SIGNAL INTERCEPT' : 'KINEMATIC POSE TELEMETRY'}
              </span>
              <span className={isCritical ? 'text-neutral-400' : 'text-neutral-400'}>
                100ms DAG
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2.5 text-xs">
              <div className={`p-3 rounded-xl border space-y-0.5 ${
                isCritical ? 'bg-white/5 border-white/10' : 'bg-white border-neutral-200'
              }`}>
                <span className={`text-3xs font-mono ${isCritical ? 'text-neutral-400' : 'text-neutral-500'}`}>Continuity</span>
                <div className="font-mono font-bold text-emerald-400 text-sm">{selectedSeat.telemetry.trackContinuity}</div>
              </div>

              <div className={`p-3 rounded-xl border space-y-0.5 ${
                isCritical ? 'bg-white/5 border-white/10' : 'bg-white border-neutral-200'
              }`}>
                <span className={`text-3xs font-mono ${isCritical ? 'text-neutral-400' : 'text-neutral-500'}`}>Gaze Vector</span>
                <div className="font-mono font-bold text-sm">{selectedSeat.telemetry.headEvidence}</div>
              </div>

              <div className={`p-3 rounded-xl border space-y-0.5 ${
                isCritical ? 'bg-white/5 border-white/10' : 'bg-white border-neutral-200'
              }`}>
                <span className={`text-3xs font-mono ${isCritical ? 'text-neutral-400' : 'text-neutral-500'}`}>Laptop Hands</span>
                <div className="font-mono font-bold text-sm">{selectedSeat.telemetry.handsEvidence}</div>
              </div>

              <div className={`p-3 rounded-xl border space-y-0.5 ${
                isCritical ? 'bg-white/5 border-white/10' : 'bg-white border-neutral-200'
              }`}>
                <span className={`text-3xs font-mono ${isCritical ? 'text-neutral-400' : 'text-neutral-500'}`}>RTSP Cameras</span>
                <div className="font-mono font-bold text-sm">{selectedSeat.cameraIds.join(', ')}</div>
              </div>
            </div>

            {/* Event Summary Box */}
            <div className={`p-4 rounded-2xl border space-y-1.5 ${
              isCritical ? 'bg-white/5 border-white/10' : 'bg-white border-neutral-200'
            }`}>
              <div className="flex items-center gap-1.5 text-3xs font-mono uppercase font-bold tracking-wider">
                <Activity size={12} className={isCritical ? 'text-amber-400' : 'text-neutral-700'} />
                <span className={isCritical ? 'text-amber-300' : 'text-neutral-600'}>Last Observation Trigger</span>
              </div>
              <p className={`text-xs leading-relaxed font-sans font-medium ${isCritical ? 'text-neutral-200' : 'text-neutral-800'}`}>
                {selectedSeat.telemetry.lastEventSummary}
              </p>
              <div className={`text-3xs font-mono pt-1.5 border-t ${
                isCritical ? 'border-white/10 text-neutral-400' : 'border-neutral-200 text-neutral-400'
              }`}>
                Timestamp: {selectedSeat.telemetry.lastEventTimeAgo}
              </div>
            </div>

            {/* Live Camera Feed Preview Box */}
            <div className={`p-4 rounded-2xl border space-y-2 ${
              isCritical ? 'bg-white/5 border-white/10' : 'bg-white border-neutral-200'
            }`}>
              <div className="flex items-center justify-between text-3xs font-mono">
                <span className="flex items-center gap-1.5 font-bold">
                  <Video size={12} />
                  <span>SYNCHRONIZED FEED LINK</span>
                </span>
                <span className="text-emerald-400 font-bold">30 FPS • HD</span>
              </div>
              <div className="text-2xs text-neutral-400 font-mono">
                Assigned Feeds: {selectedSeat.cameraIds.join(' + ')} • Active Spatial Bounding Box Locked
              </div>
            </div>
          </div>

          {/* Actions Bar */}
          <div className={`pt-3 border-t space-y-2 ${isCritical ? 'border-white/10' : 'border-neutral-200'}`}>
            {selectedSeatEvent ? (
              <Button
                variant="default"
                size="default"
                onClick={() => {
                  playHapticSound('snap');
                  onReviewEvent(selectedSeatEvent);
                }}
                className={`w-full gap-2 font-semibold rounded-xl shadow-xs py-3 text-xs ${
                  isCritical ? 'bg-white text-neutral-950 hover:bg-neutral-100 border-white' : ''
                }`}
              >
                <span>Launch Forensic Review</span>
                <ArrowRight size={14} />
              </Button>
            ) : (
              <Button
                variant="outline"
                size="default"
                onClick={() => {
                  playHapticSound('snap');
                  if (onViewCandidateDossier) {
                    onViewCandidateDossier(selectedSeat.id);
                  }
                }}
                className="w-full gap-2 font-semibold rounded-xl py-3 text-xs bg-white"
              >
                <User size={14} />
                <span>View Student Dossier</span>
              </Button>
            )}

            <Button
              variant="ghost"
              size="xs"
              onClick={() => {
                playHapticSound('click');
                onClearSelectedSeat();
              }}
              className={`w-full text-3xs ${isCritical ? 'text-neutral-400 hover:text-white hover:bg-white/5' : 'text-neutral-500'}`}
            >
              ← Back to Live AI Stream
            </Button>
          </div>
        </div>
      ) : (
        /* ========================================================================= */
        /* CASE B: DEFAULT LIVE AI EVENT STREAM & MULTI-CAM DISPATCH DECK */
        /* ========================================================================= */
        <div className="space-y-4 flex-1 flex flex-col justify-between">
          {/* Top Section: Header + Filter Pills */}
          <div className="space-y-3.5">
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-neutral-200 gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse shrink-0 shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
                  <span className="font-heading font-extrabold text-sm text-neutral-950">
                    Live AI Event Stream
                  </span>
                </div>
                <div className="text-2xs text-neutral-500 font-sans mt-0.5">
                  Surveillance multi-camera ingestion cues
                </div>
              </div>
              <span className="text-3xs font-mono font-bold text-sky-800 px-2.5 py-1 rounded-full bg-sky-50 border border-sky-200 shrink-0 whitespace-nowrap shadow-2xs">
                {events.length} Active Feeds
              </span>
            </div>

            {/* EQUALLY BALANCED SLIDER FILTER TABS */}
            <div className="flex items-center justify-between p-1 bg-white rounded-2xl border border-neutral-200 shadow-2xs relative">
              {[
                { id: 'ALL', label: 'All' },
                { id: 'escalated', label: 'Critical' },
                { id: 'review', label: 'Review' },
                { id: 'observe', label: 'Observe' }
              ].map((tab) => {
                const isActive = filter === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => {
                      playHapticSound('click');
                      setFilter(tab.id as any);
                    }}
                    className={`relative flex-1 py-1.5 px-2 rounded-xl text-3xs font-mono font-bold transition-colors cursor-pointer text-center select-none z-10 ${
                      isActive ? 'text-white' : 'text-neutral-500 hover:text-neutral-900'
                    }`}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="inspectorFilterSlider"
                        transition={{ type: 'spring', damping: 28, stiffness: 420 }}
                        className="absolute inset-0 bg-[#181716] rounded-xl shadow-xs -z-10"
                      />
                    )}
                    <span className="relative z-10 whitespace-nowrap">{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Middle Section: Scrollable Event Stream Cards */}
          <div className="space-y-2.5 overflow-y-auto max-h-[310px] pr-1 pb-1">
            <AnimatePresence mode="popLayout">
              {filteredEvents.map((event) => {
                const isEscalated = event.severity === 'escalated';
                const isReview = event.severity === 'review';

                return (
                  <motion.div
                    key={event.id}
                    layout
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    whileHover={{ y: -2, transition: { type: 'spring', damping: 20, stiffness: 350 } }}
                    className={`p-3.5 rounded-2xl border transition-all shadow-xs ${
                      isEscalated
                        ? 'bg-rose-50/85 border-rose-200'
                        : isReview
                        ? 'bg-amber-50/85 border-amber-200'
                        : 'bg-violet-50/75 border-violet-200'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-1.5 gap-2">
                      <div className="flex items-center gap-1.5 font-mono text-neutral-500 text-3xs truncate">
                        <Clock size={10} className="shrink-0 text-neutral-400" />
                        <span>{event.timestamp}</span>
                        <span>•</span>
                        <strong className="text-neutral-950 font-bold">Seat {event.seatId}</strong>
                      </div>
                      <StatusBadge status={event.severity} size="sm" />
                    </div>

                    <div className="font-heading font-bold text-xs text-neutral-950 mb-2 leading-snug font-sans">
                      {event.behaviorDisplayTitle}
                    </div>

                    <div className="flex items-center justify-between pt-2 border-t border-black/[0.06] text-xs">
                      <span className="text-3xs font-semibold text-neutral-600 font-sans">
                        Candidate #{event.candidateId}
                      </span>
                      <button
                        onClick={() => {
                          playHapticSound('snap');
                          onReviewEvent(event);
                        }}
                        className="flex items-center gap-1.5 bg-[#181716] hover:bg-black text-white px-2.5 py-1 rounded-lg text-3xs font-bold font-mono transition-all shadow-xs cursor-pointer active:scale-95"
                      >
                        <span>Inspect</span>
                        <ChevronRight size={11} />
                      </button>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>

          {/* ========================================================================= */}
          {/* BOTTOM SECTION: 4-CAMERA RTSP STREAM MATRIX & QUICK DISPATCH (BALANCED) */}
          {/* ========================================================================= */}
          <div className="space-y-3 pt-2">
            <div className="p-4 rounded-2xl bg-white border border-neutral-200/90 shadow-2xs space-y-3">
              {/* Mini Header */}
              <div className="flex items-center justify-between text-3xs font-mono">
                <span className="flex items-center gap-1.5 font-bold text-neutral-900">
                  <Camera size={13} className="text-neutral-700" />
                  <span>4-CAMERA INGESTION STATUS</span>
                </span>
                <span className="text-emerald-700 font-bold bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">
                  30 FPS • Sync
                </span>
              </div>

              {/* 4 Camera Feeds 2x2 Mini Matrix */}
              <div className="grid grid-cols-2 gap-2 text-3xs font-mono">
                <div className="p-2 rounded-xl bg-neutral-50 border border-neutral-200/80 flex items-center justify-between">
                  <span className="font-bold text-neutral-800">CAM-01 (North L)</span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                </div>
                <div className="p-2 rounded-xl bg-neutral-50 border border-neutral-200/80 flex items-center justify-between">
                  <span className="font-bold text-neutral-800">CAM-02 (North R)</span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                </div>
                <div className="p-2 rounded-xl bg-neutral-50 border border-neutral-200/80 flex items-center justify-between">
                  <span className="font-bold text-neutral-800">CAM-03 (South L)</span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                </div>
                <div className="p-2 rounded-xl bg-neutral-50 border border-neutral-200/80 flex items-center justify-between">
                  <span className="font-bold text-neutral-800">CAM-04 (South R)</span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                </div>
              </div>

              {/* Quick Proctor Intercept Dispatch Button */}
              <button
                onClick={() => {
                  playHapticSound('snap');
                  setDispatched(true);
                  setTimeout(() => setDispatched(false), 3000);
                }}
                className={`w-full py-2.5 px-3 rounded-xl text-3xs font-mono font-bold transition-all flex items-center justify-center gap-2 cursor-pointer shadow-xs ${
                  dispatched
                    ? 'bg-emerald-700 text-white'
                    : 'bg-[#181716] hover:bg-black text-white active:scale-98'
                }`}
              >
                {dispatched ? (
                  <>
                    <CheckCircle2 size={13} className="text-emerald-300" />
                    <span>PROCTOR DISPATCHED TO SEAT A06 (1.8m AISLE)</span>
                  </>
                ) : (
                  <>
                    <Zap size={13} className="text-amber-400" />
                    <span>DISPATCH PROCTOR TO SEAT A06 (CRITICAL CUE)</span>
                  </>
                )}
              </button>
            </div>

            {/* Bottom Guide Notice */}
            <div className="p-2.5 bg-white/80 rounded-xl border border-neutral-200/70 text-3xs text-neutral-500 text-center font-sans">
              💡 Select any terminal on the lab grid to inspect real-time posture telemetry.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
