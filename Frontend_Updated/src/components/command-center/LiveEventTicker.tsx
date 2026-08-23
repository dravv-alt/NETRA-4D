import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { TriagedAIEvent, EventSeverity } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { ArrowRight, Clock, CheckCircle2, Radio } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';

interface LiveEventTickerProps {
  events: TriagedAIEvent[];
  onReviewEvent: (event: TriagedAIEvent) => void;
}

export const LiveEventTicker: React.FC<LiveEventTickerProps> = ({
  events,
  onReviewEvent,
}) => {
  const [filter, setFilter] = useState<'ALL' | EventSeverity>('ALL');

  const filteredEvents = events.filter((e) => {
    if (filter === 'ALL') return true;
    return e.severity === filter;
  });

  return (
    <div className="bg-white p-6 flex flex-col h-full rounded-2xl border border-neutral-200 shadow-xs font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
        <div className="flex items-center gap-2.5">
          <div className="p-1 rounded-full bg-rose-50 border border-rose-200">
            <span className="block w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
          </div>
          <span className="font-semibold text-base text-neutral-950">
            Live AI Event Stream
          </span>
        </div>
        <span className="text-xs font-semibold text-sky-800 px-2.5 py-0.5 rounded-full bg-sky-50 border border-sky-200">
          {events.length} Feeds
        </span>
      </div>

      {/* Filter Tabs in Pastel */}
      <div className="flex items-center gap-2 py-3 border-b border-neutral-200 text-xs">
        {(['ALL', 'escalated', 'review', 'observe'] as const).map((tab) => {
          const isActive = filter === tab;
          const label = tab === 'ALL' ? 'All' : tab.toUpperCase();
          return (
            <button
              key={tab}
              onClick={() => {
                playHapticSound('click');
                setFilter(tab);
              }}
              className={`px-3 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
                isActive
                  ? 'bg-neutral-900 text-white font-semibold shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100'
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Events List */}
      <div className="space-y-3.5 overflow-y-auto flex-1 my-2 pr-1 max-h-[500px]">
        <AnimatePresence>
          {filteredEvents.map((event) => {
            const isEscalated = event.severity === 'escalated';
            const isReview = event.severity === 'review';

            return (
              <motion.div
                key={event.id}
                layout
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96 }}
                whileHover={{ y: -1 }}
                transition={{ type: 'spring', damping: 28, stiffness: 350 }}
                className={`p-4 rounded-xl border transition-colors ${
                  isEscalated
                    ? 'bg-rose-50/40 border-rose-200 hover:border-rose-300'
                    : isReview
                    ? 'bg-amber-50/40 border-amber-200 hover:border-amber-300'
                    : 'bg-violet-50/30 border-violet-200 hover:border-violet-300'
                }`}
              >
                {/* Card Header */}
                <div className="flex items-center justify-between text-xs mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-neutral-400 flex items-center gap-1">
                      <Clock size={12} /> {event.timestamp}
                    </span>
                    <span className="text-neutral-300">•</span>
                    <span className="font-semibold text-neutral-950">Seat {event.seatId}</span>
                  </div>
                  <StatusBadge status={event.severity} size="sm" />
                </div>

                {/* Behavior Title */}
                <div className="font-semibold text-sm text-neutral-950 mb-2.5 tracking-tight leading-snug">
                  {event.behaviorDisplayTitle}
                </div>

                {/* AI Confidence & Metrics */}
                <div className="mb-3 space-y-2">
                  <ConfidenceBar
                    value={event.aiConfidence}
                    label="Model Confidence"
                    color={isEscalated ? 'rose' : isReview ? 'amber' : 'lavender'}
                    size="sm"
                  />
                  <div className="flex items-center justify-between text-xs text-neutral-500 pt-1">
                    <span>Persistence: <strong className="text-neutral-900 font-semibold">{event.persistenceSec}s</strong></span>
                    <span>Priority: <strong className="text-neutral-900 font-semibold">{event.reviewPriority}</strong></span>
                  </div>
                </div>

                {/* Counter Evidence Tag in Pastel Mint */}
                {event.counterEvidence.riskSuppressed && (
                  <div className="text-xs bg-emerald-50 border border-emerald-200 text-emerald-800 p-2.5 rounded-lg mb-2.5 flex items-center gap-2">
                    <CheckCircle2 size={14} className="shrink-0 text-emerald-600" />
                    <span className="truncate">{event.counterEvidence.suppressionReason}</span>
                  </div>
                )}

                {/* Action Button */}
                <div className="flex items-center justify-between pt-2.5 border-t border-black/[0.06]">
                  <span className="text-xs text-neutral-600 font-medium truncate">
                    Candidate #{event.candidateId}
                  </span>
                  <Button
                    variant="default"
                    size="xs"
                    onClick={() => {
                      playHapticSound('snap');
                      onReviewEvent(event);
                    }}
                    className="gap-1.5"
                  >
                    <span>Inspect</span>
                    <ArrowRight size={12} />
                  </Button>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
};
