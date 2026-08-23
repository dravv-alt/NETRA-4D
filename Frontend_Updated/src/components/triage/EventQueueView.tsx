import React, { useState } from 'react';
import { TriagedAIEvent } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { Filter, Search, ArrowRight, Clock, CheckCircle2, Sparkles, Inbox } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { ParallaxCard } from '../ui/parallax-card';

interface EventQueueViewProps {
  events: TriagedAIEvent[];
  onReviewEvent: (event: TriagedAIEvent) => void;
}

export const EventQueueView: React.FC<EventQueueViewProps> = ({
  events,
  onReviewEvent,
}) => {
  const [selectedSeverity, setSelectedSeverity] = useState<string>('ALL');
  const [selectedBehavior, setSelectedBehavior] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<'priority' | 'newest' | 'confidence' | 'persistence'>('priority');

  // Filter Logic
  const filteredEvents = events.filter((ev) => {
    if (selectedSeverity !== 'ALL' && ev.severity !== selectedSeverity) return false;
    if (selectedBehavior !== 'ALL' && ev.behavior !== selectedBehavior) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      const matchSeat = ev.seatId.toLowerCase().includes(q);
      const matchCandidate = `candidate #${ev.candidateId}`.toLowerCase().includes(q) || ev.candidateId.toLowerCase().includes(q);
      const matchBehavior = ev.behaviorDisplayTitle.toLowerCase().includes(q);
      if (!matchSeat && !matchCandidate && !matchBehavior) return false;
    }
    return true;
  });

  // Sort Logic
  const sortedEvents = [...filteredEvents].sort((a, b) => {
    if (sortBy === 'confidence') return b.aiConfidence - a.aiConfidence;
    if (sortBy === 'persistence') return b.persistenceSec - a.persistenceSec;
    if (sortBy === 'newest') return b.timestamp.localeCompare(a.timestamp);
    const scoreMap = { escalated: 3, review: 2, observe: 1 };
    return (scoreMap[b.severity] || 0) - (scoreMap[a.severity] || 0);
  });

  const needsReviewCount = events.filter((e) => e.severity === 'review').length;
  const escalatedCount = events.filter((e) => e.severity === 'escalated').length;

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      {/* Header Banner */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200 flex-wrap gap-4">
        <div>
          <div className="font-bold text-lg text-neutral-950 flex items-center gap-3">
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
              <Inbox size={18} />
            </div>
            <span>AI Anomaly Triage & Investigation Queue</span>
            <span className="text-xs px-3 py-1 rounded-full bg-neutral-900 text-white font-bold">
              {needsReviewCount} Needs Review
            </span>
            {escalatedCount > 0 && (
              <span className="text-xs px-3 py-1 rounded-full bg-rose-700 text-white font-bold">
                {escalatedCount} Escalated
              </span>
            )}
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Temporal evidence weighting • Review priority computed from persistence and counter-evidence.
          </div>
        </div>

        {/* Sort Dropdown */}
        <div className="flex items-center gap-2.5">
          <span className="text-neutral-500 text-xs font-semibold uppercase tracking-wider">Sort:</span>
          <select
            value={sortBy}
            onChange={(e) => {
              playHapticSound('click');
              setSortBy(e.target.value as any);
            }}
            className="bg-white border border-neutral-300 text-neutral-900 text-xs px-3.5 py-2 rounded-xl font-sans focus:outline-none focus:border-neutral-900 cursor-pointer shadow-2xs font-semibold"
          >
            <option value="priority">Review Priority (Highest First)</option>
            <option value="confidence">Model Confidence</option>
            <option value="persistence">Signal Persistence</option>
            <option value="newest">Most Recent</option>
          </select>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Left 3.5 Cols: Filter Rail */}
        <div className="col-span-12 lg:col-span-4 space-y-5">
          {/* Search Box */}
          <div className="relative">
            <Search size={16} className="absolute left-4 top-3.5 text-neutral-400" />
            <input
              type="text"
              placeholder="Search seat, candidate ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-white border border-neutral-300 rounded-2xl pl-11 pr-4 py-3 text-xs text-neutral-900 placeholder-neutral-400 font-sans focus:outline-none focus:border-neutral-900 shadow-2xs"
            />
          </div>

          {/* Severity Filter */}
          <div className="bg-white p-6 space-y-4 rounded-3xl border border-neutral-200 shadow-xs">
            <div className="text-xs uppercase text-neutral-500 font-bold tracking-wider flex items-center justify-between">
              <span>Severity Level</span>
              <Filter size={14} className="text-neutral-700" />
            </div>
            <div className="space-y-1.5">
              {[
                { id: 'ALL', label: 'All Signals', count: events.length },
                { id: 'escalated', label: 'Escalated (Critical)', count: events.filter(e => e.severity === 'escalated').length },
                { id: 'review', label: 'Review (Elevated)', count: events.filter(e => e.severity === 'review').length },
                { id: 'observe', label: 'Observe (Routine)', count: events.filter(e => e.severity === 'observe').length },
              ].map((item) => {
                const isSelected = selectedSeverity === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      playHapticSound('click');
                      setSelectedSeverity(item.id);
                    }}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-xl text-xs transition-all cursor-pointer font-medium ${
                      isSelected
                        ? 'bg-neutral-900 text-white font-bold shadow-xs'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                    }`}
                  >
                    <span>{item.label}</span>
                    <span className={`font-mono tabular-nums ${isSelected ? 'text-neutral-300' : 'text-neutral-400'}`}>({item.count})</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Behavior Category Filter */}
          <div className="bg-white p-6 space-y-4 rounded-3xl border border-neutral-200 shadow-xs">
            <div className="text-xs uppercase text-neutral-500 font-bold tracking-wider">
              Pattern Category
            </div>
            <div className="space-y-1.5">
              {[
                { id: 'ALL', label: 'All Patterns' },
                { id: 'possible_object_interaction', label: 'Object / Phone Interaction' },
                { id: 'repeated_sideward_attention', label: 'Sideward Glance / Yaw' },
                { id: 'peer_interaction', label: 'Peer Interaction' },
              ].map((item) => {
                const isSelected = selectedBehavior === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      playHapticSound('click');
                      setSelectedBehavior(item.id);
                    }}
                    className={`w-full text-left px-4 py-3 rounded-xl text-xs transition-all cursor-pointer font-medium ${
                      isSelected
                        ? 'bg-neutral-900 text-white font-bold shadow-xs'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                    }`}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Explainer Box */}
          <div className="bg-neutral-50 p-6 rounded-3xl space-y-2 text-xs text-neutral-600 border border-neutral-200 shadow-2xs">
            <div className="font-bold text-neutral-950 uppercase text-xs flex items-center gap-2">
              <Sparkles size={14} className="text-sky-700" />
              <span>Explainable Triage System</span>
            </div>
            <p className="leading-relaxed">
              <strong className="text-neutral-900">Model Confidence:</strong> Raw multi-modal certainty score.
            </p>
            <p className="leading-relaxed">
              <strong className="text-neutral-900">Review Priority:</strong> Duration persistence + spatial context offset.
            </p>
          </div>
        </div>

        {/* Right 8.5 Cols: Event Cards */}
        <div className="col-span-12 lg:col-span-8 space-y-5">
          {sortedEvents.length === 0 ? (
            <div className="p-16 text-center bg-white rounded-3xl border border-neutral-200 text-neutral-500">
              No AI events match the selected triage filter criteria.
            </div>
          ) : (
            sortedEvents.map((event) => {
              const isEscalated = event.severity === 'escalated';
              const isReview = event.severity === 'review';
              const isDecisionMade = event.humanDecision?.status && event.humanDecision.status !== 'pending';

              return (
                <ParallaxCard
                  key={event.id}
                  tiltAngle={2}
                  className={`p-7 transition-all rounded-3xl bg-white shadow-xs border ${
                    isEscalated
                      ? 'border-rose-300 shadow-rose-50/50'
                      : isReview
                      ? 'border-amber-300 shadow-amber-50/50'
                      : 'border-neutral-200 hover:border-neutral-300'
                  }`}
                >
                  {/* Top Bar of Card */}
                  <div className="flex items-center justify-between pb-4 border-b border-neutral-200 flex-wrap gap-2">
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-neutral-950 text-sm">
                        {event.id}
                      </span>
                      <span className="text-neutral-300">•</span>
                      <span className="font-bold text-neutral-900">
                        Seat {event.seatId}
                      </span>
                      <span className="text-neutral-500 font-semibold">(Candidate #{event.candidateId})</span>
                      <span className="text-neutral-300">•</span>
                      <span className="text-neutral-500 flex items-center gap-1.5 text-xs">
                        <Clock size={13} /> {event.timestamp} ({event.cameraName})
                      </span>
                    </div>

                    <div className="flex items-center gap-2.5">
                      <StatusBadge status={event.severity} />
                      {isDecisionMade && (
                        <span className="px-3 py-0.5 rounded-full bg-neutral-900 text-white text-xs uppercase font-semibold">
                          {event.humanDecision?.status?.replace('_', ' ')}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Body Content */}
                  <div className="grid grid-cols-12 gap-6 py-4 items-stretch">
                    {/* Left: Behavior & Evidence Stack */}
                    <div className="col-span-12 md:col-span-8 space-y-3">
                      <div className="font-bold text-base text-neutral-950 tracking-tight">
                        {event.behaviorDisplayTitle}
                      </div>

                      {/* Evidence Stack Chips */}
                      <div className="space-y-2">
                        {event.evidenceStack.slice(0, 3).map((item) => (
                          <div key={item.id} className="flex items-center justify-between text-xs bg-neutral-50 px-4 py-2.5 rounded-2xl border border-neutral-200">
                            <span className="text-neutral-700">
                              <strong className="text-neutral-950 font-bold font-mono">{item.stepNumber}</strong> {item.title}: {item.description}
                            </span>
                            <span className="text-neutral-900 font-bold font-mono tabular-nums ml-3 shrink-0">
                              {(item.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* Counter Evidence Tag */}
                      {event.counterEvidence.riskSuppressed && (
                        <div className="text-xs bg-emerald-50/80 border border-emerald-200 text-emerald-900 p-3.5 rounded-2xl flex items-center gap-2.5">
                          <CheckCircle2 size={16} className="shrink-0 text-emerald-700" />
                          <span>{event.counterEvidence.suppressionReason}</span>
                        </div>
                      )}
                    </div>

                    {/* Right: Confidence, Priority & CTA */}
                    <div className="col-span-12 md:col-span-4 bg-neutral-50 p-5 rounded-2xl border border-neutral-200 flex flex-col justify-between space-y-4">
                      <div className="space-y-3">
                        <ConfidenceBar
                          value={event.aiConfidence}
                          label="Model Confidence"
                          size="sm"
                        />
                        <div className="flex justify-between text-xs pt-1 border-t border-neutral-200/60">
                          <span className="text-neutral-500">Review Priority:</span>
                          <strong className="text-neutral-900 font-bold">
                            {event.reviewPriority}
                          </strong>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-neutral-500">Duration:</span>
                          <span className="text-neutral-900 font-mono tabular-nums font-bold">{event.persistenceSec} sec</span>
                        </div>
                      </div>

                      <Button
                        variant="default"
                        size="sm"
                        onClick={() => {
                          playHapticSound('snap');
                          onReviewEvent(event);
                        }}
                        className="w-full gap-2 rounded-xl font-semibold shadow-xs"
                      >
                        <span>Review Evidence</span>
                        <ArrowRight size={14} />
                      </Button>
                    </div>
                  </div>
                </ParallaxCard>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
