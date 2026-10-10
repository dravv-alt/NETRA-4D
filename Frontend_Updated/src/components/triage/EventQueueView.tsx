import React, { useState, useRef, useEffect, useCallback } from 'react';
import { TriagedAIEvent } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { Filter, Search, ArrowRight, Clock, CheckCircle2, Sparkles, Inbox, RotateCcw, ChevronUp, ChevronDown } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';

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
  const [selectedDecision, setSelectedDecision] = useState<string>('PENDING');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<'priority' | 'newest' | 'confidence' | 'persistence'>('priority');

  const pendingEvents = events.filter((event) => !event.humanDecision || event.humanDecision.status === 'pending');
  const behaviorOptions = Array.from(new Set(events.map((event) => event.behavior))).sort();

  // Filter Logic: default to evidence still awaiting an invigilator decision.
  const filteredEvents = events.filter((ev) => {
    const isPending = !ev.humanDecision || ev.humanDecision.status === 'pending';
    if (selectedDecision === 'PENDING' && !isPending) return false;
    if (selectedDecision === 'DECIDED' && isPending) return false;
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
    if (sortBy === 'newest') return b.videoTimestampSec - a.videoTimestampSec;
    const scoreMap = { escalated: 3, review: 2, observe: 1 };
    return (scoreMap[b.severity] || 0) - (scoreMap[a.severity] || 0);
  });

  const needsReviewCount = pendingEvents.length;
  const escalatedCount = pendingEvents.filter((e) => e.severity === 'escalated').length;

  // Discrete 1-Scroll = 1-Card Snap Navigation
  const [activeCardIndex, setActiveCardIndex] = useState(0);
  const rightColRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);
  const isScrollingRef = useRef(false);

  // Reset card index & scroll to top when filters or search change
  useEffect(() => {
    setActiveCardIndex(0);
    cardRefs.current = [];
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [selectedSeverity, selectedBehavior, selectedDecision, searchQuery, sortBy]);

  const scrollToCard = useCallback((index: number) => {
    const container = containerRef.current;
    const target = cardRefs.current[index];
    if (!container || !target) return;

    container.scrollTo({
      top: target.offsetTop,
      behavior: 'smooth',
    });
    setActiveCardIndex(index);
  }, []);

  const handleStepCard = useCallback((direction: 'up' | 'down') => {
    if (direction === 'down') {
      if (activeCardIndex < sortedEvents.length - 1) {
        scrollToCard(activeCardIndex + 1);
      }
    } else {
      if (activeCardIndex > 0) {
        scrollToCard(activeCardIndex - 1);
      }
    }
  }, [activeCardIndex, sortedEvents.length, scrollToCard]);

  // Non-passive wheel listener: 1 wheel gesture = 1 card advance cleanly
  useEffect(() => {
    const col = rightColRef.current;
    const container = containerRef.current;
    if (!col || !container) return;

    let wheelTimeout: ReturnType<typeof setTimeout> | null = null;

    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) < 15) return;

      e.preventDefault();

      if (isScrollingRef.current) return;

      const isDown = e.deltaY > 0;
      isScrollingRef.current = true;

      setActiveCardIndex((prev) => {
        const next = isDown
          ? Math.min(prev + 1, sortedEvents.length - 1)
          : Math.max(prev - 1, 0);

        const target = cardRefs.current[next];
        if (target && container) {
          container.scrollTo({
            top: target.offsetTop,
            behavior: 'smooth',
          });
        }
        return next;
      });

      if (wheelTimeout) clearTimeout(wheelTimeout);
      wheelTimeout = setTimeout(() => {
        isScrollingRef.current = false;
      }, 350);
    };

    col.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      col.removeEventListener('wheel', onWheel);
      if (wheelTimeout) clearTimeout(wheelTimeout);
    };
  }, [sortedEvents.length]);

  // Track closest card on manual scrollbar drag
  const handleContainerScroll = useCallback(() => {
    if (isScrollingRef.current || !containerRef.current) return;
    const scrollTop = containerRef.current.scrollTop;
    let closest = 0;
    let minDiff = Infinity;
    cardRefs.current.forEach((ref, idx) => {
      if (!ref) return;
      const diff = Math.abs(ref.offsetTop - scrollTop);
      if (diff < minDiff) {
        minDiff = diff;
        closest = idx;
      }
    });
    if (closest !== activeCardIndex) {
      setActiveCardIndex(closest);
    }
  }, [activeCardIndex]);

  return (
    <div className="space-y-4 sm:space-y-6 max-w-7xl mx-auto font-sans text-xs sm:text-sm text-neutral-900">
      <section className="rounded-2xl sm:rounded-3xl border border-neutral-200 bg-white p-3.5 sm:p-5 lg:p-7 shadow-xs flex flex-col lg:h-[calc(100dvh-10rem)] lg:max-h-[900px]">
      {/* Header Banner */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200 flex-wrap gap-4 shrink-0">
        <div>
          <div className="font-bold text-lg text-neutral-950 flex items-center gap-3">
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
              <Inbox size={18} />
            </div>
            <span>AI Anomaly Triage & Investigation Queue</span>
            <span className="text-xs px-3 py-1 rounded-full bg-neutral-900 text-white font-bold">
              {needsReviewCount} Pending Review
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

      {/* Main Grid: matched stretch columns with internal scrolling */}
      <div className="grid grid-cols-12 gap-8 items-stretch flex-1 min-h-0 pt-4">
        {/* Left 3.5 Cols: Filter Rail */}
        <div className="col-span-12 lg:col-span-4 h-full min-h-0 overflow-y-auto space-y-4 pr-2 scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent">
          {/* Search Box */}
          <div className="relative">
            <Search size={16} className="absolute left-4 top-3.5 text-neutral-400" />
            <input
              type="text"
              placeholder="Search candidate, workstation, event..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-white border border-neutral-300 rounded-2xl pl-11 pr-4 py-3 text-xs text-neutral-900 placeholder-neutral-400 font-sans focus:outline-none focus:border-neutral-900 shadow-2xs"
            />
          </div>

          <div className="bg-white p-6 space-y-3 rounded-3xl border border-neutral-200 shadow-xs">
            <div className="text-xs uppercase text-neutral-500 font-bold tracking-wider">Decision State</div>
            {[{ id: 'PENDING', label: 'Awaiting Invigilator', count: pendingEvents.length }, { id: 'DECIDED', label: 'Already Decided', count: events.length - pendingEvents.length }, { id: 'ALL', label: 'All Evidence', count: events.length }].map((item) => (
              <button key={item.id} onClick={() => setSelectedDecision(item.id)} className={`w-full flex items-center justify-between px-4 py-3 rounded-xl text-xs transition-all cursor-pointer font-medium ${selectedDecision === item.id ? 'bg-neutral-900 text-white font-bold' : 'text-neutral-600 hover:bg-neutral-100'}`}>
                <span>{item.label}</span><span className={selectedDecision === item.id ? 'text-neutral-300' : 'text-neutral-400'}>({item.count})</span>
              </button>
            ))}
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
                ...behaviorOptions.map((behavior) => ({ id: behavior, label: behavior.replaceAll('_', ' ') })),
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

          <button onClick={() => { setSelectedSeverity('ALL'); setSelectedBehavior('ALL'); setSelectedDecision('PENDING'); setSearchQuery(''); }} className="w-full flex items-center justify-center gap-2 rounded-xl border border-neutral-200 bg-white px-4 py-3 text-xs font-semibold text-neutral-600 hover:bg-neutral-50">
            <RotateCcw size={13} /> Reset queue filters
          </button>

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

        {/* Right 8.5 Cols: Event Cards with Internal Discrete Snap Scroll */}
        <div
          ref={rightColRef}
          className="col-span-12 lg:col-span-8 h-full min-h-0 flex flex-col"
        >
          {/* Snap-scroll Queue Status & Stepper */}
          <div className="flex items-center justify-between pb-3 text-xs text-neutral-500 font-mono shrink-0">
            <div className="flex items-center gap-2">
              <span className="font-bold text-neutral-900">
                Card {sortedEvents.length > 0 ? activeCardIndex + 1 : 0} of {sortedEvents.length}
              </span>
              <span className="text-neutral-300">•</span>
              <span className="text-neutral-500 font-sans text-xs">
                1 scroll = 1 card snap
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => handleStepCard('up')}
                disabled={activeCardIndex <= 0}
                className="p-1.5 rounded-lg border border-neutral-200 bg-white hover:bg-neutral-100 disabled:opacity-30 disabled:pointer-events-none text-neutral-700 transition-colors cursor-pointer"
                title="Previous Card (Up)"
                aria-label="Previous Card"
              >
                <ChevronUp size={15} />
              </button>
              <button
                type="button"
                onClick={() => handleStepCard('down')}
                disabled={activeCardIndex >= sortedEvents.length - 1}
                className="p-1.5 rounded-lg border border-neutral-200 bg-white hover:bg-neutral-100 disabled:opacity-30 disabled:pointer-events-none text-neutral-700 transition-colors cursor-pointer"
                title="Next Card (Down)"
                aria-label="Next Card"
              >
                <ChevronDown size={15} />
              </button>
            </div>
          </div>

          {/* Internal Scrollable Container */}
          <div
            ref={containerRef}
            onScroll={handleContainerScroll}
            className="relative flex-1 min-h-0 overflow-y-auto overscroll-contain snap-y snap-mandatory scroll-smooth space-y-4 p-2.5 rounded-3xl bg-neutral-100/60 border border-neutral-200/80 focus:outline-none scrollbar-thin scrollbar-thumb-neutral-300 scrollbar-track-transparent"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault();
                handleStepCard('down');
              } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                handleStepCard('up');
              }
            }}
            aria-label="Internal snap-scrollable event investigation queue"
          >
          {sortedEvents.length === 0 ? (
            <div className="p-16 text-center bg-white rounded-3xl border border-neutral-200 text-neutral-500">
              No AI events match the selected triage filter criteria.
            </div>
          ) : (
            sortedEvents.map((event, index) => {
              const isEscalated = event.severity === 'escalated';
              const isReview = event.severity === 'review';
              const isDecisionMade = event.humanDecision?.status && event.humanDecision.status !== 'pending';
              const isActiveCard = index === activeCardIndex;

              const cardStyle = isEscalated
                ? 'bg-gradient-to-br from-[#FFF5F6] via-[#FFF9FA] to-white border-rose-300 shadow-[0_4px_24px_-4px_rgba(244,63,94,0.12)]'
                : isReview
                ? 'bg-gradient-to-br from-[#FFFDF2] via-[#FFFFFA] to-white border-amber-300 shadow-[0_4px_24px_-4px_rgba(245,158,11,0.12)]'
                : 'bg-gradient-to-br from-[#F0F7FF] via-[#F8FAFC] to-white border-sky-200 shadow-[0_4px_24px_-4px_rgba(14,165,233,0.08)]';

              const cardHeaderBorder = isEscalated
                ? 'border-rose-200/70'
                : isReview
                ? 'border-amber-200/70'
                : 'border-sky-200/70';

              const chipBg = isEscalated
                ? 'bg-white/90 border-rose-200/60 shadow-2xs'
                : isReview
                ? 'bg-white/90 border-amber-200/60 shadow-2xs'
                : 'bg-white/90 border-sky-200/60 shadow-2xs';

              const metricsBg = isEscalated
                ? 'bg-rose-50/70 border-rose-200/70'
                : isReview
                ? 'bg-amber-50/70 border-amber-200/70'
                : 'bg-sky-50/60 border-sky-200/70';

              return (
                <div
                  key={event.id}
                  ref={(el) => {
                    cardRefs.current[index] = el;
                  }}
                  className={`snap-start snap-always scroll-mt-0 p-7 transition-all duration-200 rounded-3xl border ${cardStyle} ${
                    isActiveCard ? 'ring-2 ring-neutral-900/10' : ''
                  }`}
                >
                  {/* Top Bar of Card */}
                  <div className={`flex items-center justify-between pb-4 border-b flex-wrap gap-2 ${cardHeaderBorder}`}>
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
                        <Clock size={13} /> Source {event.videoTimestampSec.toFixed(3)}s · {event.timestamp} ({event.cameraName})
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
                        {(event.evidenceStack.length ? event.evidenceStack.slice(0, 3) : [{ id: `${event.id}-no-signal`, stepNumber: '--', title: 'No additional signal', description: 'Open the evidence record for the raw event payload.', confidence: event.aiConfidence }]).map((item) => (
                          <div key={item.id} className={`flex items-center justify-between text-xs px-4 py-2.5 rounded-2xl border ${chipBg}`}>
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
                    <div className={`col-span-12 md:col-span-4 p-5 rounded-2xl border flex flex-col justify-between space-y-4 ${metricsBg}`}>
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
                          <span className="text-neutral-900 font-mono tabular-nums font-bold">{event.persistenceSec.toFixed(2)} sec</span>
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
                        <span>{isDecisionMade ? 'Open Decision Record' : 'Review Evidence'}</span>
                        <ArrowRight size={14} />
                      </Button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
          </div>
        </div>
      </div>
      </section>
    </div>
  );
};
