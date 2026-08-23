import React, { useEffect, useState } from 'react';
import { AppShell, NavigationTab } from './components/shell/AppShell';
import { CommandCenterView } from './components/command-center/CommandCenterView';
import { LiveMonitorView } from './components/video/LiveMonitorView';
import { EventQueueView } from './components/triage/EventQueueView';
import { EvidenceReviewView } from './components/evidence-review/EvidenceReviewView';
import { BenchmarkLabView } from './components/benchmark/BenchmarkLabView';
import { SystemControlView } from './components/system/SystemControlView';
import { CandidatesView } from './components/candidates/CandidatesView';
import { CasesView } from './components/cases/CasesView';
import { AnalyticsView } from './components/analytics/AnalyticsView';
import { DesignSystemView } from './components/design-system/DesignSystemView';

import { SEATS_DATA, INITIAL_EVENTS, CAMERA_FEEDS, SYSTEM_HEALTH } from './data/demoData';
import { SeatNodeData, TriagedAIEvent, SystemHealthData } from './types';

const toFrontendEvent = (record: any): TriagedAIEvent => {
  const candidate = String(record.candidate_ids?.[0] ?? 'UNKNOWN');
  const score = Number(record.evidence_score || 0);
  const eventType = String(record.event_type || 'REVIEW');
  const behavior = eventType.includes('SCREEN') ? 'repeated_sideward_attention'
    : eventType.includes('INTERACTION') ? 'peer_interaction'
    : eventType.includes('HAND') || eventType.includes('CHIT') ? 'unusual_hand_movement'
    : 'possible_object_interaction';
  return {
    id: record.event_id, timestamp: record.started_at_utc ? new Date(record.started_at_utc).toLocaleTimeString('en-GB') : `${Number(record.started_at_sec || 0).toFixed(3)}s`,
    videoTimestampSec: Number(record.started_at_sec || 0), seatId: record.workstation_ids?.[0] || `TRACK-${candidate}`,
    candidateId: candidate, candidateName: `Tracked Candidate ${candidate}`, cameraId: record.camera_id,
    cameraName: record.camera_id, behavior, behaviorDisplayTitle: record.label || eventType,
    severity: score >= .7 ? 'escalated' : score >= .45 ? 'review' : 'observe', aiConfidence: score,
    reviewPriority: score >= .7 ? 'CRITICAL' : score >= .45 ? 'ELEVATED' : 'ROUTINE', persistenceSec: Number(record.duration_sec || 0),
    videoClipUrl: record.clip_path ? `http://127.0.0.1:8000/api/events/${record.event_id}/clip` : undefined,
    evidenceStack: (record.source_signals || []).map((signal: string, index: number) => ({ id: `${record.event_id}-${index}`, stepNumber: String(index + 1).padStart(2, '0'), title: signal, category: 'temporal' as const, confidence: score, durationSec: Number(record.duration_sec || 0), description: signal })),
    counterEvidence: { factorsFound: [], riskSuppressed: false },
    context: { seatId: record.workstation_ids?.[0] || `TRACK-${candidate}`, neighborSeatId: record.workstation_ids?.[1] || 'UNKNOWN', boardDirection: 'FRONT', invigilatorProximity: 'ABSENT' },
  };
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavigationTab>('command-center');
  const [seats, setSeats] = useState<SeatNodeData[]>(SEATS_DATA);
  const [events, setEvents] = useState<TriagedAIEvent[]>(INITIAL_EVENTS);
  const [health] = useState<SystemHealthData>(SYSTEM_HEALTH);
  const [selectedSeatId, setSelectedSeatId] = useState<string | null>(null);
  const [activeEventForReview, setActiveEventForReview] = useState<TriagedAIEvent>(INITIAL_EVENTS[0]);

  useEffect(() => {
    const api = (import.meta.env.VITE_NETRA_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
    fetch(`${api}/api/events?limit=200`).then((response) => response.ok ? response.json() : Promise.reject()).then((records) => {
      if (records.length) setEvents(records.map(toFrontendEvent));
    }).catch(() => undefined);
    const socket = new WebSocket(`${api.replace(/^http/, 'ws')}/ws/events`);
    socket.onmessage = (message) => {
      const incoming = toFrontendEvent(JSON.parse(message.data));
      setEvents((current) => [incoming, ...current.filter((event) => event.id !== incoming.id)]);
    };
    return () => socket.close();
  }, []);

  // Handle seat selection from Map
  const handleSelectSeat = (seatId: string) => {
    setSelectedSeatId(seatId);
  };

  // Handle "Review Evidence" CTA from anywhere
  const handleReviewEvent = (event: TriagedAIEvent) => {
    setActiveEventForReview(event);
    setSelectedSeatId(event.seatId);
    setActiveTab('evidence-review');
  };

  // Human-in-the-Loop Decision Recording
  const handleDecision = (
    eventId: string,
    decision: 'marked_normal' | 'needs_follow_up' | 'confirmed_violation' | 'escalated',
    note?: string
  ) => {
    setEvents((prev) =>
      prev.map((ev) => {
        if (ev.id === eventId) {
          return {
            ...ev,
            humanDecision: {
              status: decision,
              reviewerId: 'Chief Invigilator Dr. K. Rao',
              notes: note || ev.humanDecision?.notes,
              decidedAt: new Date().toLocaleTimeString('en-GB') + ' IST',
            },
            severity:
              decision === 'marked_normal'
                ? 'observe'
                : decision === 'confirmed_violation'
                ? 'escalated'
                : ev.severity,
          };
        }
        return ev;
      })
    );

    // Also update terminal state in Map
    const targetEvent = events.find((e) => e.id === eventId);
    if (targetEvent) {
      setSeats((prev) =>
        prev.map((s) => {
          if (s.id === targetEvent.seatId) {
            const nextState =
              decision === 'marked_normal'
                ? 'normal'
                : decision === 'confirmed_violation' || decision === 'escalated'
                ? 'escalated'
                : decision === 'needs_follow_up'
                ? 'review'
                : s.state;
            return { ...s, state: nextState };
          }
          return s;
        })
      );
    }
  };

  // Demo Preset Selector Engine
  const handleSelectDemoPreset = (presetId: string) => {
    switch (presetId) {
      case 'preset-phone': {
        const ev = events.find((e) => e.id === 'EV-2038') || events[0];
        handleReviewEvent(ev);
        break;
      }
      case 'preset-glance': {
        const ev = events.find((e) => e.id === 'EV-2037') || events[0];
        handleReviewEvent(ev);
        break;
      }
      case 'preset-counter': {
        const ev = events.find((e) => e.id === 'EV-2036') || events[0];
        handleReviewEvent(ev);
        break;
      }
      case 'preset-hard-negative': {
        const ev = events.find((e) => e.id === 'EV-2035') || events[0];
        handleReviewEvent(ev);
        break;
      }
      case 'preset-peer': {
        const ev = events.find((e) => e.id === 'EV-2034') || events[0];
        handleReviewEvent(ev);
        break;
      }
      case 'preset-benchmark': {
        setActiveTab('benchmark');
        break;
      }
      default:
        break;
    }
  };

  const needsReviewCount = events.filter(
    (e) => e.severity === 'review' && (!e.humanDecision || e.humanDecision.status === 'pending')
  ).length;

  return (
    <AppShell
      activeTab={activeTab}
      onSelectTab={setActiveTab}
      health={health}
      eventsCount={{ needsReview: needsReviewCount, total: events.length, escalated: events.filter(e => e.severity === 'escalated').length }}
      onSelectDemoPreset={handleSelectDemoPreset}
      activeEventForReview={activeEventForReview}
    >
      {/* 1. COMMAND CENTER */}
      {activeTab === 'command-center' && (
        <CommandCenterView
          seats={seats}
          events={events}
          health={health}
          onSelectSeat={handleSelectSeat}
          onReviewEvent={handleReviewEvent}
          selectedSeatId={selectedSeatId}
          onUpdateSeats={setSeats}
          cameras={CAMERA_FEEDS}
          onViewCandidateDossier={(seatId) => {
            setSelectedSeatId(seatId);
            setActiveTab('candidates');
          }}
        />
      )}

      {/* 2. LIVE MONITOR */}
      {activeTab === 'live-monitor' && (
        <LiveMonitorView
          cameras={CAMERA_FEEDS}
          onReviewEvent={handleReviewEvent}
          events={events}
        />
      )}

      {/* 3. EVENT QUEUE (TRIAGE) */}
      {activeTab === 'event-queue' && (
        <EventQueueView
          events={events}
          onReviewEvent={handleReviewEvent}
        />
      )}

      {/* 4. EVIDENCE REVIEW */}
      {activeTab === 'evidence-review' && (
        <EvidenceReviewView
          event={activeEventForReview}
          onDecision={handleDecision}
          onBackToQueue={() => setActiveTab('event-queue')}
        />
      )}

      {/* 5. CANDIDATE INTELLIGENCE */}
      {activeTab === 'candidates' && (
        <CandidatesView
          seats={seats}
          events={events}
          onReviewEvent={handleReviewEvent}
          selectedSeatId={selectedSeatId}
        />
      )}

      {/* 6. FORMAL CASES & AUDIT EXPORT */}
      {activeTab === 'cases' && <CasesView />}

      {/* 7. TELEMETRY & OPERATIONS ANALYTICS */}
      {activeTab === 'analytics' && (
        <AnalyticsView
          seats={seats}
          onSelectCandidate={(seatId) => {
            setSelectedSeatId(seatId);
            setActiveTab('candidates');
          }}
        />
      )}

      {/* 8. SYSTEM HARDWARE & CAMERAS */}
      {activeTab === 'system-cameras' && (
        <SystemControlView cameras={CAMERA_FEEDS} health={health} />
      )}

      {/* 9. BENCHMARK LAB */}
      {activeTab === 'benchmark' && <BenchmarkLabView />}

      {/* 10. LIVING DESIGN SYSTEM SPECIFICATION */}
      {activeTab === 'design-system' && <DesignSystemView />}
    </AppShell>
  );
};
export default App;
