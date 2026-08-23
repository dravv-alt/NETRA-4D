import React, { useState } from 'react';
import { SeatNodeData, TriagedAIEvent } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { Users, Search, ArrowRight, Video, FileText, Activity, ShieldCheck, Sparkles, User, Laptop } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { IllustratedIcon } from '../ui/IllustratedIcon';
import { ExamCohortPostureBiometricsRadar } from '../telemetry/ExamCohortPostureBiometricsRadar';

interface CandidatesViewProps {
  seats: SeatNodeData[];
  events: TriagedAIEvent[];
  onReviewEvent: (event: TriagedAIEvent) => void;
  selectedSeatId?: string | null;
}

export const CandidatesView: React.FC<CandidatesViewProps> = ({
  seats,
  events,
  onReviewEvent,
  selectedSeatId,
}) => {
  const [selectedSeat, setSelectedSeat] = useState<string>(selectedSeatId || (seats[0]?.id || 'A01'));
  const [search, setSearch] = useState<string>('');

  const activeCandidate = seats.find((s) => s.id === selectedSeat) || seats[0];
  const candidateEvents = events.filter((e) => e.seatId === selectedSeat);

  const filteredSeats = seats.filter((s) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return s.id.toLowerCase().includes(q) || `candidate #${s.candidateId}`.toLowerCase().includes(q) || s.candidateId.toLowerCase().includes(q);
  });

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      {/* 1. STANDARDIZED PAGE HEADER */}
      <div className="pb-6 border-b border-neutral-200/80">
        <div className="flex items-center gap-2.5 text-xs font-mono uppercase tracking-widest text-neutral-500 mb-2">
          <IllustratedIcon icon={Users} variant="sky" size="sm" />
          <span>INDIVIDUAL COHORT PROFILES • 20 LAB TERMINALS</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl sm:text-4xl font-extrabold font-heading text-neutral-950 tracking-tight">
              Candidate Dossiers & Telemetry Records
            </h1>
            <p className="text-neutral-600 text-sm mt-1.5 max-w-2xl leading-relaxed">
              Longitudinal exam trail, 5-axis posture kinematics comparison, and incident history categorized by Candidate ID.
            </p>
          </div>
          <span className="text-xs font-mono font-bold text-neutral-900 px-3.5 py-1.5 rounded-full bg-white border border-neutral-200 shadow-2xs">
            20 CANDIDATES ACTIVE
          </span>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MASTER SUITE: CANDIDATE DIRECTORY & DEEP FORENSIC PROFILE MASTER DECK */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Left 4 Cols: Candidate Directory Master Card */}
        <div className="col-span-12 lg:col-span-4 p-6 sm:p-7 rounded-[2.5rem] bg-white border border-neutral-200/90 shadow-[0_24px_50px_rgba(0,0,0,0.04)] space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-200/80">
            <span className="font-heading font-extrabold text-base text-neutral-950">
              Cohort Roster
            </span>
            <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded-full bg-neutral-100 text-neutral-700">
              {filteredSeats.length} Active
            </span>
          </div>

          <div className="relative">
            <Search size={15} className="absolute left-3.5 top-3 text-neutral-400" />
            <input
              type="text"
              placeholder="Search Candidate # or Terminal..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-[#F5F4F0] border border-neutral-200/80 rounded-2xl pl-10 pr-4 py-2.5 text-xs text-neutral-900 placeholder-neutral-400 font-mono focus:outline-none focus:border-neutral-900 focus:bg-white transition-all shadow-inner"
            />
          </div>

          <div className="space-y-2 max-h-[calc(100vh-320px)] overflow-y-auto pr-1">
            {filteredSeats.map((seat) => {
              const isSelected = selectedSeat === seat.id;
              return (
                <button
                  key={seat.id}
                  onClick={() => {
                    playHapticSound('click');
                    setSelectedSeat(seat.id);
                  }}
                  className={`w-full p-4 rounded-2xl text-left transition-all border cursor-pointer ${
                    isSelected
                      ? 'bg-[#181716] text-[#FAF7F3] border-neutral-900 shadow-md'
                      : 'bg-[#F8F9FA] text-neutral-700 hover:text-neutral-950 hover:bg-neutral-100 border-neutral-200/80'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="font-heading font-bold text-sm">
                      Candidate #{seat.candidateId}
                    </span>
                    <StatusBadge status={seat.state} size="sm" />
                  </div>
                  <div className={`text-2xs font-mono ${isSelected ? 'text-neutral-400' : 'text-neutral-500'}`}>
                    Terminal {seat.id} ({seat.gridPosition.row === 'A' ? 'North Desk' : 'South Desk'})
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right 8 Cols: Candidate Deep Telemetry Profile Master Card */}
        <div className="col-span-12 lg:col-span-8 p-8 sm:p-9 rounded-[2.5rem] bg-white border border-neutral-200/90 shadow-[0_24px_50px_rgba(0,0,0,0.04)] space-y-7">
          {activeCandidate && (
            <>
              {/* Profile Header */}
              <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-3">
                <div className="flex items-center gap-3.5">
                  <div className="w-12 h-12 rounded-2xl bg-neutral-900 text-white font-mono font-extrabold text-base flex items-center justify-center shadow-sm">
                    {activeCandidate.id}
                  </div>
                  <div>
                    <div className="font-heading font-extrabold text-xl text-neutral-950 flex items-center gap-2.5">
                      <span>Candidate #{activeCandidate.candidateId}</span>
                      <StatusBadge status={activeCandidate.state} />
                    </div>
                    <div className="text-xs text-neutral-500 font-mono mt-0.5">
                      Terminal {activeCandidate.id} • {activeCandidate.gridPosition.row === 'A' ? 'North Wall Desk (Facing North)' : 'South Wall Desk (Facing South)'}
                    </div>
                  </div>
                </div>

                <div className="text-right font-mono text-xs">
                  <div className="text-3xs text-neutral-400 font-bold uppercase">CCTV Feeds</div>
                  <div className="font-bold text-neutral-900">{activeCandidate.cameraIds.join(', ')}</div>
                </div>
              </div>

              {/* 3 Inset Telemetry Cards (Level 2 Insets) */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-1">
                  <div className="text-3xs font-mono uppercase text-neutral-500 font-bold">Track Continuity</div>
                  <div className="font-mono font-extrabold text-xl text-emerald-700">{activeCandidate.telemetry.trackContinuity}</div>
                </div>

                <div className="p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-1">
                  <div className="text-3xs font-mono uppercase text-neutral-500 font-bold">Head / Gaze Vector</div>
                  <div className="font-mono font-extrabold text-xl text-neutral-950">{activeCandidate.telemetry.headEvidence}</div>
                </div>

                <div className="p-5 rounded-2xl bg-[#F5F4F0] border border-neutral-200/60 space-y-1">
                  <div className="text-3xs font-mono uppercase text-neutral-500 font-bold">Laptop Keyboard Hands</div>
                  <div className="font-mono font-extrabold text-xl text-neutral-950">{activeCandidate.telemetry.handsEvidence}</div>
                </div>
              </div>

              {/* Candidate Flagged Incidents Breakdown */}
              <div className="space-y-4 pt-2">
                <div className="font-heading font-extrabold text-base text-neutral-950">
                  Surveillance Findings for Candidate #{activeCandidate.candidateId}
                </div>

                {candidateEvents.length > 0 ? (
                  <div className="space-y-3">
                    {candidateEvents.map((ev) => (
                      <div
                        key={ev.id}
                        className="p-5 rounded-2xl border border-neutral-200/80 bg-[#F8F9FA] hover:border-neutral-400 transition-all flex items-center justify-between shadow-2xs flex-wrap gap-3"
                      >
                        <div>
                          <div className="font-heading font-bold text-sm text-neutral-950 flex items-center gap-2">
                            <span>{ev.behaviorDisplayTitle}</span>
                            <StatusBadge status={ev.severity} size="sm" />
                          </div>
                          <div className="text-xs text-neutral-500 font-mono mt-1">
                            {ev.timestamp} IST • AI Confidence: {(ev.aiConfidence * 100).toFixed(0)}% • Duration: {ev.persistenceSec}s
                          </div>
                        </div>

                        <Button
                          variant="default"
                          size="sm"
                          onClick={() => {
                            playHapticSound('snap');
                            onReviewEvent(ev);
                          }}
                          className="gap-2 rounded-xl font-mono text-xs font-bold shadow-xs px-4"
                        >
                          <span>Launch Forensic Review</span>
                          <ArrowRight size={13} />
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-6 text-center text-xs font-mono text-neutral-500 bg-[#F5F4F0] rounded-2xl border border-neutral-200/60">
                    ✓ No active anomalies or flagged incidents recorded for Candidate #{activeCandidate.candidateId}. Nominal screen focus verified.
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
