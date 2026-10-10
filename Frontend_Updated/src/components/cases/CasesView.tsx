import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { FileCheck, Download, CheckCircle2, User, Lock, ShieldCheck, FileSpreadsheet, ExternalLink } from 'lucide-react';
import { StatusBadge } from '../ui/StatusBadge';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { ParallaxCard } from '../ui/parallax-card';
import { InstitutionalAuditExportModal } from './InstitutionalAuditExportModal';

export const CasesView: React.FC = () => {
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const cases = [
    {
      id: 'CASE #C-024',
      candidate: 'Candidate #18 (Seat B08)',
      session: 'Computer Lab 01 • CBT Session 01',
      status: 'CONFIRMED VIOLATION',
      reviewer: 'Chief Invigilator Dr. K. Rao',
      linkedEvents: ['EV-2034 (Paper Exchange Interaction)'],
      clips: 2,
      screenshots: 14,
      aiSummary: 'Reciprocal body rotation (+44° / -41°), wrist boundary crossing, physical sheet transfer confirmed across laptops.',
      notes: 'Invigilator intercepted candidate with unauthorized formula sheet after reciprocal anomaly alert.',
      createdAt: '15:34:00 IST',
    },
    {
      id: 'CASE #C-025',
      candidate: 'Candidate #6 (Seat A06)',
      session: 'Computer Lab 01 • CBT Session 01',
      status: 'UNDER INVESTIGATION',
      reviewer: 'Invigilator 02 (Proctor Walkway)',
      linkedEvents: ['EV-2038 (Possible Object Interaction)'],
      clips: 1,
      screenshots: 8,
      aiSummary: 'Phone candidate detected below desk plane (0.74 score) sustained over 6.2s with downward head pitch.',
      notes: 'Candidate flagged for physical desk inspection at exam completion.',
      createdAt: '15:42:25 IST',
    }
  ];

  const handleOpenExportModal = (caseId: string) => {
    playHapticSound('snap');
    setSelectedCaseId(caseId);
    setIsModalOpen(true);
  };

  return (
    <div className="space-y-4 sm:space-y-8 max-w-7xl mx-auto font-sans text-xs sm:text-sm text-neutral-900">
      {/* 1. STANDARDIZED PAGE HEADER */}
      <div className="pb-4 sm:pb-6 border-b border-neutral-200/80">
        <div className="flex items-center gap-2 text-2xs sm:text-xs font-mono uppercase tracking-widest text-neutral-500 mb-1.5 sm:mb-2">
          <FileCheck size={13} className="text-neutral-700" />
          <span>INSTITUTIONAL DISCIPLINARY RECORDS • SHA-256 IMMUTABLE</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-3 sm:gap-4">
          <div>
            <h1 className="text-xl sm:text-3xl md:text-4xl font-extrabold font-heading text-neutral-950 tracking-tight">
              Formal Incident Cases & Audit Records
            </h1>
            <p className="text-neutral-600 text-xs sm:text-sm mt-1 sm:mt-1.5 max-w-2xl leading-relaxed">
              Cryptographically sealed incident packages ready for university disciplinary boards and examination controller review.
            </p>
          </div>
          <span className="text-2xs sm:text-xs font-mono font-bold text-neutral-900 px-3 py-1 sm:px-3.5 sm:py-1.5 rounded-full bg-white border border-neutral-200 shadow-2xs">
            2 ACTIVE CASES
          </span>
        </div>
      </div>

      {/* Cases List */}
      <div className="space-y-4 sm:space-y-6">
        {cases.map((c) => (
          <div
            key={c.id}
            className="p-4 sm:p-6 lg:p-8 space-y-4 sm:space-y-6 bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] rounded-2xl sm:rounded-[2rem]"
          >
            {/* Header */}
            <div className="flex items-center justify-between pb-4 sm:pb-5 border-b border-neutral-200/80 flex-wrap gap-3 sm:gap-4">
              <div className="flex items-center gap-3 sm:gap-4">
                <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl sm:rounded-2xl bg-neutral-900 flex items-center justify-center text-white font-bold shadow-xs shrink-0">
                  <Lock size={18} />
                </div>
                <div>
                  <div className="font-heading font-extrabold text-base text-neutral-950 flex items-center gap-2.5">
                    <span>{c.id}</span>
                    <span className="text-neutral-300">•</span>
                    <span>{c.candidate}</span>
                  </div>
                  <div className="text-xs font-mono text-neutral-500 mt-1">
                    {c.session} • Created: {c.createdAt}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <StatusBadge
                  status={c.status === 'CONFIRMED VIOLATION' ? 'escalated' : 'review'}
                  label={c.status}
                />
                <Button
                  variant="default"
                  size="sm"
                  onClick={() => handleOpenExportModal(c.id)}
                  className="gap-2 rounded-xl font-mono text-xs shadow-xs"
                >
                  <ExternalLink size={13} />
                  <span>Generate Institutional Dossier</span>
                </Button>
              </div>
            </div>

            {/* Evidence & Details Grid */}
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-12 lg:col-span-8 space-y-4 text-xs">
                <div className="bg-neutral-50 p-5 rounded-2xl border border-neutral-200/80 space-y-2">
                  <div className="font-heading font-bold text-neutral-950 flex items-center gap-2">
                    <ShieldCheck size={15} className="text-emerald-700" />
                    <span>AI Multi-Modal Forensic Summary</span>
                  </div>
                  <div className="text-neutral-700 leading-relaxed font-sans">{c.aiSummary}</div>
                </div>

                <div className="bg-white p-5 rounded-2xl border border-neutral-200/80 space-y-2 shadow-2xs">
                  <div className="font-heading font-bold text-neutral-950 flex items-center gap-2">
                    <User size={15} className="text-neutral-800" />
                    <span>Invigilator Review Note ({c.reviewer})</span>
                  </div>
                  <div className="text-neutral-700 italic leading-relaxed font-sans">"{c.notes}"</div>
                </div>
              </div>

              <div className="col-span-12 lg:col-span-4 bg-neutral-50 p-5 rounded-2xl border border-neutral-200/80 space-y-3 text-xs font-mono">
                <div className="font-heading font-bold text-neutral-950 pb-2 border-b border-neutral-200 flex items-center gap-2">
                  <FileSpreadsheet size={15} className="text-neutral-800" />
                  <span>Archived Artifacts</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-500">Linked Telemetry:</span>
                  <strong className="text-neutral-950 font-bold">{c.linkedEvents.length} Events</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-500">Video Clips:</span>
                  <strong className="text-neutral-950 font-bold">{c.clips} MP4 Clips</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-500">Keypoint Stills:</span>
                  <strong className="text-neutral-950 font-bold">{c.screenshots} JPEGs</strong>
                </div>
                <div className="flex justify-between pt-2 border-t border-neutral-200">
                  <span className="text-neutral-500">Integrity Signature:</span>
                  <span className="text-emerald-800 font-bold text-3xs bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">SHA-256 Verified</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Cryptographic Export Modal */}
      <InstitutionalAuditExportModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        caseId={selectedCaseId || undefined}
      />
    </div>
  );
};
