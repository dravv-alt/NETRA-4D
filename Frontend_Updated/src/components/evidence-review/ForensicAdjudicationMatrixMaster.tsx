import React from 'react';
import { motion } from 'motion/react';
import { Shield, Smartphone, Eye, Hand, Activity, CheckCircle2, AlertTriangle, ArrowRight, FileCheck, Truck, Lock } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';

interface ForensicAdjudicationMatrixMasterProps {
  onConfirmViolation?: () => void;
  onDismissAnomaly?: () => void;
  onReturnToCockpit?: () => void;
}

export const ForensicAdjudicationMatrixMaster: React.FC<ForensicAdjudicationMatrixMasterProps> = ({
  onConfirmViolation,
  onDismissAnomaly,
  onReturnToCockpit,
}) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 font-sans">
      {/* ================= QUADRANT 1: TOP-LEFT HERO DARK GRAPHITE CARD ================= */}
      <div className="p-8 rounded-[2rem] bg-[#181716] text-[#FAF7F3] shadow-[0_24px_48px_rgba(0,0,0,0.16)] flex flex-col justify-between space-y-6">
        <div className="space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <span className="text-3xs font-mono font-bold uppercase tracking-widest text-amber-400">
              EXAMINATION INCIDENT SURVEILLANCE
            </span>
            <span className="text-3xs font-mono font-bold px-3 py-1 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30">
              REPLENISHMENT / ESCALATION NEEDED
            </span>
          </div>

          <div>
            <h2 className="text-3xl font-extrabold font-heading tracking-tight text-white">
              SEAT A06 — CANDIDATE #6
            </h2>
            <p className="text-xs text-neutral-400 font-mono mt-1">
              Unauthorized mobile device in lap zone • North Wall Desk / Station A06
            </p>
          </div>
        </div>

        {/* 3 Dark Inset Metric Cards */}
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-3xs font-mono uppercase text-neutral-400">FEEDS SYNCED</div>
            <div className="font-heading font-extrabold text-2xl text-white font-mono">2</div>
          </div>

          <div className="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-3xs font-mono uppercase text-neutral-400">PEAK RISK</div>
            <div className="font-heading font-extrabold text-2xl text-rose-400 font-mono">84%</div>
          </div>

          <div className="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-3xs font-mono uppercase text-neutral-400">TRIAGE SLA</div>
            <div className="font-heading font-extrabold text-2xl text-emerald-400 font-mono">24s</div>
          </div>
        </div>

        <div className="text-3xs font-mono text-neutral-500 pt-2 border-t border-white/10">
          Forensic evidence timeline locked under SHA-256 immutable session digest CBT-A01.
        </div>
      </div>

      {/* ================= QUADRANT 2: TOP-RIGHT PHYSICAL WORKFLOW STEPS ================= */}
      <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] flex flex-col justify-between space-y-6">
        <div>
          <div className="flex items-center justify-between">
            <span className="text-3xs font-mono font-bold uppercase tracking-widest text-neutral-400">
              PHYSICAL IN-SITU LOGISTICS
            </span>
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-100">
              <Truck size={16} />
            </div>
          </div>
          <h3 className="font-heading font-extrabold text-2xl text-neutral-950 tracking-tight mt-1">
            Invigilator Verification Workflow
          </h3>
        </div>

        {/* 4 Step Cards in 2x2 Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          <div className="p-4 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 flex items-center gap-3.5">
            <div className="w-8 h-8 rounded-full bg-rose-100 text-rose-700 font-mono font-bold text-xs flex items-center justify-center shrink-0">
              1
            </div>
            <div>
              <div className="font-bold text-xs text-neutral-950">Aisle Proximity</div>
              <div className="text-3xs text-neutral-500">1.8m from Station A06</div>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 flex items-center gap-3.5">
            <div className="w-8 h-8 rounded-full bg-rose-100 text-rose-700 font-mono font-bold text-xs flex items-center justify-center shrink-0">
              2
            </div>
            <div>
              <div className="font-bold text-xs text-neutral-950">Multi-Cam Verified</div>
              <div className="text-3xs text-neutral-500">CAM-01 + CAM-02 Sync</div>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 flex items-center gap-3.5">
            <div className="w-8 h-8 rounded-full bg-neutral-200 text-neutral-700 font-mono font-bold text-xs flex items-center justify-center shrink-0">
              3
            </div>
            <div>
              <div className="font-bold text-xs text-neutral-950">Desk Intercept</div>
              <div className="text-3xs text-neutral-500">Proctor physical check</div>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 flex items-center gap-3.5">
            <div className="w-8 h-8 rounded-full bg-neutral-200 text-neutral-700 font-mono font-bold text-xs flex items-center justify-center shrink-0">
              4
            </div>
            <div>
              <div className="font-bold text-xs text-neutral-950">Council Handoff</div>
              <div className="text-3xs text-neutral-500">Dossier #C-024 Sealed</div>
            </div>
          </div>
        </div>
      </div>

      {/* ================= QUADRANT 3: BOTTOM-LEFT KINEMATIC SENSOR MATRIX ================= */}
      <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6">
        <div className="flex items-start justify-between flex-wrap gap-2">
          <div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Kinematic Pose & Spatial Allocation
            </h3>
            <p className="text-xs text-neutral-500 mt-0.5">
              2-D COCO keypoints, head-direction proxy, and candidate-relative wrist geometry.
            </p>
          </div>
          <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-800 border border-rose-200">
            UNAVAILABLE FOR EXEMPTION
          </span>
        </div>

        {/* 4 Inset Data Blocks */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          <div className="p-4.5 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 space-y-1">
            <div className="text-3xs font-mono uppercase text-neutral-500">Head Direction Evidence</div>
            <div className="font-mono font-bold text-sm text-neutral-950">2-D Proxy · Review Required</div>
          </div>

          <div className="p-4.5 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 space-y-1">
            <div className="text-3xs font-mono uppercase text-neutral-500">Wrist / Hip Relation</div>
            <div className="font-mono font-bold text-sm text-neutral-950">Pose-Relative · Normalized</div>
          </div>

          <div className="p-4.5 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 space-y-1">
            <div className="text-3xs font-mono uppercase text-neutral-500">Continuous Duration</div>
            <div className="font-mono font-bold text-sm text-neutral-950">Read from Event JSONL</div>
          </div>

          <div className="p-4.5 rounded-2xl bg-[#F8F9FA] border border-neutral-200/60 space-y-1">
            <div className="text-3xs font-mono uppercase text-neutral-500">Counter-Evidence Mask</div>
            <div className="font-mono font-bold text-sm text-neutral-950">Preserved for Reviewer</div>
          </div>
        </div>
      </div>

      {/* ================= QUADRANT 4: BOTTOM-RIGHT ADJUDICATION CONTROL ================= */}
      <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] flex flex-col justify-between space-y-6">
        <div>
          <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
            Adjudication & Legal Disciplinary Control
          </h3>
          <p className="text-xs text-neutral-500 mt-0.5">
            Links this finding to Institutional Case Record and pending proctor escalation order.
          </p>
        </div>

        {/* High-Consequence Red Alert Box with Black CTA Button */}
        <div className="p-5 rounded-2xl border border-rose-200 bg-rose-50/40 space-y-3.5">
          <div className="flex items-center gap-2 text-xs font-bold text-rose-900 font-heading">
            <AlertTriangle size={15} className="text-rose-600" />
            <span>High-Consequence Examination Action</span>
          </div>

          <p className="text-2xs text-rose-950 leading-relaxed font-sans font-medium">
            Confirming this violation locks the 4-camera video replay into an immutable SHA-256 dossier and notifies the examination superintendent.
          </p>

          <Button
            variant="default"
            size="default"
            onClick={() => {
              playHapticSound('snap');
              onConfirmViolation?.();
            }}
            className="w-full bg-neutral-950 hover:bg-neutral-900 text-white rounded-xl py-3 font-semibold text-xs shadow-sm"
          >
            Confirm Violation & Seal Cryptographic Audit
          </Button>
        </div>

        {/* Back Link */}
        <button
          onClick={() => {
            playHapticSound('click');
            onReturnToCockpit?.();
          }}
          className="text-xs text-neutral-500 hover:text-neutral-950 transition-colors font-medium text-left cursor-pointer"
        >
          ← Return to Surveillance Cockpit
        </button>
      </div>
    </div>
  );
};
