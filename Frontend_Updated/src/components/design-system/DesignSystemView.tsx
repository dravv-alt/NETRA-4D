import React, { useState } from 'react';
import { motion } from 'motion/react';
import {
  Layers,
  Sparkles,
  Shield,
  Search,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ArrowRight,
  Download,
  Terminal,
  Activity,
  Sliders,
  Maximize2,
  Lock,
  Cpu,
  Eye,
  Smartphone,
  Users
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';

export const DesignSystemView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'typography' | 'surfaces' | 'semantics' | 'controls' | 'tables'>('typography');

  return (
    <div className="space-y-10 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      {/* 1. LIVING SPECIFICATION HEADER */}
      <div className="pb-6 border-b border-neutral-200/80">
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-neutral-500 mb-2">
          <Terminal size={13} className="text-neutral-700" />
          <span>MANUFACTURE FLOW SPECIFICATION • DESIGN SYSTEM 2.0</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl sm:text-4xl font-extrabold font-heading text-neutral-950 tracking-tight">
              Industrial Clarity & Executive Restraint
            </h1>
            <p className="text-neutral-600 text-sm mt-1.5 max-w-2xl leading-relaxed">
              Complete visual grammar foundation: warm near-white canvas, graphite structure, 2rem executive surfaces, IBM Plex typography, and sparse semantic colors.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3.5 py-1.5 rounded-full bg-neutral-900 text-white font-mono text-xs font-semibold">
              SPEC: V2.4.0
            </span>
          </div>
        </div>
      </div>

      {/* 2. NAVIGATION PILLS */}
      <div className="flex items-center gap-2 border-b border-neutral-200/80 pb-4 overflow-x-auto">
        {[
          { id: 'typography', label: '01. Typography & Hierarchy' },
          { id: 'surfaces', label: '02. Surfaces & Command Cards' },
          { id: 'semantics', label: '03. Semantic Color Matrix' },
          { id: 'controls', label: '04. Controls & Buttons' },
          { id: 'tables', label: '05. Operational Tables & Feeds' },
        ].map((tab) => {
          const isSelected = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => {
                playHapticSound('click');
                setActiveTab(tab.id as any);
              }}
              className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer whitespace-nowrap ${
                isSelected
                  ? 'bg-neutral-900 text-white shadow-xs'
                  : 'text-neutral-600 hover:text-neutral-950 hover:bg-neutral-100'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* SECTION 1: TYPOGRAPHY */}
      {activeTab === 'typography' && (
        <div className="space-y-8">
          <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6">
            <div className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-bold">
              Typography Matrix (Manrope + IBM Plex Sans + IBM Plex Mono)
            </div>

            <div className="space-y-6 divide-y divide-neutral-100">
              {/* Display Large */}
              <div className="pt-4 grid grid-cols-12 gap-4 items-baseline">
                <div className="col-span-12 md:col-span-4 font-mono text-xs text-neutral-500">
                  Display Large (Manrope 44px / 600 / -0.03em)
                </div>
                <div className="col-span-12 md:col-span-8 font-heading text-4xl sm:text-5xl font-extrabold text-neutral-950 tracking-tight">
                  Recover Terminal A06 with Multi-Modal Telemetry
                </div>
              </div>

              {/* Headline Large */}
              <div className="pt-4 grid grid-cols-12 gap-4 items-baseline">
                <div className="col-span-12 md:col-span-4 font-mono text-xs text-neutral-500">
                  Headline Large (Manrope 32px / 600 / -0.02em)
                </div>
                <div className="col-span-12 md:col-span-8 font-heading text-2xl sm:text-3xl font-bold text-neutral-950 tracking-tight">
                  Autonomous Anomaly Verification & Triage
                </div>
              </div>

              {/* Title Medium */}
              <div className="pt-4 grid grid-cols-12 gap-4 items-baseline">
                <div className="col-span-12 md:col-span-4 font-mono text-xs text-neutral-500">
                  Title Medium (Manrope 18px / 600)
                </div>
                <div className="col-span-12 md:col-span-8 font-heading text-lg font-bold text-neutral-950">
                  Multi-Camera Spatial Homography & Seat Alignment
                </div>
              </div>

              {/* Body Operational */}
              <div className="pt-4 grid grid-cols-12 gap-4 items-baseline">
                <div className="col-span-12 md:col-span-4 font-mono text-xs text-neutral-500">
                  Body Operational (IBM Plex Sans 15px / 400 / 1.6)
                </div>
                <div className="col-span-12 md:col-span-8 text-neutral-700 text-sm leading-relaxed max-w-2xl">
                  Exam hall spatial coordinates dynamically correlate 2D bounding boxes across four synchronized RTSP camera feeds to enforce invigilator observation certainty.
                </div>
              </div>

              {/* Technical Kicker & Telemetry */}
              <div className="pt-4 grid grid-cols-12 gap-4 items-baseline">
                <div className="col-span-12 md:col-span-4 font-mono text-xs text-neutral-500">
                  Technical Identifier & Telemetry (IBM Plex Mono)
                </div>
                <div className="col-span-12 md:col-span-8 flex items-center gap-4 flex-wrap">
                  <span className="font-mono text-xs font-bold text-neutral-900 bg-neutral-100 px-3 py-1 rounded-lg border border-neutral-200">
                    TERMINAL_ID: A06-NORTH
                  </span>
                  <span className="font-mono text-xs font-bold text-neutral-900 bg-neutral-100 px-3 py-1 rounded-lg border border-neutral-200">
                    LATENCY: 34.7ms
                  </span>
                  <span className="font-mono text-xs font-bold text-emerald-800 bg-emerald-50 px-3 py-1 rounded-lg border border-emerald-200">
                    CONFIDENCE: 94.2%
                  </span>
                  <span className="font-mono text-xs font-bold text-neutral-600">
                    15:42:18.420 IST
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 2: SURFACES & COMMAND CARDS */}
      {activeTab === 'surfaces' && (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
            {/* 1. Executive White Surface (32px / 2rem Radius) */}
            <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] flex flex-col justify-between space-y-6">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-neutral-200/80">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
                      <Layers size={16} />
                    </div>
                    <span className="font-heading font-bold text-base text-neutral-950">
                      Executive Dashboard Surface
                    </span>
                  </div>
                  <span className="text-xs font-mono font-semibold text-neutral-500">2rem / 32px Radius</span>
                </div>
                <p className="text-xs text-neutral-600 leading-relaxed mt-4">
                  White executive card architecture with wide, ultra-soft low-opacity drop shadows and hairline neutral borders. Used for primary operational cockpits and high-consequence dashboards.
                </p>
              </div>

              <div className="p-4 rounded-2xl bg-neutral-50 border border-neutral-200/80 flex items-center justify-between">
                <span className="text-xs text-neutral-600 font-medium">Elevation Signature</span>
                <span className="text-xs font-mono font-bold text-neutral-900">0 20px 40px rgba(0,0,0,0.035)</span>
              </div>
            </div>

            {/* 2. Strategic Dark Command Surface */}
            <div className="p-8 rounded-[2rem] bg-[#242323] text-[#FAF7F3] border border-white/10 shadow-[0_24px_48px_rgba(0,0,0,0.14)] flex flex-col justify-between space-y-6">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-white/10">
                  <div className="flex items-center gap-2.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
                    <span className="font-mono text-xs font-bold uppercase tracking-widest text-amber-300">
                      LIVE COMMAND SURFACE
                    </span>
                  </div>
                  <span className="text-xs font-mono text-neutral-400">#242323 Graphite</span>
                </div>
                <div className="mt-4 space-y-2">
                  <h3 className="text-xl font-heading font-bold text-white tracking-tight">
                    Intercept Signal: Terminal A06
                  </h3>
                  <p className="text-xs text-neutral-300 leading-relaxed">
                    High-priority AI detection cue below laptop plane sustained over 6.2s. Strategic dark emphasis card turns complex telemetry into an actionable decision cockpit.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-3 border-t border-white/10">
                <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                  <span className="text-2xs font-mono uppercase text-neutral-400">Phone Score</span>
                  <div className="text-base font-mono font-bold text-rose-300 mt-0.5">0.74 (74%)</div>
                </div>
                <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                  <span className="text-2xs font-mono uppercase text-neutral-400">Duration</span>
                  <div className="text-base font-mono font-bold text-white mt-0.5">6.2s Active</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 3: SEMANTIC COLORS */}
      {activeTab === 'semantics' && (
        <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6">
          <div className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-bold">
            Controlled Semantic Palette (Color = Information Meaning)
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Controlled Green */}
            <div className="p-5 rounded-2xl bg-emerald-50/60 border border-emerald-200 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-emerald-900">Success / Nominal</span>
                <CheckCircle2 size={16} className="text-emerald-700" />
              </div>
              <div className="text-xs text-emerald-800 leading-relaxed">
                Nominal laptop posture, verified ground truth, healthy RTSP streaming nodes.
              </div>
              <StatusBadge status="normal" label="NOMINAL" />
            </div>

            {/* Controlled Amber */}
            <div className="p-5 rounded-2xl bg-amber-50/60 border border-amber-200 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-amber-900">Warning / Review</span>
                <AlertTriangle size={16} className="text-amber-700" />
              </div>
              <div className="text-xs text-amber-800 leading-relaxed">
                Elevated gaze deflection, off-keyboard wrist transition, pending invigilator triage.
              </div>
              <StatusBadge status="review" label="REVIEW" />
            </div>

            {/* Controlled Rose */}
            <div className="p-5 rounded-2xl bg-rose-50/60 border border-rose-200 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-rose-900">Critical / Alert</span>
                <XCircle size={16} className="text-rose-700" />
              </div>
              <div className="text-xs text-rose-800 leading-relaxed">
                Confirmed physical sheet exchange, unauthorized phone candidate, auto-escalation.
              </div>
              <StatusBadge status="escalated" label="ESCALATED" />
            </div>

            {/* Graphite Primary */}
            <div className="p-5 rounded-2xl bg-neutral-100/70 border border-neutral-300 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-neutral-950">Neutral / Command</span>
                <Lock size={16} className="text-neutral-900" />
              </div>
              <div className="text-xs text-neutral-700 leading-relaxed">
                Primary actions, high-contrast typography, baseline architectural frame.
              </div>
              <span className="px-3 py-1 rounded-full bg-neutral-900 text-white font-mono text-xs font-bold">
                COMMAND
              </span>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 4: CONTROLS & BUTTONS */}
      {activeTab === 'controls' && (
        <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6">
          <div className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-bold">
            Button & Input Primitives
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <Button variant="default" size="default">
              Primary Action (Graphite)
            </Button>
            <Button variant="secondary" size="default">
              Secondary Button
            </Button>
            <Button variant="outline" size="default">
              Outline Control
            </Button>
            <Button variant="ghost" size="default">
              Ghost Link
            </Button>
            <Button variant="destructive" size="default">
              Destructive Action
            </Button>
          </div>

          <div className="pt-4 border-t border-neutral-200/80 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="relative">
              <Search size={16} className="absolute left-4 top-3.5 text-neutral-400" />
              <input
                type="text"
                placeholder="Search candidate ID, terminal, or signal..."
                className="w-full bg-white border border-neutral-300 rounded-full pl-11 pr-4 py-3 text-xs text-neutral-900 placeholder-neutral-400 font-sans focus:outline-none focus:border-neutral-900 shadow-2xs"
              />
            </div>
            <div className="flex items-center gap-2">
              <select className="w-full bg-white border border-neutral-300 rounded-2xl px-4 py-3 text-xs text-neutral-900 font-sans focus:outline-none focus:border-neutral-900 shadow-2xs font-semibold">
                <option>Filter by Severity: All Signals</option>
                <option>Escalated (Critical)</option>
                <option>Needs Review</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 5: OPERATIONAL TABLES */}
      {activeTab === 'tables' && (
        <div className="bg-white rounded-[2rem] border border-neutral-200/80 overflow-hidden shadow-[0_20px_40px_rgba(0,0,0,0.035)]">
          <div className="p-6 bg-neutral-50/80 border-b border-neutral-200/80 flex items-center justify-between">
            <div className="font-heading font-bold text-base text-neutral-950">
              High-Density Operational Table Pattern
            </div>
            <span className="text-xs font-mono text-neutral-500">Monospace Telemetry Alignment</span>
          </div>

          <table className="w-full text-left text-xs">
            <thead className="bg-neutral-50/50 text-neutral-500 uppercase border-b border-neutral-200 font-semibold text-2xs tracking-wider font-mono">
              <tr>
                <th className="py-3.5 px-6">Terminal ID</th>
                <th className="py-3.5 px-6">Candidate</th>
                <th className="py-3.5 px-6">Kinematic Status</th>
                <th className="py-3.5 px-6">Continuity</th>
                <th className="py-3.5 px-6">Confidence</th>
                <th className="py-3.5 px-6">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200/80">
              {[
                { seat: 'A06', cand: 'Candidate #6', status: 'review', cont: '100%', conf: '74%' },
                { seat: 'B04', cand: 'Candidate #14', status: 'review', cont: '98%', conf: '81%' },
                { seat: 'B08', cand: 'Candidate #18', status: 'escalated', cont: '96%', conf: '88%' },
                { seat: 'A01', cand: 'Candidate #1', status: 'normal', cont: '100%', conf: '12%' },
              ].map((row, idx) => (
                <tr key={idx} className="hover:bg-neutral-50/80 transition-colors">
                  <td className="py-4 px-6 font-mono font-bold text-neutral-950">{row.seat}</td>
                  <td className="py-4 px-6 font-semibold text-neutral-900">{row.cand}</td>
                  <td className="py-4 px-6"><StatusBadge status={row.status as any} size="sm" /></td>
                  <td className="py-4 px-6 font-mono text-emerald-700 font-bold">{row.cont}</td>
                  <td className="py-4 px-6 font-mono font-bold">{row.conf}</td>
                  <td className="py-4 px-6">
                    <Button variant="outline" size="xs" className="rounded-lg">
                      Inspect
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
