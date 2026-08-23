import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SystemHealthData, TriagedAIEvent } from '../../types';
import {
  LayoutGrid,
  Video,
  ListFilter,
  PlaySquare,
  Users,
  FileCheck,
  BarChart3,
  Cpu,
  FlaskConical,
  Activity,
  ShieldCheck,
  ChevronDown,
  Sparkles,
  Radio,
  SlidersHorizontal,
  Clock,
  Layers,
  Terminal,
  PanelBottom,
  Search,
  Zap,
  Compass,
  AlertCircle,
  Eye,
  Shield
} from 'lucide-react';
import { SystemHealthModal } from './SystemHealthModal';
import { MagneticDock, DockItemData } from '../ui/magnetic-dock';
import { playHapticSound } from '../../lib/utils';

export type NavigationTab =
  | 'command-center'
  | 'live-monitor'
  | 'event-queue'
  | 'evidence-review'
  | 'candidates'
  | 'cases'
  | 'analytics'
  | 'system-cameras'
  | 'benchmark'
  | 'design-system';

interface AppShellProps {
  activeTab: NavigationTab | string;
  onSelectTab: (tabId: any) => void;
  health: SystemHealthData;
  eventsCount: {
    total: number;
    needsReview: number;
    escalated?: number;
  };
  onSelectDemoPreset: (presetId: string) => void;
  activeEventForReview?: TriagedAIEvent;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  activeTab,
  onSelectTab,
  health,
  eventsCount,
  onSelectDemoPreset,
  children,
}) => {
  const [isHealthOpen, setIsHealthOpen] = useState(false);
  const [demoMenuOpen, setDemoMenuOpen] = useState(false);
  const [showFloatingDock, setShowFloatingDock] = useState(false);
  const [isSidebarHovered, setIsSidebarHovered] = useState(false);

  const navOperations = [
    {
      id: 'command-center',
      label: 'Command Center',
      icon: <LayoutGrid size={16} />,
      badge: undefined,
    },
    {
      id: 'live-monitor',
      label: 'Live Monitor',
      icon: <Video size={16} />,
      badge: undefined,
    },
    {
      id: 'event-queue',
      label: 'Event Queue',
      icon: <ListFilter size={16} />,
      badge: eventsCount.needsReview > 0 ? String(eventsCount.needsReview) : undefined,
    },
    {
      id: 'evidence-review',
      label: 'Evidence Review',
      icon: <PlaySquare size={16} />,
      badge: undefined,
    },
  ];

  const navIntelligence = [
    {
      id: 'candidates',
      label: 'Candidates',
      icon: <Users size={16} />,
      badge: undefined,
    },
    {
      id: 'cases',
      label: 'Cases & Audit',
      icon: <FileCheck size={16} />,
      badge: undefined,
    },
    {
      id: 'analytics',
      label: 'Analytics & Radar',
      icon: <BarChart3 size={16} />,
      badge: undefined,
    },
    {
      id: 'system-cameras',
      label: 'AI & Cameras',
      icon: <Cpu size={16} />,
      badge: undefined,
    },
    {
      id: 'benchmark',
      label: 'Benchmark Lab',
      icon: <FlaskConical size={16} />,
      badge: undefined,
    },
    {
      id: 'design-system',
      label: 'Design Tokens',
      icon: <Layers size={16} />,
      badge: undefined,
    },
  ];

  const allNavItems = [...navOperations, ...navIntelligence];

  const dockItems: DockItemData[] = allNavItems.map((item) => ({
    id: item.id,
    label: item.label,
    icon: item.icon,
    isActive: activeTab === item.id,
    badge: item.badge ? Number(item.badge) || undefined : undefined,
    onClick: () => onSelectTab(item.id),
  }));

  const demoPresets = [
    { id: 'preset-phone', title: '1. Phone in Lap (Candidate #6)', seat: 'Seat A06 (North Desk)', type: 'Review' },
    { id: 'preset-glance', title: '2. Turning Back to Aisle (Candidate #14)', seat: 'Seat B04 (South Desk)', type: 'Review' },
    { id: 'preset-counter', title: '3. Wall Timer Look (Candidate #3)', seat: 'Seat A03 (Suppressed)', type: 'Observe' },
    { id: 'preset-hard-negative', title: '4. Mouse Hard-Negative Check', seat: 'Seat A04 (Suppressed)', type: 'Observe' },
    { id: 'preset-peer', title: '5. Peer Paper Transfer (Candidate #18)', seat: 'Seat B08 & B09', type: 'Escalated' },
    { id: 'preset-benchmark', title: '6. Multi-Provider Benchmarks', seat: '8 CCTV Footage Clips', type: 'Lab' },
  ];

  return (
    <div className="flex h-screen w-screen bg-[#FAF9F6] text-neutral-900 overflow-hidden select-none font-sans">
      {/* FLOATING PILL SIDEBAR THAT EXPANDS ON HOVER */}
      <aside
        onMouseEnter={() => setIsSidebarHovered(true)}
        onMouseLeave={() => setIsSidebarHovered(false)}
        className={`fixed left-4 top-4 bottom-4 z-50 transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] flex flex-col justify-between bg-white/95 backdrop-blur-2xl border border-neutral-200/90 shadow-[0_20px_40px_rgba(0,0,0,0.06)] overflow-hidden ${
          isSidebarHovered
            ? 'w-64 rounded-[2rem] shadow-[0_24px_64px_rgba(0,0,0,0.12)] p-4.5'
            : 'w-16 rounded-[2.25rem] p-2.5 items-center'
        }`}
      >
        {/* Top: Brand Emblem & Title */}
        <div className="w-full">
          <div className={`flex items-center gap-3 ${!isSidebarHovered ? 'justify-center' : 'px-1'}`}>
            <div className="w-10 h-10 rounded-2xl bg-[#181716] text-white flex items-center justify-center font-heading font-extrabold text-sm shadow-sm shrink-0 ring-4 ring-neutral-100">
              <Shield size={18} className="text-white" />
            </div>
            {isSidebarHovered && (
              <motion.div
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className="overflow-hidden min-w-0"
              >
                <div className="font-heading font-extrabold tracking-tight text-sm text-neutral-950 flex items-center gap-2">
                  <span>DRISHTI</span>
                  <span className="text-[10px] font-mono font-bold px-2 py-0.5 bg-neutral-100 text-neutral-700 border border-neutral-200 rounded-full">
                    v2.4
                  </span>
                </div>
                <div className="text-xs text-neutral-500 font-sans truncate">
                  Autonomous Surveillance AI
                </div>
              </motion.div>
            )}
          </div>
        </div>

        {/* Middle: Clean Categorized Navigation */}
        <div className={`w-full overflow-y-auto flex-1 my-3 scrollbar-none ${isSidebarHovered ? 'px-0.5 space-y-4' : 'flex flex-col items-center space-y-1.5'}`}>
          {/* Group 1: Live Operations */}
          <div className="space-y-1 w-full">
            {isSidebarHovered && (
              <div className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider px-3 pb-1 font-sans">
                Live Operations
              </div>
            )}

            {navOperations.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <motion.button
                  key={item.id}
                  onClick={() => {
                    playHapticSound('click');
                    onSelectTab(item.id);
                  }}
                  whileTap={{ scale: 0.96 }}
                  className={`relative flex items-center transition-all duration-150 cursor-pointer ${
                    isSidebarHovered
                      ? 'w-full justify-between px-3 py-2 rounded-xl text-left'
                      : 'w-10 h-10 justify-center rounded-2xl'
                  } ${
                    isActive
                      ? isSidebarHovered
                        ? 'bg-[#181716] text-white shadow-xs font-semibold'
                        : 'bg-[#181716] text-white shadow-md'
                      : 'text-neutral-600 hover:text-neutral-950 hover:bg-neutral-100 font-medium'
                  }`}
                  title={!isSidebarHovered ? item.label : undefined}
                >
                  <div className="relative z-10 flex items-center gap-3">
                    <span className={isActive ? 'text-white' : 'text-neutral-500'}>
                      {item.icon}
                    </span>
                    {isSidebarHovered && (
                      <span className="truncate text-xs font-sans">{item.label}</span>
                    )}
                  </div>

                  {/* Expanded Pill Badge */}
                  {isSidebarHovered && item.badge && (
                    <span className="relative z-10 px-2 py-0.5 text-[10px] rounded-full bg-amber-500 text-white font-mono font-bold tabular-nums">
                      {item.badge}
                    </span>
                  )}

                  {/* Collapsed Mini Badge */}
                  {!isSidebarHovered && item.badge && (
                    <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-amber-500 text-white font-mono text-[9px] font-bold flex items-center justify-center shadow-xs ring-2 ring-white">
                      {item.badge}
                    </span>
                  )}
                </motion.button>
              );
            })}
          </div>

          {/* Hairline Divider in Collapsed Mode */}
          {!isSidebarHovered && (
            <div className="w-6 h-[1px] bg-neutral-200 my-1 rounded-full shrink-0" />
          )}

          {/* Group 2: Audit & Intelligence */}
          <div className="space-y-1 w-full">
            {isSidebarHovered && (
              <div className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider px-3 pb-1 pt-1 font-sans">
                Audit & Intelligence
              </div>
            )}

            {navIntelligence.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <motion.button
                  key={item.id}
                  onClick={() => {
                    playHapticSound('click');
                    onSelectTab(item.id);
                  }}
                  whileTap={{ scale: 0.96 }}
                  className={`relative flex items-center transition-all duration-150 cursor-pointer ${
                    isSidebarHovered
                      ? 'w-full justify-between px-3 py-2 rounded-xl text-left'
                      : 'w-10 h-10 justify-center rounded-2xl'
                  } ${
                    isActive
                      ? isSidebarHovered
                        ? 'bg-[#181716] text-white shadow-xs font-semibold'
                        : 'bg-[#181716] text-white shadow-md'
                      : 'text-neutral-600 hover:text-neutral-950 hover:bg-neutral-100 font-medium'
                  }`}
                  title={!isSidebarHovered ? item.label : undefined}
                >
                  <div className="relative z-10 flex items-center gap-3">
                    <span className={isActive ? 'text-white' : 'text-neutral-500'}>
                      {item.icon}
                    </span>
                    {isSidebarHovered && (
                      <span className="truncate text-xs font-sans">{item.label}</span>
                    )}
                  </div>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Bottom: Professional Hardware Status Card */}
        {isSidebarHovered ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="w-full p-3.5 bg-neutral-50 rounded-2xl border border-neutral-200/80 text-xs font-sans space-y-2"
          >
            <div className="flex items-center justify-between text-xs text-neutral-700">
              <span className="flex items-center gap-2 text-emerald-800 font-semibold font-sans">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Pipeline Synced
              </span>
              <span className="font-mono text-neutral-950 font-bold">{health.pipelineLatencyMs}ms</span>
            </div>
            <button
              onClick={() => {
                playHapticSound('snap');
                setIsHealthOpen(true);
              }}
              className="w-full py-2 px-3 rounded-xl bg-white hover:bg-neutral-100 border border-neutral-200/80 text-neutral-900 text-xs font-sans font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-2xs"
            >
              <Cpu size={14} className="text-neutral-700" />
              <span>NVIDIA T4 Diagnostics</span>
            </button>
          </motion.div>
        ) : (
          <div className="flex flex-col items-center">
            <button
              onClick={() => {
                playHapticSound('snap');
                setIsHealthOpen(true);
              }}
              className="w-10 h-10 rounded-2xl bg-neutral-100 hover:bg-neutral-200 text-neutral-700 flex items-center justify-center cursor-pointer transition-colors relative"
              title="NVIDIA T4 GPU Diagnostics"
            >
              <Cpu size={16} />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-emerald-500 ring-2 ring-white animate-pulse" />
            </button>
          </div>
        )}
      </aside>

      {/* MAIN VIEW AREA */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#FAF9F6] relative pl-24 sm:pl-28">
        {/* TOP STATUS BAR */}
        <header className="h-16 px-8 bg-white/90 backdrop-blur-md border-b border-neutral-200/80 flex items-center justify-between z-10 shrink-0">
          <div className="flex items-center gap-4 sm:gap-6">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-semibold shadow-2xs font-sans">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0" />
              <span>Surveillance Grid Synchronized</span>
            </div>
            <div className="hidden md:flex items-center gap-2 text-xs text-neutral-500 font-mono">
              <span>{health.pipelineLatencyMs}ms DAG Latency</span>
              <span>•</span>
              <span>{health.t4GpuUtilPct}% GPU Load</span>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            {/* Dock Visibility Toggle */}
            <button
              onClick={() => {
                playHapticSound('click');
                setShowFloatingDock(!showFloatingDock);
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-sans font-semibold transition-colors cursor-pointer border ${
                showFloatingDock
                  ? 'bg-neutral-900 text-white border-neutral-900'
                  : 'bg-white hover:bg-neutral-50 text-neutral-700 border-neutral-300 shadow-2xs'
              }`}
              title="Toggle Floating Dock"
            >
              <PanelBottom size={13} />
              <span className="hidden sm:inline">Dock</span>
            </button>

            {/* Demo Presets Dropdown */}
            <div className="relative">
              <button
                onClick={() => {
                  playHapticSound('click');
                  setDemoMenuOpen(!demoMenuOpen);
                }}
                className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl border border-neutral-300 bg-white hover:bg-neutral-50 text-xs font-semibold text-neutral-900 shadow-2xs cursor-pointer transition-colors font-sans"
              >
                <Sparkles size={13} className="text-amber-500" />
                <span>Story Mode / Presets</span>
                <ChevronDown size={13} className="text-neutral-500" />
              </button>

              <AnimatePresence>
                {demoMenuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 6, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 6, scale: 0.98 }}
                    className="absolute right-0 top-full mt-2 w-80 bg-white rounded-2xl border border-neutral-200 shadow-xl p-2 z-50 space-y-1 font-sans"
                  >
                    <div className="p-2 text-2xs font-mono uppercase text-neutral-400 font-bold border-b border-neutral-100">
                      Story Mode Investigation Presets
                    </div>
                    {demoPresets.map((preset) => (
                      <button
                        key={preset.id}
                        onClick={() => {
                          playHapticSound('snap');
                          onSelectDemoPreset(preset.id);
                          setDemoMenuOpen(false);
                        }}
                        className="w-full p-2.5 rounded-xl text-left hover:bg-neutral-50 flex items-start justify-between cursor-pointer text-xs group transition-colors"
                      >
                        <div>
                          <div className="font-semibold text-neutral-950 group-hover:text-black">
                            {preset.title}
                          </div>
                          <div className="text-2xs text-neutral-500 mt-0.5">
                            {preset.seat}
                          </div>
                        </div>
                        <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded-md bg-neutral-100 text-neutral-700 border border-neutral-200 shrink-0 ml-2">
                          {preset.type}
                        </span>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Health Diagnostics Modal Button */}
            <button
              onClick={() => {
                playHapticSound('snap');
                setIsHealthOpen(true);
              }}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-neutral-100 hover:bg-neutral-200 text-xs font-mono font-semibold text-neutral-800 border border-neutral-200 transition-colors cursor-pointer"
            >
              <Cpu size={13} />
              <span>NVIDIA T4</span>
            </button>
          </div>
        </header>

        {/* SCROLLABLE MAIN CANVAS CONTAINER */}
        <main className="flex-1 overflow-y-auto px-6 sm:px-10 py-10 pb-36 max-w-7xl w-full mx-auto">
          {children}
        </main>

        {/* FLOATING MAGNETIC DOCK (TOGGLEABLE) */}
        {showFloatingDock && (
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40">
            <MagneticDock items={dockItems} iconSize={42} />
          </div>
        )}
      </div>

      {/* SYSTEM HEALTH DIAGNOSTICS MODAL */}
      <SystemHealthModal
        isOpen={isHealthOpen}
        onClose={() => setIsHealthOpen(false)}
        health={health}
      />
    </div>
  );
};
