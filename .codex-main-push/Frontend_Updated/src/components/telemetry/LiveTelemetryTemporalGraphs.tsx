import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Activity,
  Play,
  Pause,
  Clock,
  Smartphone,
  Eye,
  Users,
  Hand,
  TrendingUp,
  Sliders,
  ChevronDown,
  Sparkles
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';

export interface TelemetryPoint {
  timeSec: number;
  timestamp: string;
  phoneScore: number;
  gazeScore: number;
  peerScore: number;
  handScore: number;
  aggregateScore: number;
}

interface LiveTelemetryTemporalGraphsProps {
  selectedSeatId?: string | null;
  onSelectCandidate?: (seatId: string) => void;
}

export const LiveTelemetryTemporalGraphs: React.FC<LiveTelemetryTemporalGraphsProps> = ({
  selectedSeatId,
  onSelectCandidate,
}) => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [timeWindow, setTimeWindow] = useState<30 | 60 | 120>(60);
  const [selectedSeat, setSelectedSeat] = useState<string>(selectedSeatId || 'ALL');
  const [hoveredPoint, setHoveredPoint] = useState<TelemetryPoint | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  // Category Toggles
  const [visibleCategories, setVisibleCategories] = useState({
    phone: true,
    gaze: true,
    peer: true,
    hand: true,
    aggregate: true,
  });

  // Generate initial temporal baseline data (60 points)
  const [dataPoints, setDataPoints] = useState<TelemetryPoint[]>(() => {
    const initial: TelemetryPoint[] = [];
    const now = Date.now();
    for (let i = 60; i >= 0; i--) {
      const t = new Date(now - i * 1000);
      const timeStr = t.toTimeString().split(' ')[0];
      const timeSec = 60 - i;
      
      const isEventA = timeSec >= 35 && timeSec <= 48; // Phone spike
      const isEventB = timeSec >= 15 && timeSec <= 26; // Gaze turn spike

      const phone = isEventA ? 0.74 + Math.sin(timeSec) * 0.08 : 0.08 + Math.random() * 0.07;
      const gaze = isEventB ? 0.81 + Math.cos(timeSec) * 0.06 : 0.14 + Math.random() * 0.09;
      const peer = timeSec >= 42 && timeSec <= 50 ? 0.65 + Math.random() * 0.12 : 0.05 + Math.random() * 0.04;
      const hand = isEventA ? 0.68 + Math.random() * 0.09 : 0.12 + Math.random() * 0.06;
      const aggregate = Math.max(phone, gaze, peer, hand) * 0.95;

      initial.push({
        timeSec,
        timestamp: timeStr,
        phoneScore: Number(Math.max(0, Math.min(1, phone)).toFixed(2)),
        gazeScore: Number(Math.max(0, Math.min(1, gaze)).toFixed(2)),
        peerScore: Number(Math.max(0, Math.min(1, peer)).toFixed(2)),
        handScore: Number(Math.max(0, Math.min(1, hand)).toFixed(2)),
        aggregateScore: Number(Math.max(0, Math.min(1, aggregate)).toFixed(2)),
      });
    }
    return initial;
  });

  useEffect(() => {
    if (selectedSeatId) {
      setSelectedSeat(selectedSeatId);
    }
  }, [selectedSeatId]);

  // Live streaming engine: append point every second
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setDataPoints((prev) => {
        const lastSec = prev[prev.length - 1]?.timeSec || 0;
        const newSec = lastSec + 1;
        const now = new Date();
        const timeStr = now.toTimeString().split(' ')[0];

        let phoneBase = 0.08 + Math.random() * 0.06;
        let gazeBase = 0.14 + Math.random() * 0.08;
        let peerBase = 0.06 + Math.random() * 0.05;
        let handBase = 0.10 + Math.random() * 0.06;

        if (selectedSeat === 'A06') {
          phoneBase = 0.74 + Math.sin(newSec * 0.5) * 0.06;
          handBase = 0.68 + Math.cos(newSec * 0.4) * 0.05;
        } else if (selectedSeat === 'B04') {
          gazeBase = 0.81 + Math.sin(newSec * 0.6) * 0.07;
        } else if (selectedSeat === 'B08') {
          peerBase = 0.78 + Math.sin(newSec * 0.4) * 0.09;
        }

        const aggregate = Math.max(phoneBase, gazeBase, peerBase, handBase) * 0.95;

        const newPoint: TelemetryPoint = {
          timeSec: newSec,
          timestamp: timeStr,
          phoneScore: Number(Math.max(0, Math.min(1, phoneBase)).toFixed(2)),
          gazeScore: Number(Math.max(0, Math.min(1, gazeBase)).toFixed(2)),
          peerScore: Number(Math.max(0, Math.min(1, peerBase)).toFixed(2)),
          handScore: Number(Math.max(0, Math.min(1, handBase)).toFixed(2)),
          aggregateScore: Number(Math.max(0, Math.min(1, aggregate)).toFixed(2)),
        };

        return [...prev.slice(1), newPoint];
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isPlaying, selectedSeat, timeWindow]);

  // Main SVG Calculations
  const svgWidth = 840;
  const svgHeight = 220;
  const padding = { top: 20, right: 30, bottom: 35, left: 45 };
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = svgHeight - padding.top - padding.bottom;

  const generatePath = (key: keyof Pick<TelemetryPoint, 'phoneScore' | 'gazeScore' | 'peerScore' | 'handScore' | 'aggregateScore'>) => {
    if (dataPoints.length < 2) return '';

    return dataPoints.reduce((acc, point, index) => {
      const x = padding.left + (index / (dataPoints.length - 1)) * chartWidth;
      const y = padding.top + chartHeight - point[key] * chartHeight;
      return `${acc} ${index === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
    }, '');
  };

  const generateAreaPath = () => {
    if (dataPoints.length < 2) return '';
    const linePath = generatePath('aggregateScore');
    const firstX = padding.left;
    const lastX = padding.left + chartWidth;
    const bottomY = padding.top + chartHeight;
    return `${linePath} L ${lastX},${bottomY} L ${firstX},${bottomY} Z`;
  };

  // Mini Sparkline Generator for individual cards
  const generateMiniSparkline = (key: keyof Pick<TelemetryPoint, 'phoneScore' | 'gazeScore' | 'peerScore' | 'handScore'>, width = 120, height = 36) => {
    const recent = dataPoints.slice(-25);
    if (recent.length < 2) return '';
    return recent.reduce((acc, point, index) => {
      const x = (index / (recent.length - 1)) * width;
      const y = height - point[key] * height * 0.9 - 2;
      return `${acc} ${index === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
    }, '');
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const relativeX = (mouseX / rect.width) * svgWidth;

    if (relativeX >= padding.left && relativeX <= padding.left + chartWidth) {
      setHoverX(relativeX);
      const ratio = (relativeX - padding.left) / chartWidth;
      const index = Math.min(dataPoints.length - 1, Math.max(0, Math.round(ratio * (dataPoints.length - 1))));
      setHoveredPoint(dataPoints[index]);
    }
  };

  const handleMouseLeave = () => {
    setHoverX(null);
    setHoveredPoint(null);
  };

  const latest = dataPoints[dataPoints.length - 1] || {
    phoneScore: 0.12,
    gazeScore: 0.18,
    peerScore: 0.08,
    handScore: 0.14,
    aggregateScore: 0.18,
  };

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-7 font-sans">
      {/* 1. Header & Controls Bar */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
            <Activity size={18} />
          </div>
          <div>
            <div className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight flex items-center gap-3">
              <span>Live Telemetry & Temporal Probability Curves</span>
              <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                1Hz Streaming Sync
              </span>
            </div>
            <p className="text-xs text-neutral-500 mt-0.5">
              Multi-detector rolling confidence timeline and sub-second anomaly probability tracking.
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Candidate Filter Dropdown */}
          <div className="flex items-center gap-2 bg-neutral-50 px-3.5 py-2 rounded-xl border border-neutral-200 text-xs font-mono">
            <span className="text-neutral-500">Terminal:</span>
            <select
              value={selectedSeat}
              onChange={(e) => {
                playHapticSound('click');
                setSelectedSeat(e.target.value);
                if (onSelectCandidate && e.target.value !== 'ALL') {
                  onSelectCandidate(e.target.value);
                }
              }}
              className="bg-transparent font-bold text-neutral-950 focus:outline-none cursor-pointer"
            >
              <option value="ALL">All Cohort (Mean)</option>
              <option value="A06">Seat A06 (Cand #6)</option>
              <option value="B04">Seat B04 (Cand #14)</option>
              <option value="B08">Seat B08 (Cand #18)</option>
              <option value="A03">Seat A03 (Cand #3)</option>
              <option value="A04">Seat A04 (Cand #4)</option>
            </select>
          </div>

          {/* Time Window Switcher */}
          <div className="flex items-center bg-neutral-100 p-1 rounded-xl text-xs font-mono">
            {([30, 60, 120] as const).map((w) => (
              <button
                key={w}
                onClick={() => {
                  playHapticSound('click');
                  setTimeWindow(w);
                }}
                className={`px-3 py-1 rounded-lg transition-all font-semibold cursor-pointer ${
                  timeWindow === w
                    ? 'bg-white text-neutral-950 shadow-2xs'
                    : 'text-neutral-500 hover:text-neutral-900'
                }`}
              >
                {w}s
              </button>
            ))}
          </div>

          {/* Play / Pause Toggle */}
          <Button
            variant={isPlaying ? 'default' : 'secondary'}
            size="sm"
            onClick={() => {
              playHapticSound('snap');
              setIsPlaying(!isPlaying);
            }}
            className="gap-1.5 rounded-xl font-mono text-xs shadow-xs px-3.5"
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            <span>{isPlaying ? 'PAUSE' : 'LIVE'}</span>
          </Button>
        </div>
      </div>

      {/* 2. Category Toggles Strip (Restrained, Unified Light Pastel Tokens) */}
      <div className="flex items-center justify-between text-xs flex-wrap gap-3">
        <div className="flex items-center gap-2 flex-wrap font-mono">
          {/* Phone Toggle */}
          <button
            onClick={() => {
              playHapticSound('click');
              setVisibleCategories((p) => ({ ...p, phone: !p.phone }));
            }}
            className={`px-3 py-1.5 rounded-xl border flex items-center gap-2 transition-all cursor-pointer ${
              visibleCategories.phone
                ? 'bg-rose-50/80 border-rose-300 text-rose-900 font-bold shadow-2xs'
                : 'bg-white border-neutral-200 text-neutral-400 opacity-60'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <span>Phone Detection</span>
            <span className="font-extrabold text-rose-900 ml-1">{(latest.phoneScore * 100).toFixed(0)}%</span>
          </button>

          {/* Gaze Toggle */}
          <button
            onClick={() => {
              playHapticSound('click');
              setVisibleCategories((p) => ({ ...p, gaze: !p.gaze }));
            }}
            className={`px-3 py-1.5 rounded-xl border flex items-center gap-2 transition-all cursor-pointer ${
              visibleCategories.gaze
                ? 'bg-amber-50/80 border-amber-300 text-amber-900 font-bold shadow-2xs'
                : 'bg-white border-neutral-200 text-neutral-400 opacity-60'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <span>Gaze / Yaw Turn</span>
            <span className="font-extrabold text-amber-900 ml-1">{(latest.gazeScore * 100).toFixed(0)}%</span>
          </button>

          {/* Peer Toggle */}
          <button
            onClick={() => {
              playHapticSound('click');
              setVisibleCategories((p) => ({ ...p, peer: !p.peer }));
            }}
            className={`px-3 py-1.5 rounded-xl border flex items-center gap-2 transition-all cursor-pointer ${
              visibleCategories.peer
                ? 'bg-violet-50/80 border-violet-300 text-violet-900 font-bold shadow-2xs'
                : 'bg-white border-neutral-200 text-neutral-400 opacity-60'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-violet-500" />
            <span>Peer Collaboration</span>
            <span className="font-extrabold text-violet-900 ml-1">{(latest.peerScore * 100).toFixed(0)}%</span>
          </button>

          {/* Hand Toggle */}
          <button
            onClick={() => {
              playHapticSound('click');
              setVisibleCategories((p) => ({ ...p, hand: !p.hand }));
            }}
            className={`px-3 py-1.5 rounded-xl border flex items-center gap-2 transition-all cursor-pointer ${
              visibleCategories.hand
                ? 'bg-sky-50/80 border-sky-300 text-sky-900 font-bold shadow-2xs'
                : 'bg-white border-neutral-200 text-neutral-400 opacity-60'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-sky-500" />
            <span>Off-Keyboard Hands</span>
            <span className="font-extrabold text-sky-900 ml-1">{(latest.handScore * 100).toFixed(0)}%</span>
          </button>
        </div>

        {/* Aggregated Indicator */}
        <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-neutral-900 text-white font-mono text-xs">
          <span className="text-neutral-400">Composite Index:</span>
          <span className="text-emerald-400 font-bold">{(latest.aggregateScore * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* 3. Main SVG Telemetry Chart */}
      <div className="relative bg-neutral-50/70 rounded-3xl border border-neutral-200/80 p-3 overflow-hidden shadow-inner">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-auto select-none cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Background Grid Lines & Y-Axis Scale */}
          {[0, 0.25, 0.5, 0.7, 0.85, 1.0].map((level) => {
            const y = padding.top + chartHeight - level * chartHeight;
            const isThreshold70 = level === 0.7;
            const isThreshold85 = level === 0.85;

            return (
              <g key={level}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={padding.left + chartWidth}
                  y2={y}
                  stroke={
                    isThreshold85
                      ? '#FECDD3'
                      : isThreshold70
                      ? '#FDE68A'
                      : '#E5E7EB'
                  }
                  strokeWidth={isThreshold70 || isThreshold85 ? 1.5 : 1}
                  strokeDasharray={isThreshold70 || isThreshold85 ? '4 4' : 'none'}
                />
                <text
                  x={padding.left - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  fontSize="9.5"
                  fill={isThreshold85 ? '#9F1239' : isThreshold70 ? '#92400E' : '#9CA3AF'}
                  fontWeight={isThreshold70 || isThreshold85 ? '700' : '400'}
                  fontFamily="IBM Plex Mono"
                >
                  {(level * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}

          {/* Threshold Legend Badges inside SVG */}
          <text
            x={padding.left + chartWidth - 8}
            y={padding.top + chartHeight - 0.85 * chartHeight - 5}
            textAnchor="end"
            fontSize="8.5"
            fill="#9F1239"
            fontWeight="bold"
            fontFamily="IBM Plex Mono"
          >
            0.85 Critical Alert Threshold
          </text>

          <text
            x={padding.left + chartWidth - 8}
            y={padding.top + chartHeight - 0.70 * chartHeight - 5}
            textAnchor="end"
            fontSize="8.5"
            fill="#92400E"
            fontWeight="bold"
            fontFamily="IBM Plex Mono"
          >
            0.70 Elevated Triage Threshold
          </text>

          {/* Aggregate Gradient Area */}
          {visibleCategories.aggregate && (
            <path
              d={generateAreaPath()}
              fill="rgba(17, 24, 39, 0.03)"
            />
          )}

          {/* Telemetry Category Curves */}
          {visibleCategories.hand && (
            <path
              d={generatePath('handScore')}
              fill="none"
              stroke="#0284C7"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {visibleCategories.peer && (
            <path
              d={generatePath('peerScore')}
              fill="none"
              stroke="#8B5CF6"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {visibleCategories.gaze && (
            <path
              d={generatePath('gazeScore')}
              fill="none"
              stroke="#F59E0B"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {visibleCategories.phone && (
            <path
              d={generatePath('phoneScore')}
              fill="none"
              stroke="#F43F5E"
              strokeWidth="2.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {visibleCategories.aggregate && (
            <path
              d={generatePath('aggregateScore')}
              fill="none"
              stroke="#111827"
              strokeWidth="1.6"
              strokeDasharray="3 3"
            />
          )}

          {/* Time Axis (X-Axis) Ticks */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((fraction, idx) => {
            const index = Math.round(fraction * (dataPoints.length - 1));
            const point = dataPoints[index];
            const x = padding.left + fraction * chartWidth;
            const y = padding.top + chartHeight + 16;

            if (!point) return null;
            return (
              <text
                key={idx}
                x={x}
                y={y}
                textAnchor="middle"
                fontSize="9.5"
                fill="#71717A"
                fontWeight="500"
                fontFamily="IBM Plex Mono"
              >
                {point.timestamp}
              </text>
            );
          })}

          {/* Hover Crosshair Scrubber Line & Active Point Dots */}
          {hoverX !== null && hoveredPoint && (
            <g>
              <line
                x1={hoverX}
                y1={padding.top}
                x2={hoverX}
                y2={padding.top + chartHeight}
                stroke="#111827"
                strokeWidth="1.5"
                strokeDasharray="2 2"
              />

              {visibleCategories.phone && (
                <circle
                  cx={hoverX}
                  cy={padding.top + chartHeight - hoveredPoint.phoneScore * chartHeight}
                  r="4.5"
                  fill="#F43F5E"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                />
              )}
              {visibleCategories.gaze && (
                <circle
                  cx={hoverX}
                  cy={padding.top + chartHeight - hoveredPoint.gazeScore * chartHeight}
                  r="4.5"
                  fill="#F59E0B"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                />
              )}
              {visibleCategories.peer && (
                <circle
                  cx={hoverX}
                  cy={padding.top + chartHeight - hoveredPoint.peerScore * chartHeight}
                  r="4.5"
                  fill="#8B5CF6"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                />
              )}
              {visibleCategories.hand && (
                <circle
                  cx={hoverX}
                  cy={padding.top + chartHeight - hoveredPoint.handScore * chartHeight}
                  r="4.5"
                  fill="#0284C7"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                />
              )}
            </g>
          )}
        </svg>

        {/* Hover Tooltip Popup Overlay */}
        <AnimatePresence>
          {hoveredPoint && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="absolute top-4 right-4 bg-white/95 backdrop-blur-md p-4 rounded-2xl border border-neutral-200 shadow-xl text-xs space-y-2 font-mono min-w-[230px] z-30"
            >
              <div className="flex items-center justify-between pb-1.5 border-b border-neutral-100 font-bold text-neutral-950">
                <span className="flex items-center gap-1.5">
                  <Clock size={12} className="text-neutral-500" />
                  {hoveredPoint.timestamp}
                </span>
                <span className="text-neutral-500 font-normal">T+{hoveredPoint.timeSec}s</span>
              </div>

              <div className="space-y-1.5 text-2xs">
                <div className="flex items-center justify-between text-rose-800">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-500" />
                    Phone Probability:
                  </span>
                  <strong className="font-bold">{(hoveredPoint.phoneScore * 100).toFixed(0)}%</strong>
                </div>

                <div className="flex items-center justify-between text-amber-800">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    Gaze Deviation:
                  </span>
                  <strong className="font-bold">{(hoveredPoint.gazeScore * 100).toFixed(0)}%</strong>
                </div>

                <div className="flex items-center justify-between text-violet-800">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-violet-500" />
                    Peer Interaction:
                  </span>
                  <strong className="font-bold">{(hoveredPoint.peerScore * 100).toFixed(0)}%</strong>
                </div>

                <div className="flex items-center justify-between text-sky-800">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-sky-500" />
                    Off-Desk Hands:
                  </span>
                  <strong className="font-bold">{(hoveredPoint.handScore * 100).toFixed(0)}%</strong>
                </div>

                <div className="flex items-center justify-between pt-1.5 border-t border-neutral-100 text-neutral-950 font-bold">
                  <span>Composite Index:</span>
                  <strong>{(hoveredPoint.aggregateScore * 100).toFixed(0)}%</strong>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 4. FOUR EXECUTIVE METRIC CARDS WITH MINI SPARKLINES (NO CLASHING CANDY COLORS) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Card 1: Phone Interaction */}
        <div className="p-5 rounded-2xl border border-neutral-200/80 bg-white shadow-2xs hover:border-neutral-300 transition-all flex items-center justify-between">
          <div className="space-y-1.5">
            <div className="text-xs font-semibold text-neutral-950 flex items-center gap-2 font-heading">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span>Phone / Electronic Device</span>
            </div>
            <div className="flex items-baseline gap-2.5">
              <span className="text-2xl font-extrabold text-neutral-950 font-mono">
                {(latest.phoneScore * 100).toFixed(0)}%
              </span>
              <span className="text-2xs text-neutral-500 font-mono font-medium">
                Peak: 84% • Lap Zone
              </span>
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            <StatusBadge
              status={latest.phoneScore >= 0.7 ? 'escalated' : 'normal'}
              label={latest.phoneScore >= 0.7 ? 'Alert' : 'Nominal'}
              size="sm"
            />
            {/* Live Mini Sparkline */}
            <svg width="100" height="26" className="overflow-visible">
              <path
                d={generateMiniSparkline('phoneScore', 100, 26)}
                fill="none"
                stroke="#F43F5E"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {/* Card 2: Head Yaw Anomaly */}
        <div className="p-5 rounded-2xl border border-neutral-200/80 bg-white shadow-2xs hover:border-neutral-300 transition-all flex items-center justify-between">
          <div className="space-y-1.5">
            <div className="text-xs font-semibold text-neutral-950 flex items-center gap-2 font-heading">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <span>Head Yaw / Gaze Deflection</span>
            </div>
            <div className="flex items-baseline gap-2.5">
              <span className="text-2xl font-extrabold text-neutral-950 font-mono">
                {(latest.gazeScore * 100).toFixed(0)}%
              </span>
              <span className="text-2xs text-neutral-500 font-mono font-medium">
                Peak: 88% • +38° Yaw
              </span>
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            <StatusBadge
              status={latest.gazeScore >= 0.7 ? 'review' : 'normal'}
              label={latest.gazeScore >= 0.7 ? 'Review' : 'Nominal'}
              size="sm"
            />
            {/* Live Mini Sparkline */}
            <svg width="100" height="26" className="overflow-visible">
              <path
                d={generateMiniSparkline('gazeScore', 100, 26)}
                fill="none"
                stroke="#F59E0B"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {/* Card 3: Peer Collaboration */}
        <div className="p-5 rounded-2xl border border-neutral-200/80 bg-white shadow-2xs hover:border-neutral-300 transition-all flex items-center justify-between">
          <div className="space-y-1.5">
            <div className="text-xs font-semibold text-neutral-950 flex items-center gap-2 font-heading">
              <span className="w-2 h-2 rounded-full bg-violet-500" />
              <span>Peer Mutual Collaboration</span>
            </div>
            <div className="flex items-baseline gap-2.5">
              <span className="text-2xl font-extrabold text-neutral-950 font-mono">
                {(latest.peerScore * 100).toFixed(0)}%
              </span>
              <span className="text-2xs text-neutral-500 font-mono font-medium">
                Peak: 78% • Vector Link
              </span>
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            <StatusBadge
              status={latest.peerScore >= 0.7 ? 'review' : 'normal'}
              label={latest.peerScore >= 0.7 ? 'Review' : 'Nominal'}
              size="sm"
            />
            {/* Live Mini Sparkline */}
            <svg width="100" height="26" className="overflow-visible">
              <path
                d={generateMiniSparkline('peerScore', 100, 26)}
                fill="none"
                stroke="#8B5CF6"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {/* Card 4: Hands Off-Keyboard */}
        <div className="p-5 rounded-2xl border border-neutral-200/80 bg-white shadow-2xs hover:border-neutral-300 transition-all flex items-center justify-between">
          <div className="space-y-1.5">
            <div className="text-xs font-semibold text-neutral-950 flex items-center gap-2 font-heading">
              <span className="w-2 h-2 rounded-full bg-sky-500" />
              <span>Hands Off-Keyboard / Lap</span>
            </div>
            <div className="flex items-baseline gap-2.5">
              <span className="text-2xl font-extrabold text-neutral-950 font-mono">
                {(latest.handScore * 100).toFixed(0)}%
              </span>
              <span className="text-2xs text-neutral-500 font-mono font-medium">
                Peak: 72% • 0.18m Offset
              </span>
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            <StatusBadge
              status={latest.handScore >= 0.7 ? 'observe' : 'normal'}
              label={latest.handScore >= 0.7 ? 'Watch' : 'Nominal'}
              size="sm"
            />
            {/* Live Mini Sparkline */}
            <svg width="100" height="26" className="overflow-visible">
              <path
                d={generateMiniSparkline('handScore', 100, 26)}
                fill="none"
                stroke="#0284C7"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
};
