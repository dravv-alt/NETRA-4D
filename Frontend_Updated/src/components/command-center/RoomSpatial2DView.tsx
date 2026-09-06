import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SeatNodeData, RoomLandmark, CameraPlacement } from '../../types';
import {
  COMPUTER_LAB_LANDMARKS,
  LAB_CAMERAS,
  COMPUTER_LAB_20_SPATIAL_SEATS,
} from '../../data/computerLabSpatialPlan';
import {
  Camera,
  Eye,
  Shield,
  Laptop,
  Compass,
  ArrowRight,
  Maximize2,
  AlertCircle,
  CheckCircle2,
  Sliders,
  Layers,
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { StatusBadge } from '../ui/StatusBadge';

interface RoomSpatial2DViewProps {
  seats: SeatNodeData[];
  onSelectSeat: (seatId: string) => void;
  selectedSeatId?: string | null;
  landmarks?: RoomLandmark[];
  cameras?: CameraPlacement[];
}

export const RoomSpatial2DView: React.FC<RoomSpatial2DViewProps> = ({
  seats = COMPUTER_LAB_20_SPATIAL_SEATS,
  onSelectSeat,
  selectedSeatId,
  landmarks = COMPUTER_LAB_LANDMARKS,
  cameras = LAB_CAMERAS,
}) => {
  const [hoveredSeatId, setHoveredSeatId] = useState<string | null>(null);
  const [activeCameraId, setActiveCameraId] = useState<string | null>('CAM-01');
  const [showFovCones, setShowFovCones] = useState<boolean>(true);
  const [showNeighbors, setShowNeighbors] = useState<boolean>(true);

  // Fallback to spatial position or calculate from row/col if not present
  const resolvedSeats = seats.map((s, idx) => {
    if (s.spatialPosition) return s;
    // Map traditional 2-row seats into 2D plane if custom coordinates are absent
    const isRowA = s.gridPosition.row === 'A';
    const col = s.gridPosition.col || ((idx % 10) + 1);
    const x = 8 + (col - 1) * 9.2;
    const y = isRowA ? 25 : 75;
    const facing = isRowA ? ('North' as const) : ('South' as const);
    return {
      ...s,
      spatialPosition: {
        x,
        y,
        facing,
        zoneId: isRowA ? 'NORTH_ROW' : 'SOUTH_ROW',
        neighbors: [
          col > 1 ? `${s.gridPosition.row}${String(col - 1).padStart(2, '0')}` : '',
          col < 10 ? `${s.gridPosition.row}${String(col + 1).padStart(2, '0')}` : '',
        ].filter(Boolean),
      },
    };
  });

  const selectedSeat = resolvedSeats.find((s) => s.id === selectedSeatId);
  const hoveredSeat = resolvedSeats.find((s) => s.id === hoveredSeatId);

  // Active camera coverage set
  const activeCamera = cameras.find((c) => c.id === activeCameraId);
  const coveredSeatIds = new Set(activeCamera ? activeCamera.coverageSeatIds : []);

  // Compute camera FOV cone polygon for SVG
  const getCameraConePolygon = (cam: CameraPlacement) => {
    const cx = cam.x;
    const cy = cam.y;
    const rad = (cam.directionDeg * Math.PI) / 180;
    const halfFov = ((cam.fovAngle / 2) * Math.PI) / 180;
    const distance = 85; // Length of vision wedge

    const leftAngle = rad - halfFov;
    const rightAngle = rad + halfFov;

    const x1 = cx + distance * Math.cos(leftAngle);
    const y1 = cy + distance * Math.sin(leftAngle);
    const x2 = cx + distance * Math.cos(rightAngle);
    const y2 = cy + distance * Math.sin(rightAngle);

    return `${cx},${cy} ${x1},${y1} ${x2},${y2}`;
  };

  return (
    <div className="relative bg-white rounded-3xl border border-neutral-200/90 shadow-sm p-5 sm:p-7 space-y-4 font-sans select-none">
      {/* 1. Header Toolbar */}
      <div className="flex items-center justify-between flex-wrap gap-3 pb-3 border-b border-neutral-200">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-neutral-900 flex items-center justify-center text-white shadow-2xs">
            <Compass size={16} />
          </div>
          <div>
            <div className="font-heading font-extrabold text-sm text-neutral-950 flex items-center gap-2">
              <span>Top-Down Architectural Room Map</span>
              <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
                Spatial Prior Active
              </span>
            </div>
            <p className="text-3xs text-neutral-500">
              Interactive 2D Hall Topology • Desks, Gaze Orientations & Camera Ray-Casting
            </p>
          </div>
        </div>

        {/* View Controls & Camera Selectors */}
        <div className="flex items-center gap-2 flex-wrap text-3xs font-mono">
          {/* Toggle FOV */}
          <button
            onClick={() => {
              playHapticSound('click');
              setShowFovCones(!showFovCones);
            }}
            className={`px-2.5 py-1 rounded-lg border transition-all cursor-pointer flex items-center gap-1.5 ${
              showFovCones
                ? 'bg-sky-50 border-sky-300 text-sky-900 font-bold'
                : 'bg-neutral-50 border-neutral-200 text-neutral-500 hover:text-neutral-900'
            }`}
          >
            <Eye size={12} />
            <span>Vision Cones</span>
          </button>

          {/* Toggle Neighbors */}
          <button
            onClick={() => {
              playHapticSound('click');
              setShowNeighbors(!showNeighbors);
            }}
            className={`px-2.5 py-1 rounded-lg border transition-all cursor-pointer flex items-center gap-1.5 ${
              showNeighbors
                ? 'bg-violet-50 border-violet-300 text-violet-900 font-bold'
                : 'bg-neutral-50 border-neutral-200 text-neutral-500 hover:text-neutral-900'
            }`}
          >
            <Layers size={12} />
            <span>Adjacency Graph</span>
          </button>

          {/* Camera Buttons */}
          <div className="flex items-center bg-neutral-100 p-0.5 rounded-xl border border-neutral-200">
            {cameras.map((cam) => (
              <button
                key={cam.id}
                onClick={() => {
                  playHapticSound('click');
                  setActiveCameraId(activeCameraId === cam.id ? null : cam.id);
                }}
                title={cam.name}
                className={`px-2 py-1 rounded-lg transition-all cursor-pointer font-bold flex items-center gap-1 ${
                  activeCameraId === cam.id
                    ? 'bg-neutral-900 text-white shadow-2xs'
                    : 'text-neutral-600 hover:text-neutral-900'
                }`}
              >
                <Camera size={11} />
                <span>{cam.id}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 2. Main SVG 2D Canvas */}
      <div className="relative w-full aspect-16/11 sm:aspect-16/10 bg-neutral-50/70 rounded-2xl border border-neutral-200 overflow-hidden shadow-inner flex items-center justify-center">
        <svg
          viewBox="0 0 100 100"
          className="w-full h-full"
          style={{ transform: 'translateZ(0)' }}
        >
          <defs>
            {/* Camera FOV Vision Cone Gradients */}
            <radialGradient id="camGlow" cx="0%" cy="0%" r="100%">
              <stop offset="0%" stopColor="#0284c7" stopOpacity="0.32" />
              <stop offset="60%" stopColor="#0284c7" stopOpacity="0.08" />
              <stop offset="100%" stopColor="#0284c7" stopOpacity="0.0" />
            </radialGradient>

            {/* Pattern for Walkway */}
            <pattern id="walkwayPattern" width="4" height="4" patternUnits="userSpaceOnUse">
              <path d="M 0 4 L 4 0 M 0 0 L 4 4" fill="none" stroke="#e5e5e5" strokeWidth="0.4" />
            </pattern>
          </defs>

          {/* A. Outer Hall Wall Boundaries */}
          <rect
            x="2"
            y="2"
            width="96"
            height="96"
            rx="1.5"
            fill="#ffffff"
            stroke="#171717"
            strokeWidth="0.9"
          />

          {/* B. Central Proctor Walkway Corridor */}
          <rect
            x="39"
            y="20"
            width="22"
            height="70"
            fill="url(#walkwayPattern)"
            opacity="0.8"
          />
          <text
            x="50"
            y="60"
            textAnchor="middle"
            fill="#a3a3a3"
            fontSize="1.6"
            fontWeight="bold"
            letterSpacing="0.2"
            className="select-none font-mono"
          >
            [ CENTRAL PROCTOR WALKWAY ]
          </text>

          {/* C. Hall Landmarks */}
          {landmarks.map((lm) => {
            if (lm.type === 'board') {
              return (
                <g key={lm.id}>
                  <rect
                    x={lm.x - lm.width / 2}
                    y={lm.y}
                    width={lm.width}
                    height={lm.height}
                    fill="#1e293b"
                    rx="0.4"
                  />
                  <text
                    x={lm.x}
                    y={lm.y + 2.1}
                    textAnchor="middle"
                    fill="#cbd5e1"
                    fontSize="1.3"
                    fontWeight="bold"
                    className="font-mono"
                  >
                    NORTH BOARD / PROJECTION SCREEN
                  </text>
                </g>
              );
            }
            if (lm.type === 'bookshelf') {
              return (
                <g key={lm.id}>
                  <rect
                    x={lm.x - lm.width / 2}
                    y={lm.y}
                    width={lm.width}
                    height={lm.height}
                    fill="#f1f5f9"
                    stroke="#94a3b8"
                    strokeWidth="0.3"
                    rx="0.3"
                  />
                  <text
                    x={lm.x}
                    y={lm.y + 2.6}
                    textAnchor="middle"
                    fill="#64748b"
                    fontSize="1.1"
                    fontWeight="bold"
                    className="font-mono"
                  >
                    RACKS / SERVERS
                  </text>
                </g>
              );
            }
            if (lm.type === 'podium') {
              return (
                <g key={lm.id}>
                  {/* Proctor Desk */}
                  <rect
                    x={lm.x - lm.width / 2}
                    y={lm.y - lm.height / 2}
                    width={lm.width}
                    height={lm.height}
                    fill="#f8fafc"
                    stroke="#0f172a"
                    strokeWidth="0.5"
                    rx="0.6"
                  />
                  {/* Proctor Laptop */}
                  <rect
                    x={lm.x - 3}
                    y={lm.y - 1.5}
                    width="6"
                    height="2.4"
                    fill="#0284c7"
                    rx="0.2"
                  />
                  <text
                    x={lm.x}
                    y={lm.y + 2.2}
                    textAnchor="middle"
                    fill="#0f172a"
                    fontSize="1.3"
                    fontWeight="bold"
                    className="font-mono"
                  >
                    PROCTOR PODIUM
                  </text>
                  {/* Shield badge */}
                  <circle cx={lm.x} cy={lm.y - 4.5} r="1.6" fill="#171717" />
                  <text
                    x={lm.x}
                    y={lm.y - 3.8}
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize="1.8"
                    fontWeight="bold"
                  >
                    🛡
                  </text>
                </g>
              );
            }
            if (lm.type === 'door') {
              return (
                <g key={lm.id}>
                  {/* Door frame */}
                  <line
                    x1={lm.x - lm.width / 2}
                    y1={lm.y}
                    x2={lm.x + lm.width / 2}
                    y2={lm.y}
                    stroke="#ffffff"
                    strokeWidth="1.6"
                  />
                  {/* Left Door Swing */}
                  <path
                    d={`M ${lm.x - lm.width / 2} ${lm.y} A ${lm.width / 2} ${lm.width / 2} 0 0 1 ${lm.x} ${lm.y - lm.height * 1.5}`}
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth="0.4"
                    strokeDasharray="0.8,0.8"
                  />
                  {/* Right Door Swing */}
                  <path
                    d={`M ${lm.x + lm.width / 2} ${lm.y} A ${lm.width / 2} ${lm.width / 2} 0 0 0 ${lm.x} ${lm.y - lm.height * 1.5}`}
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth="0.4"
                    strokeDasharray="0.8,0.8"
                  />
                  <text
                    x={lm.x}
                    y={lm.y - 1}
                    textAnchor="middle"
                    fill="#0284c7"
                    fontSize="1.3"
                    fontWeight="bold"
                    className="font-mono"
                  >
                    ENTRANCE
                  </text>
                </g>
              );
            }
            return null;
          })}

          {/* D. Camera Vision Cones (When Active) */}
          {showFovCones &&
            activeCamera && (
              <polygon
                points={getCameraConePolygon(activeCamera)}
                fill="url(#camGlow)"
                stroke="#0284c7"
                strokeWidth="0.25"
                strokeDasharray="1.2,1.2"
                className="transition-all duration-300 pointer-events-none"
              />
            )}

          {/* E. Neighbor Adjacency Graph Vectors (Dashed lines to spatial peers) */}
          {showNeighbors &&
            selectedSeat &&
            selectedSeat.spatialPosition &&
            selectedSeat.spatialPosition.neighbors && (
              <g className="pointer-events-none">
                {selectedSeat.spatialPosition.neighbors.map((nId) => {
                  const nSeat = resolvedSeats.find((s) => s.id === nId);
                  const selPos = selectedSeat.spatialPosition;
                  if (!nSeat?.spatialPosition || !selPos) return null;
                  return (
                    <line
                      key={`${selectedSeat.id}-${nId}`}
                      x1={selPos.x}
                      y1={selPos.y}
                      x2={nSeat.spatialPosition.x}
                      y2={nSeat.spatialPosition.y}
                      stroke="#8b5cf6"
                      strokeWidth="0.5"
                      strokeDasharray="1.2,1.2"
                      className="animate-pulse"
                    />
                  );
                })}
              </g>
            )}

          {/* F. Workstation Terminals (20 Desks + Monitors + Chairs) */}
          {resolvedSeats.map((seat) => {
            const pos = seat.spatialPosition!;
            const isSelected = seat.id === selectedSeatId;
            const isHovered = seat.id === hoveredSeatId;
            const isCovered = coveredSeatIds.has(seat.id);

            // Status colors
            let fillColor = '#ffffff';
            let strokeColor = '#d4d4d4';
            let textColor = '#171717';

            if (seat.state === 'escalated') {
              fillColor = '#ffe4e6';
              strokeColor = '#e11d48';
              textColor = '#9f1239';
            } else if (seat.state === 'review') {
              fillColor = '#fef3c7';
              strokeColor = '#d97706';
              textColor = '#92400e';
            } else if (seat.state === 'observed') {
              fillColor = '#ede9fe';
              strokeColor = '#7c3aed';
              textColor = '#5b21b6';
            } else if (isSelected) {
              fillColor = '#171717';
              strokeColor = '#000000';
              textColor = '#ffffff';
            }

            // Facing direction vectors for chair & monitor
            const facing = pos.facing || 'East';
            let chairDx = 0;
            let chairDy = 0;
            let screenDx = 0;
            let screenDy = 0;
            let deskW = 5.2;
            let deskH = 4.2;

            if (facing === 'East') {
              deskW = 4.2;
              deskH = 6.2;
              chairDx = -2.8;
              chairDy = 0;
              screenDx = 1.0;
              screenDy = 0;
            } else if (facing === 'West') {
              deskW = 4.2;
              deskH = 6.2;
              chairDx = 2.8;
              chairDy = 0;
              screenDx = -1.0;
              screenDy = 0;
            } else if (facing === 'South') {
              deskW = 6.2;
              deskH = 4.2;
              chairDx = 0;
              chairDy = -2.8;
              screenDx = 0;
              screenDy = 1.0;
            } else {
              // North
              deskW = 6.2;
              deskH = 4.2;
              chairDx = 0;
              chairDy = 2.8;
              screenDx = 0;
              screenDy = -1.0;
            }

            return (
              <g
                key={seat.id}
                onClick={() => {
                  playHapticSound('snap');
                  onSelectSeat(seat.id);
                }}
                onMouseEnter={() => setHoveredSeatId(seat.id)}
                onMouseLeave={() => setHoveredSeatId(null)}
                className="cursor-pointer transition-all duration-150"
                transform={`translate(${pos.x}, ${pos.y})`}
              >
                {/* Highlight ring if monitored by active camera */}
                {activeCameraId && isCovered && (
                  <circle
                    cx="0"
                    cy="0"
                    r="4.8"
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth="0.35"
                    strokeDasharray="0.8,0.8"
                  />
                )}

                {/* Desk Surface */}
                <rect
                  x={-deskW / 2}
                  y={-deskH / 2}
                  width={deskW}
                  height={deskH}
                  rx="0.5"
                  fill={fillColor}
                  stroke={isSelected ? '#171717' : strokeColor}
                  strokeWidth={isSelected ? '0.7' : '0.35'}
                  filter={isSelected ? 'drop-shadow(0 2px 4px rgba(0,0,0,0.25))' : undefined}
                />

                {/* Monitor / Screen */}
                <rect
                  x={screenDx - (facing === 'East' || facing === 'West' ? 0.6 : 1.8)}
                  y={screenDy - (facing === 'East' || facing === 'West' ? 1.8 : 0.6)}
                  width={facing === 'East' || facing === 'West' ? 1.2 : 3.6}
                  height={facing === 'East' || facing === 'West' ? 3.6 : 1.2}
                  rx="0.2"
                  fill={isSelected ? '#38bdf8' : '#0284c7'}
                />

                {/* Directional Ergonomic Chair */}
                <circle
                  cx={chairDx}
                  cy={chairDy}
                  r="1.4"
                  fill="#404040"
                  stroke="#171717"
                  strokeWidth="0.3"
                />

                {/* Seat Code Label */}
                <text
                  x="0"
                  y={facing === 'South' || facing === 'North' ? '0.5' : '0.6'}
                  textAnchor="middle"
                  fill={textColor}
                  fontSize="1.5"
                  fontWeight="bold"
                  className="font-mono select-none"
                >
                  {seat.id}
                </text>

                {/* Red/Amber Object Indicator badge (Phone / Chit) */}
                {seat.telemetry.objectEvidence && (
                  <g transform={`translate(${deskW / 2 - 0.5}, ${-deskH / 2 + 0.5})`}>
                    <circle cx="0" cy="0" r="1.1" fill="#ef4444" />
                    <text
                      x="0"
                      y="0.35"
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize="1.0"
                      fontWeight="bold"
                    >
                      !
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* G. Camera Anchors in Hall Corners */}
          {cameras.map((cam) => {
            const isCamActive = activeCameraId === cam.id;
            return (
              <g
                key={cam.id}
                onClick={() => {
                  playHapticSound('click');
                  setActiveCameraId(isCamActive ? null : cam.id);
                }}
                className="cursor-pointer"
                transform={`translate(${cam.x}, ${cam.y})`}
              >
                <circle
                  cx="0"
                  cy="0"
                  r={isCamActive ? '3.0' : '2.4'}
                  fill={isCamActive ? '#0284c7' : '#171717'}
                  stroke="#ffffff"
                  strokeWidth="0.5"
                  className="transition-all duration-200"
                />
                <text
                  x="0"
                  y="0.6"
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize="1.4"
                  fontWeight="bold"
                  className="font-mono"
                >
                  📹
                </text>
                {/* Camera Name Tag */}
                <text
                  x="0"
                  y={cam.y > 50 ? '-3.8' : '4.8'}
                  textAnchor="middle"
                  fill={isCamActive ? '#0284c7' : '#525252'}
                  fontSize="1.4"
                  fontWeight="bold"
                  className="font-mono"
                >
                  {cam.id}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Floating Tooltip for Selected/Hovered Terminal */}
        <AnimatePresence>
          {(hoveredSeat || selectedSeat) && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 5 }}
              className="absolute bottom-3 left-3 bg-neutral-900/95 text-white backdrop-blur-md px-3.5 py-2.5 rounded-xl border border-neutral-800 shadow-xl max-w-xs text-3xs font-mono pointer-events-none z-30 space-y-1"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="font-bold text-xs text-sky-400">
                  Seat {(hoveredSeat || selectedSeat)?.id} • Candidate #{(hoveredSeat || selectedSeat)?.candidateId}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 uppercase">
                  {(hoveredSeat || selectedSeat)?.spatialPosition?.facing} Facing
                </span>
              </div>
              <div className="text-neutral-300">
                {(hoveredSeat || selectedSeat)?.telemetry.lastEventSummary}
              </div>
              <div className="text-neutral-400 flex items-center justify-between pt-0.5 border-t border-neutral-800">
                <span>
                  Neighbors: {(hoveredSeat || selectedSeat)?.spatialPosition?.neighbors?.join(', ') || 'None'}
                </span>
                <span className="text-amber-400">
                  {(hoveredSeat || selectedSeat)?.telemetry.objectEvidence?.label || ''}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 3. Bottom Camera & Spatial Legend */}
      <div className="flex items-center justify-between flex-wrap gap-2 text-3xs font-mono text-neutral-500 pt-1">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-white border border-neutral-300" /> Normal Desk
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-100 border border-amber-400" /> Review
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-100 border border-rose-400" /> Escalated
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-neutral-900 border border-white" /> Ergonomic Chair
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span>Active Camera Coverage:</span>
          <strong className="text-neutral-900">
            {activeCamera ? `${activeCamera.name} (${activeCamera.coverageSeatIds.length} Terminals)` : 'None (All Hall Visible)'}
          </strong>
        </div>
      </div>
    </div>
  );
};
