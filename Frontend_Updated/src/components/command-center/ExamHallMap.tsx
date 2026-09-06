import React, { useState, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SeatNodeData, CameraFeedInfo } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import {
  ArrowRight,
  Laptop,
  Shield,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  Sliders,
  Compass,
  Layers,
  Camera,
  Eye,
  RotateCw,
  Move,
  MapPin,
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { LabLayoutStudioModal } from '../layout-designer/LabLayoutStudioModal';
import { IllustratedIcon } from '../ui/IllustratedIcon';
import { CAMERA_FEEDS } from '../../data/demoData';

interface ExamHallMapProps {
  seats: SeatNodeData[];
  onSelectSeat: (seatId: string) => void;
  selectedSeatId?: string | null;
  onUpdateSeats?: (newSeats: SeatNodeData[]) => void;
  cameras?: CameraFeedInfo[];
}

export const ExamHallMap: React.FC<ExamHallMapProps> = ({
  seats,
  onSelectSeat,
  selectedSeatId,
  onUpdateSeats,
  cameras = CAMERA_FEEDS,
}) => {
  const [hoveredSeat, setHoveredSeat] = useState<SeatNodeData | null>(null);
  const [isStudioOpen, setIsStudioOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'floorplan' | 'compact-grid'>('floorplan');
  const [rotation, setRotation] = useState<0 | 90 | 180 | 270>(0);
  const [cameraMode, setCameraMode] = useState<'single' | 'quad'>('single');
  // CCTV Camera is NOT placed by default; the user places it anywhere in the room
  const [customCamPos, setCustomCamPos] = useState<{ x: number; y: number } | null>(null);
  const [camAimAngle, setCamAimAngle] = useState<number | null>(null); // null = auto-aim toward room center
  const [isPlacingCam, setIsPlacingCam] = useState<boolean>(false);
  const [isDraggingCam, setIsDraggingCam] = useState<boolean>(false);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [showFovCones, setShowFovCones] = useState<boolean>(true);
  const [showAdjacencyGraph, setShowAdjacencyGraph] = useState<boolean>(true);
  const [activeCameraFilter, setActiveCameraFilter] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const isPortrait = rotation === 90 || rotation === 270;

  // Visual camera position in current rotated space (null if not yet placed by user)
  const visualCamPos = useMemo(() => {
    if (!customCamPos) return null;
    return getRotatedPoint(customCamPos.x, customCamPos.y, rotation);
  }, [customCamPos, rotation]);

  // Universal Camera Drag Handler: works whether starting from placed badge or toolbar token
  const startCameraDrag = () => {
    setIsDraggingCam(true);
    setIsPlacingCam(false);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const rawX = ((moveEvent.clientX - rect.left) / rect.width) * 100;
      const rawY = ((moveEvent.clientY - rect.top) / rect.height) * 100;
      const clampedX = Math.max(3, Math.min(97, rawX));
      const clampedY = Math.max(3, Math.min(97, rawY));

      const canonical = unrotatePoint(clampedX, clampedY, rotation);
      setCustomCamPos(canonical);
    };

    const handlePointerUp = () => {
      setIsDraggingCam(false);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      playHapticSound('snap');
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
  };

  const handlePointerDownCam = (e: React.PointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    startCameraDrag();
  };

  // Track mouse position over canvas for placement preview
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPlacingCam || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const rawX = ((e.clientX - rect.left) / rect.width) * 100;
    const rawY = ((e.clientY - rect.top) / rect.height) * 100;
    const clampedX = Math.max(3, Math.min(97, rawX));
    const clampedY = Math.max(3, Math.min(97, rawY));
    setMousePos({ x: clampedX, y: clampedY });
  };

  const handleCanvasMouseLeave = () => {
    setMousePos(null);
  };

  // Handle clicking on room canvas to place camera anywhere in the room
  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPlacingCam || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const rawX = ((e.clientX - rect.left) / rect.width) * 100;
    const rawY = ((e.clientY - rect.top) / rect.height) * 100;
    const clampedX = Math.max(3, Math.min(97, rawX));
    const clampedY = Math.max(3, Math.min(97, rawY));

    const canonical = unrotatePoint(clampedX, clampedY, rotation);
    setCustomCamPos(canonical);
    setIsPlacingCam(false);
    setMousePos(null);
    playHapticSound('snap');
  };

  // Fallback / Normalized Spatial Coordinate Resolution with Strict Collision-Free Bands
  const resolvedSeats = useMemo(() => {
    return seats.map((seat, idx) => {
      if (seat.spatialPosition && seat.spatialPosition.x && seat.spatialPosition.y) {
        return seat;
      }
      // Standard dual-row seats:
      // Row A positioned strictly at y = 33% (below proctor podium at 13%, with clear aisle)
      // Row B positioned strictly at y = 67% (below central walkway at 50%, with clear aisle)
      const isRowA = seat.gridPosition.row === 'A';
      const col = seat.gridPosition.col || ((idx % 10) + 1);
      const rowSeatsCount = seats.filter((s) => s.gridPosition.row === seat.gridPosition.row).length || 10;

      // Evenly distribute across 8% to 92%
      const step = (92.0 - 8.0) / Math.max(1, rowSeatsCount - 1);
      const x = roundTo1(8.0 + (col - 1) * step);
      const y = isRowA ? 33.0 : 67.0;
      const facing = isRowA ? ('North' as const) : ('South' as const);

      return {
        ...seat,
        spatialPosition: {
          x,
          y,
          facing,
          zoneId: isRowA ? 'NORTH_ROW_A' : 'SOUTH_ROW_B',
          zoneName: isRowA ? 'Row A (North Wall)' : 'Row B (South Wall)',
          neighbors: [
            col > 1 ? `${seat.gridPosition.row}${String(col - 1).padStart(2, '0')}` : '',
            col < rowSeatsCount ? `${seat.gridPosition.row}${String(col + 1).padStart(2, '0')}` : '',
          ].filter(Boolean),
        },
      };
    });
  }, [seats]);

  // Dynamic Workstation Card & Layout Scaling Engine
  // Calculates percentage width and compact typography so cards NEVER overlap horizontally or vertically
  const dynamicLayout = useMemo(() => {
    const count = resolvedSeats.length;

    // Portrait Mode (90° / 270°): View length-wise longer terminal deck
    // 10 desks are vertically arrayed down the height; only 2 columns across width.
    if (isPortrait) {
      return {
        widthPercent: 18.0,
        maxWidthPx: 82,
        minWidthPx: 60,
        aspectRatio: '1 / 0.70',
        titleFont: 'text-[10px] sm:text-[11px] font-black',
        candidateFont: 'text-[7.5px] px-1',
        iconSize: 8,
        padding: 'p-1',
        showChair: count <= 24,
      };
    }

    // Landscape Mode (0° / 180°): View width-wise longer terminal deck
    if (count === 0) {
      return {
        widthPercent: 7.2,
        maxWidthPx: 68,
        minWidthPx: 44,
        aspectRatio: '1 / 1.05',
        titleFont: 'text-[10px] font-black',
        candidateFont: 'text-[8px]',
        iconSize: 8,
        padding: 'p-1',
        showChair: true,
      };
    }

    // Calculate minimum horizontal distance between any pair of workstations in percentage space
    let minGapX = 100;
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const p1 = resolvedSeats[i].spatialPosition || { x: 50, y: 50 };
        const p2 = resolvedSeats[j].spatialPosition || { x: 50, y: 50 };
        const dx = Math.abs(p1.x - p2.x);
        if (dx > 1.5 && dx < minGapX) minGapX = dx;
      }
    }

    // Proportional width: Take at most 75% of the distance between workstation centers
    // This leaves a guaranteed 25% empty margin between adjacent cards (zero overlap)
    let widthPercent = Math.min(12.0, Math.max(5.8, minGapX * 0.75));
    let maxWidthPx = 80;
    let minWidthPx = 44;

    if (count <= 6) {
      widthPercent = 11.5;
      maxWidthPx = 96;
      minWidthPx = 68;
    } else if (count <= 10) {
      widthPercent = 8.5;
      maxWidthPx = 76;
      minWidthPx = 54;
    } else if (count <= 20) {
      widthPercent = 7.0;
      maxWidthPx = 68;
      minWidthPx = 46;
    } else {
      widthPercent = 5.6;
      maxWidthPx = 56;
      minWidthPx = 40;
    }

    const isSmall = maxWidthPx < 65;

    return {
      widthPercent,
      maxWidthPx,
      minWidthPx,
      aspectRatio: '1 / 1.05',
      titleFont: isSmall ? 'text-[9px] sm:text-[10px] font-black' : 'text-[10px] sm:text-[11px] font-black',
      candidateFont: isSmall ? 'text-[7px] px-0.5' : 'text-[8px] px-1',
      iconSize: isSmall ? 7 : 8,
      padding: isSmall ? 'p-0.5 sm:p-1' : 'p-1 sm:p-1.5',
      showChair: count <= 24,
    };
  }, [resolvedSeats, isPortrait]);

  const rowA = resolvedSeats.filter((s) => s.gridPosition.row === 'A');
  const rowB = resolvedSeats.filter((s) => s.gridPosition.row === 'B');

  const getFacingIcon = (facing?: string, size = 8) => {
    switch (facing) {
      case 'West':
        return <span title="Facing West"><ArrowLeft size={size} /></span>;
      case 'East':
        return <span title="Facing East"><ArrowRight size={size} /></span>;
      case 'South':
        return <span title="Facing South"><ArrowDown size={size} /></span>;
      case 'North':
      default:
        return <span title="Facing North"><ArrowUp size={size} /></span>;
    }
  };

  const getSeatStyle = (state: SeatNodeData['state'], isSelected: boolean, isDimmed: boolean) => {
    let base = 'transition-all duration-150 cursor-pointer flex flex-col justify-between select-none font-sans rounded-xl shadow-2xs ';

    if (isDimmed) {
      base += 'opacity-25 scale-95 ';
    }

    if (isSelected) {
      base += 'bg-[#181716] text-[#FAF7F3] ring-2 ring-neutral-900 shadow-md border-neutral-900 scale-[1.05] z-30 ';
      return base;
    }

    switch (state) {
      case 'escalated':
        return base + 'bg-rose-50 border border-rose-400 text-rose-950 hover:border-rose-600 hover:shadow-sm z-20 ring-1 ring-rose-200';
      case 'review':
        return base + 'bg-amber-50 border border-amber-400 text-amber-950 hover:border-amber-600 hover:shadow-sm z-20 ring-1 ring-amber-200';
      case 'observed':
        return base + 'bg-violet-50 border border-violet-400 text-violet-950 hover:border-violet-600 hover:shadow-sm z-10';
      case 'normal':
      default:
        return base + 'bg-white border border-neutral-200 text-neutral-900 hover:border-emerald-400 hover:shadow-sm hover:bg-emerald-50/20 z-10';
    }
  };

  // Camera Coverage Definitions for Quad-Camera mode (Room coordinate points in 0-1000 scale)
  const CAMERA_COORDINATES = {
    'CAM-02': { x: 30, y: 30, name: 'North-West CCTV', cone: '30,30 450,30 30,450' },
    'CAM-01': { x: 970, y: 30, name: 'North-East CCTV', cone: '970,30 550,30 970,450' },
    'CAM-03': { x: 30, y: 970, name: 'South-West CCTV', cone: '30,970 450,970 30,550' },
    'CAM-04': { x: 970, y: 970, name: 'South-East CCTV', cone: '970,970 550,970 970,550' },
  };

  // Render Compact Workstation Card
  const renderWorkstationCard = (seat: SeatNodeData, isCompactGrid = false) => {
    const isSelected = selectedSeatId === seat.id;
    const assignedCam = cameraMode === 'single' ? 'CAM-01' : (seat.cameraIds?.[0] || 'CAM-01');
    const isDimmed = Boolean(
      activeCameraFilter &&
      (cameraMode === 'single'
        ? activeCameraFilter !== 'CAM-01'
        : !seat.cameraIds?.includes(activeCameraFilter))
    );
    const rawFacing = (seat.spatialPosition?.facing || (seat.gridPosition.row === 'A' ? 'North' : 'South')) as 'North' | 'South' | 'East' | 'West';
    const facing = isCompactGrid ? rawFacing : rotateFacing(rawFacing, rotation);

    return (
      <div className="relative group/workstation">
        {/* Subtle Ergonomic Chair Indicator */}
        {!isCompactGrid && dynamicLayout.showChair && (
          <>
            {facing === 'North' && (
              <div
                className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-3.5 h-1.5 rounded-b-full bg-neutral-400/80 border border-neutral-500 shadow-2xs pointer-events-none transition-colors group-hover/workstation:bg-neutral-800"
                title="Chair (Facing North)"
              />
            )}
            {facing === 'South' && (
              <div
                className="absolute -top-2 left-1/2 -translate-x-1/2 w-3.5 h-1.5 rounded-t-full bg-neutral-400/80 border border-neutral-500 shadow-2xs pointer-events-none transition-colors group-hover/workstation:bg-neutral-800"
                title="Chair (Facing South)"
              />
            )}
            {facing === 'East' && (
              <div
                className="absolute top-1/2 -translate-y-1/2 -left-2 w-1.5 h-3.5 rounded-l-full bg-neutral-400/80 border border-neutral-500 shadow-2xs pointer-events-none transition-colors group-hover/workstation:bg-neutral-800"
                title="Chair (Facing East)"
              />
            )}
            {facing === 'West' && (
              <div
                className="absolute top-1/2 -translate-y-1/2 -right-2 w-1.5 h-3.5 rounded-r-full bg-neutral-400/80 border border-neutral-500 shadow-2xs pointer-events-none transition-colors group-hover/workstation:bg-neutral-800"
                title="Chair (Facing West)"
              />
            )}
          </>
        )}

        {/* The Interactive Workstation Card */}
        <motion.div
          key={seat.id}
          onClick={() => {
            playHapticSound('snap');
            onSelectSeat(seat.id);
          }}
          onMouseEnter={() => {
            setHoveredSeat(seat);
            playHapticSound('click');
          }}
          onMouseLeave={() => setHoveredSeat(null)}
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.97 }}
          style={
            isCompactGrid
              ? undefined
              : {
                  width: `${dynamicLayout.widthPercent}%`,
                  minWidth: `${dynamicLayout.minWidthPx}px`,
                  maxWidth: `${dynamicLayout.maxWidthPx}px`,
                  aspectRatio: dynamicLayout.aspectRatio,
                }
          }
          className={`${dynamicLayout.padding} ${
            isCompactGrid ? 'min-h-[76px] w-full' : ''
          } ${getSeatStyle(seat.state, isSelected, isDimmed)}`}
        >
          {/* Top Bar: Laptop Icon & Facing Vector */}
          <div className="flex items-center justify-between text-[8px] pb-0.5 border-b border-black/[0.05] font-mono leading-none">
            <span className="flex items-center gap-0.5 opacity-75">
              <Laptop size={dynamicLayout.iconSize} className={isSelected ? 'text-white' : 'text-neutral-700'} />
              <span className="text-[7px]">LPT</span>
            </span>
            <span className="opacity-75 flex items-center gap-0.5">
              {getFacingIcon(facing, dynamicLayout.iconSize)}
              <span className="text-[7px] font-bold uppercase">{facing[0]}</span>
            </span>
          </div>

          {/* Center: Terminal ID & Candidate ID */}
          <div className="my-0.5 flex items-center justify-between leading-none">
            <div className={`font-mono ${dynamicLayout.titleFont} tracking-tight ${isSelected ? 'text-white' : 'text-neutral-950'}`}>
              {seat.id}
            </div>
            <div className={`font-mono font-bold rounded ${dynamicLayout.candidateFont} ${
              isSelected ? 'bg-white/20 text-white' : 'bg-neutral-100 text-neutral-700'
            }`}>
              #{seat.candidateId}
            </div>
          </div>

          {/* Bottom Bar: Camera Tag & Alerts */}
          <div className="pt-0.5 border-t border-black/[0.05] flex items-center justify-between text-[8px] font-mono leading-none opacity-80">
            <span className="flex items-center gap-0.5 truncate text-[7px]" title={`Monitored by ${assignedCam}`}>
              <Camera size={dynamicLayout.iconSize - 1} />
              <span>{assignedCam.replace('CAM-', 'C')}</span>
            </span>
            {seat.telemetry?.objectEvidence ? (
              <span className="font-bold px-1 py-0.2 rounded bg-rose-500 text-white text-[7px] animate-pulse">
                OBJ
              </span>
            ) : (
              <span className={`w-1.5 h-1.5 rounded-full ${
                seat.state === 'normal' ? 'bg-emerald-500' : seat.state === 'observed' ? 'bg-violet-500' : 'bg-amber-500'
              }`} />
            )}
          </div>
        </motion.div>
      </div>
    );
  };

  return (
    <div className="p-4 sm:p-5 rounded-2xl bg-neutral-50/80 border border-neutral-200/90 space-y-3 font-sans select-none">
      {/* 1. Header Toolbar */}
      <div className="flex items-center justify-between pb-2 border-b border-neutral-200 flex-wrap gap-2">
        <div className="flex items-center gap-2.5">
          <IllustratedIcon icon={Compass} variant="sky" size="sm" />
          <div>
            <div className="font-heading font-extrabold text-sm text-neutral-950 flex items-center gap-2">
              <span>Computer Lab Terminal Deck</span>
              <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded-full bg-white border border-neutral-200 text-neutral-800">
                {seats.length} Workstations
              </span>
              <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 flex items-center gap-1">
                <span className={`w-1.5 h-1.5 rounded-full ${customCamPos ? 'bg-emerald-500 animate-pulse' : 'bg-neutral-400'}`} />
                {cameraMode === 'single'
                  ? customCamPos
                    ? `CCTV @ (${customCamPos.x}%, ${customCamPos.y}%)`
                    : 'CCTV: Unplaced (User Placed)'
                  : '4 Distributed Feeds'}
              </span>
            </div>
            <p className="text-3xs text-neutral-500 mt-0.5">
              Architectural Room Floorplan • Spatial Telemetry & Camera Coverage
            </p>
          </div>
        </div>

        {/* View Controls & Action Buttons */}
        <div className="flex items-center gap-2 flex-wrap text-3xs font-mono">
          {/* Floorplan vs Compact Toggle */}
          <div className="flex items-center bg-white p-0.5 rounded-lg border border-neutral-200">
            <button
              onClick={() => {
                playHapticSound('snap');
                setViewMode('floorplan');
              }}
              title="Architectural Room Floorplan View"
              className={`px-2 py-1 rounded-md transition-all font-bold cursor-pointer flex items-center gap-1 ${
                viewMode === 'floorplan' ? 'bg-neutral-900 text-white shadow-2xs' : 'text-neutral-500 hover:text-neutral-900'
              }`}
            >
              <Compass size={11} />
              <span>Floorplan</span>
            </button>
            <button
              onClick={() => {
                playHapticSound('click');
                setViewMode('compact-grid');
              }}
              title="Compact Linear Grid View"
              className={`px-2 py-1 rounded-md transition-all font-bold cursor-pointer flex items-center gap-1 ${
                viewMode === 'compact-grid' ? 'bg-neutral-900 text-white shadow-2xs' : 'text-neutral-500 hover:text-neutral-900'
              }`}
            >
              <span>Linear Grid</span>
            </button>
          </div>

          {/* Camera Setup Selector */}
          {viewMode === 'floorplan' && (
            <div className="flex items-center bg-white p-0.5 rounded-lg border border-neutral-200">
              <button
                onClick={() => {
                  playHapticSound('snap');
                  setCameraMode('single');
                  setActiveCameraFilter(null);
                }}
                title="Single User-Placeable CCTV (Drag or click anywhere to position)"
                className={`px-2 py-1 rounded-md transition-all font-bold cursor-pointer flex items-center gap-1 ${
                  cameraMode === 'single'
                    ? 'bg-neutral-900 text-white shadow-2xs'
                    : 'text-neutral-500 hover:text-neutral-900'
                }`}
              >
                <Camera size={11} className={cameraMode === 'single' ? 'text-emerald-400' : ''} />
                <span>Custom CCTV</span>
              </button>
              <button
                onClick={() => {
                  playHapticSound('snap');
                  setCameraMode('quad');
                  setActiveCameraFilter(null);
                }}
                title="4-Corner Distributed CCTV Array"
                className={`px-2 py-1 rounded-md transition-all font-bold cursor-pointer flex items-center gap-1 ${
                  cameraMode === 'quad'
                    ? 'bg-neutral-900 text-white shadow-2xs'
                    : 'text-neutral-500 hover:text-neutral-900'
                }`}
              >
                <span>4 Corners</span>
              </button>
            </div>
          )}

          {/* User-Placeable CCTV Camera Controls & Quick Snap Presets */}
          {viewMode === 'floorplan' && cameraMode === 'single' && (
            <div className="flex items-center bg-white p-0.5 rounded-lg border border-neutral-200 gap-0.5">
              {/* Place / Reposition Button */}
              <button
                onClick={() => {
                  playHapticSound('snap');
                  setIsPlacingCam(!isPlacingCam);
                }}
                title={customCamPos ? "Click anywhere on room to reposition CCTV" : "Click anywhere on room to place CCTV camera"}
                className={`px-2 py-1 rounded-md transition-all font-bold cursor-pointer flex items-center gap-1 ${
                  isPlacingCam
                    ? 'bg-sky-600 text-white shadow-2xs animate-pulse ring-2 ring-sky-300'
                    : customCamPos
                    ? 'text-neutral-700 hover:text-neutral-950 hover:bg-neutral-100'
                    : 'bg-sky-50 text-sky-800 border border-sky-300 hover:bg-sky-100 shadow-2xs'
                }`}
              >
                <MapPin size={11} className={isPlacingCam ? 'text-white animate-bounce' : 'text-sky-600'} />
                <span>{isPlacingCam ? 'Click Map to Place' : customCamPos ? 'Reposition' : '📍 Place Camera'}</span>
              </button>

              {/* Draggable Camera Chip for direct drag-onto-map */}
              {!customCamPos && (
                <div
                  onPointerDown={(e) => {
                    e.preventDefault();
                    startCameraDrag();
                  }}
                  title="Drag this CAM-01 badge directly onto the floorplan"
                  className="px-1.5 py-1 rounded-md bg-neutral-900 text-white flex items-center gap-1 cursor-grab active:cursor-grabbing font-mono text-[7.5px] hover:bg-neutral-800 shadow-2xs select-none"
                >
                  <Move size={8} className="text-sky-400" />
                  <span>Drag CAM-01</span>
                </div>
              )}

              {/* Camera Aim Angle Cycler (When camera is placed) */}
              {customCamPos && (
                <button
                  onClick={() => {
                    playHapticSound('click');
                    setCamAimAngle((prev) => {
                      if (prev === null) return 0;
                      if (prev === 270) return null;
                      return prev + 90;
                    });
                  }}
                  title={`Aim: ${camAimAngle !== null ? camAimAngle + '°' : 'Auto (Center)'}. Click to rotate aim.`}
                  className="px-1.5 py-1 text-[7.5px] rounded text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 border-l border-neutral-200 font-bold"
                >
                  Aim: {camAimAngle !== null ? `${camAimAngle}°` : 'Auto'}
                </button>
              )}

              {/* Quick Snaps */}
              <div className="flex items-center pl-1 border-l border-neutral-200">
                <span className="text-[7px] text-neutral-400 font-bold uppercase mr-1">Snap:</span>
                <button
                  onClick={() => {
                    playHapticSound('snap');
                    setCustomCamPos({ x: 5, y: 50 });
                    setIsPlacingCam(false);
                  }}
                  title="Snap camera to Aisle Head (West)"
                  className="px-1.5 py-0.5 text-[7.5px] rounded text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100"
                >
                  Head
                </button>
                <button
                  onClick={() => {
                    playHapticSound('snap');
                    setCustomCamPos({ x: 50, y: 50 });
                    setIsPlacingCam(false);
                  }}
                  title="Snap camera to Room Center (Ceiling)"
                  className="px-1.5 py-0.5 text-[7.5px] rounded text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100"
                >
                  Center
                </button>
                <button
                  onClick={() => {
                    playHapticSound('snap');
                    setCustomCamPos({ x: 95, y: 50 });
                    setIsPlacingCam(false);
                  }}
                  title="Snap camera to Aisle End (East)"
                  className="px-1.5 py-0.5 text-[7.5px] rounded text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100"
                >
                  End
                </button>
              </div>

              {/* Remove Button */}
              {customCamPos && (
                <button
                  onClick={() => {
                    playHapticSound('snap');
                    setCustomCamPos(null);
                    setIsPlacingCam(false);
                    setActiveCameraFilter(null);
                  }}
                  title="Remove CCTV camera from room"
                  className="px-1.5 py-1 text-[7.5px] rounded text-rose-600 hover:text-rose-800 hover:bg-rose-50 border-l border-neutral-200 font-bold ml-0.5 cursor-pointer"
                >
                  ✕ Remove
                </button>
              )}
            </div>
          )}

          {/* Orientation Mode: Wide (0°) vs Tall (90°) vs Rotate 90° */}
          <div className="flex items-center bg-white p-0.5 rounded-lg border border-neutral-200">
            <button
              onClick={() => {
                playHapticSound('snap');
                setRotation(0);
              }}
              title="Width-wise Horizontal Deck (0° Landscape)"
              className={`px-2 py-1 rounded-md transition-all font-bold cursor-pointer flex items-center gap-1 ${
                rotation === 0
                  ? 'bg-neutral-900 text-white shadow-2xs'
                  : 'text-neutral-500 hover:text-neutral-900'
              }`}
            >
              <span>↔ Wide</span>
            </button>
            <button
              onClick={() => {
                playHapticSound('snap');
                setRotation(90);
              }}
              title="Length-wise Vertical Deck (90° Portrait)"
              className={`px-2 py-1 rounded-md transition-all font-bold cursor-pointer flex items-center gap-1 ${
                rotation === 90
                  ? 'bg-neutral-900 text-white shadow-2xs'
                  : 'text-neutral-500 hover:text-neutral-900'
              }`}
            >
              <span>↕ Tall</span>
            </button>
            <button
              onClick={() => {
                playHapticSound('snap');
                setRotation((prev) => ((prev + 90) % 360) as 0 | 90 | 180 | 270);
              }}
              title="Rotate Entire View +90° Clockwise"
              className="px-1.5 py-1 rounded-md transition-all font-bold cursor-pointer text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 flex items-center gap-0.5 ml-0.5 border-l border-neutral-200"
            >
              <RotateCw size={11} className={`transition-transform duration-300 ${rotation === 90 ? 'rotate-90' : rotation === 180 ? 'rotate-180' : rotation === 270 ? '-rotate-90' : ''}`} />
              <span className="text-[7.5px] font-mono">{rotation}°</span>
            </button>
          </div>

          {viewMode === 'floorplan' && (
            <>
              {/* Vision Cones Toggle */}
              <button
                onClick={() => {
                  playHapticSound('click');
                  setShowFovCones(!showFovCones);
                }}
                className={`px-2 py-1 rounded-lg border transition-all cursor-pointer flex items-center gap-1 ${
                  showFovCones
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-bold'
                    : 'bg-white border-neutral-200 text-neutral-500 hover:text-neutral-900'
                }`}
              >
                <Eye size={11} />
                <span>Vision</span>
              </button>

              {/* Adjacency Graph Toggle */}
              <button
                onClick={() => {
                  playHapticSound('click');
                  setShowAdjacencyGraph(!showAdjacencyGraph);
                }}
                className={`px-2 py-1 rounded-lg border transition-all cursor-pointer flex items-center gap-1 ${
                  showAdjacencyGraph
                    ? 'bg-violet-50 border-violet-300 text-violet-900 font-bold'
                    : 'bg-white border-neutral-200 text-neutral-500 hover:text-neutral-900'
                }`}
              >
                <Layers size={11} />
                <span>Vectors</span>
              </button>
            </>
          )}

          {/* Design Studio Button */}
          <Button
            variant="default"
            size="xs"
            onClick={() => {
              playHapticSound('snap');
              setIsStudioOpen(true);
            }}
            className="gap-1 rounded-lg font-mono text-3xs shadow-2xs h-7 px-2"
          >
            <Sliders size={11} />
            <span>Design Studio</span>
          </Button>
        </div>
      </div>

      {/* Camera Placement Active Hint Banner */}
      {isPlacingCam && viewMode === 'floorplan' && cameraMode === 'single' && (
        <div className="py-1 px-3 bg-sky-50 text-sky-900 border border-sky-200 rounded-xl text-3xs font-mono flex items-center justify-between animate-fadeIn">
          <span className="flex items-center gap-1.5">
            <MapPin size={11} className="text-sky-600 animate-bounce" />
            <strong>Interactive Placement Active:</strong> Click anywhere on the floorplan to position CCTV, or drag the CAM-01 badge directly.
          </span>
          <button
            onClick={() => setIsPlacingCam(false)}
            className="text-sky-700 underline font-bold cursor-pointer"
          >
            Done
          </button>
        </div>
      )}

      {/* ================= VIEW 1: COMPACT ARCHITECTURAL ROOM FLOORPLAN ================= */}
      {viewMode === 'floorplan' ? (
        <div
          ref={containerRef}
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={handleCanvasMouseLeave}
          className={`relative w-full ${
            isPortrait
              ? 'max-w-[480px] sm:max-w-[530px] mx-auto aspect-[10/16] min-h-[540px] max-h-[680px]'
              : 'aspect-[16/10] min-h-[360px] max-h-[460px]'
          } ${isPlacingCam ? 'cursor-crosshair ring-2 ring-sky-400' : ''} bg-white rounded-2xl border-2 border-neutral-900 shadow-xs overflow-hidden select-none transition-all duration-300`}
        >
          {/* Subtle architectural room background grid texture */}
          <div
            className="absolute inset-0 pointer-events-none opacity-30"
            style={{
              backgroundImage: 'radial-gradient(circle, #cbd5e1 1px, transparent 1px)',
              backgroundSize: '20px 20px',
            }}
          />

          {/* SVG Overlay for Vision Cones & Adjacency Graph */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none z-0"
            viewBox="0 0 1000 1000"
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="fov-wedge-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#0284c7" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#0284c7" stopOpacity="0.01" />
              </linearGradient>
              <radialGradient
                id="fov-cam-grad"
                cx={`${visualCamPos ? visualCamPos.x : 50}%`}
                cy={`${visualCamPos ? visualCamPos.y : 50}%`}
                r="90%"
                fx={`${visualCamPos ? visualCamPos.x : 50}%`}
                fy={`${visualCamPos ? visualCamPos.y : 50}%`}
              >
                <stop offset="0%" stopColor="#0284c7" stopOpacity="0.30" />
                <stop offset="35%" stopColor="#0284c7" stopOpacity="0.14" />
                <stop offset="75%" stopColor="#0284c7" stopOpacity="0.04" />
                <stop offset="100%" stopColor="#0284c7" stopOpacity="0.0" />
              </radialGradient>
              <clipPath id="room-bounds">
                <rect x="0" y="0" width="1000" height="1000" rx="16" />
              </clipPath>
            </defs>

            {/* Vision Cones / Dynamic Coverage Field from User Placement */}
            {showFovCones && (
              cameraMode === 'single' ? (
                visualCamPos ? (
                  <g className="transition-opacity duration-200" clipPath="url(#room-bounds)">
                    {(() => {
                      const Xc = visualCamPos.x * 10;
                      const Yc = visualCamPos.y * 10;
                      const dx = 500 - Xc;
                      const dy = 500 - Yc;
                      const distFromCenter = Math.sqrt(dx * dx + dy * dy);

                      // If user placed camera near room center & aim is Auto, render 360° omnidirectional radial coverage
                      if (distFromCenter < 80 && camAimAngle === null) {
                        return (
                          <>
                            <ellipse
                              cx={Xc}
                              cy={Yc}
                              rx={isPortrait ? 420 : 480}
                              ry={isPortrait ? 480 : 420}
                              fill="url(#fov-cam-grad)"
                              opacity={!activeCameraFilter || activeCameraFilter === 'CAM-01' ? 1 : 0.15}
                            />
                            <ellipse
                              cx={Xc}
                              cy={Yc}
                              rx={isPortrait ? 420 : 480}
                              ry={isPortrait ? 480 : 420}
                              fill="none"
                              stroke="#0284c7"
                              strokeWidth="1.5"
                              strokeDasharray="4 4"
                              opacity={!activeCameraFilter || activeCameraFilter === 'CAM-01' ? 0.35 : 0.1}
                            />
                            <line x1={Math.max(40, Xc - 400)} y1={Yc} x2={Math.min(960, Xc + 400)} y2={Yc} stroke="#0284c7" strokeWidth="0.75" strokeDasharray="3 3" opacity={0.25} />
                            <line x1={Xc} y1={Math.max(40, Yc - 400)} x2={Xc} y2={Math.min(960, Yc + 400)} stroke="#0284c7" strokeWidth="0.75" strokeDasharray="3 3" opacity={0.25} />
                          </>
                        );
                      }

                      // Directional frustum: respect user's custom aim angle OR auto-aim toward center of room
                      let angleRad = Math.atan2(dy, dx);
                      if (camAimAngle !== null) {
                        const effectiveAimDeg = (camAimAngle + rotation) % 360;
                        angleRad = (effectiveAimDeg * Math.PI) / 180;
                      }

                      const fovRad = 52 * (Math.PI / 180);
                      const R = 1500;

                      const x1 = Xc + R * Math.cos(angleRad - fovRad);
                      const y1 = Yc + R * Math.sin(angleRad - fovRad);
                      const xmid = Xc + R * Math.cos(angleRad);
                      const ymid = Yc + R * Math.sin(angleRad);
                      const x2 = Xc + R * Math.cos(angleRad + fovRad);
                      const y2 = Yc + R * Math.sin(angleRad + fovRad);

                      const conePoints = `${Xc},${Yc} ${x1},${y1} ${xmid},${ymid} ${x2},${y2}`;

                      return (
                        <>
                          <polygon
                            points={conePoints}
                            fill="url(#fov-cam-grad)"
                            opacity={!activeCameraFilter || activeCameraFilter === 'CAM-01' ? 1 : 0.15}
                          />
                          <polygon
                            points={conePoints}
                            fill="none"
                            stroke="#0284c7"
                            strokeWidth="1.5"
                            strokeDasharray="4 4"
                            opacity={!activeCameraFilter || activeCameraFilter === 'CAM-01' ? 0.4 : 0.1}
                          />
                          {/* Optical Boresight Center Axis */}
                          <line
                            x1={Xc}
                            y1={Yc}
                            x2={xmid}
                            y2={ymid}
                            stroke="#0284c7"
                            strokeWidth="1.2"
                            strokeDasharray="5 3"
                            opacity={0.35}
                          />
                        </>
                      );
                    })()}
                  </g>
                ) : isPlacingCam && mousePos ? (
                  /* Placement Ghost Frustum Preview */
                  <g className="pointer-events-none opacity-50" clipPath="url(#room-bounds)">
                    {(() => {
                      const Xm = mousePos.x * 10;
                      const Ym = mousePos.y * 10;
                      const dx = 500 - Xm;
                      const dy = 500 - Ym;
                      const angleRad = Math.atan2(dy, dx);
                      const fovRad = 50 * (Math.PI / 180);
                      const R = 750;
                      const x1 = Xm + R * Math.cos(angleRad - fovRad);
                      const y1 = Ym + R * Math.sin(angleRad - fovRad);
                      const x2 = Xm + R * Math.cos(angleRad + fovRad);
                      const y2 = Ym + R * Math.sin(angleRad + fovRad);
                      return (
                        <>
                          <polygon
                            points={`${Xm},${Ym} ${x1},${y1} ${x2},${y2}`}
                            fill="#0284c7"
                            fillOpacity="0.15"
                            stroke="#0284c7"
                            strokeWidth="1.2"
                            strokeDasharray="4 4"
                          />
                          <circle cx={Xm} cy={Ym} r="16" fill="#0284c7" fillOpacity="0.3" stroke="#0284c7" strokeWidth="1.5" />
                        </>
                      );
                    })()}
                  </g>
                ) : null
              ) : (
                <g className="transition-opacity duration-200">
                  {Object.entries(CAMERA_COORDINATES).map(([camId, cam]) => {
                    const isActive = !activeCameraFilter || activeCameraFilter === camId;
                    const rotatedCone = rotatePolyPoints(cam.cone, rotation);
                    return (
                      <polygon
                        key={camId}
                        points={rotatedCone}
                        fill="url(#fov-wedge-grad)"
                        opacity={isActive ? 1 : 0.15}
                        className="transition-opacity duration-200"
                      />
                    );
                  })}
                </g>
              )
            )}

            {/* Adjacency Collusion Vectors */}
            {showAdjacencyGraph && (
              <g className="transition-opacity duration-200">
                {resolvedSeats.map((seat) => {
                  const p1 = rotateSvgPoint(
                    (seat.spatialPosition?.x || 50) * 10,
                    (seat.spatialPosition?.y || 50) * 10,
                    rotation
                  );

                  return (seat.spatialPosition?.neighbors || []).map((neighborId) => {
                    const neighbor = resolvedSeats.find((s) => s.id === neighborId);
                    if (!neighbor) return null;
                    const p2 = rotateSvgPoint(
                      (neighbor.spatialPosition?.x || 50) * 10,
                      (neighbor.spatialPosition?.y || 50) * 10,
                      rotation
                    );

                    if (seat.id > neighbor.id) return null;

                    const isHighRisk = seat.state === 'escalated' || neighbor.state === 'escalated';

                    return (
                      <line
                        key={`${seat.id}-${neighbor.id}`}
                        x1={p1.x}
                        y1={p1.y}
                        x2={p2.x}
                        y2={p2.y}
                        stroke={isHighRisk ? '#f43f5e' : '#cbd5e1'}
                        strokeWidth={isHighRisk ? 2 : 1}
                        strokeDasharray={isHighRisk ? '3 2' : '2 2'}
                        opacity={0.8}
                      />
                    );
                  });
                })}
              </g>
            )}
          </svg>

          {/* ---------------- 1. WHITEBOARD (ROTATES WITH ROOM ORIENTATION) ---------------- */}
          {rotation === 0 && (
            <div className="absolute top-0 left-1/2 -translate-x-1/2 z-10 w-44 sm:w-64 h-5 bg-neutral-950 text-white rounded-b-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex items-center justify-center gap-1.5 border-t-0">
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
              <span>WHITEBOARD / PRESENTATION SCREEN</span>
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
            </div>
          )}
          {rotation === 90 && (
            <div className="absolute right-0 top-1/2 -translate-y-1/2 z-10 w-5 h-44 sm:h-64 bg-neutral-950 text-white rounded-l-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex flex-col items-center justify-center gap-1.5 border-r-0">
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
              <span className="[writing-mode:vertical-rl] rotate-180">WHITEBOARD / SCREEN</span>
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
            </div>
          )}
          {rotation === 180 && (
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 z-10 w-44 sm:w-64 h-5 bg-neutral-950 text-white rounded-t-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex items-center justify-center gap-1.5 border-b-0">
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
              <span>WHITEBOARD / PRESENTATION SCREEN</span>
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
            </div>
          )}
          {rotation === 270 && (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-5 h-44 sm:h-64 bg-neutral-950 text-white rounded-r-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex flex-col items-center justify-center gap-1.5 border-l-0">
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
              <span className="[writing-mode:vertical-rl]">WHITEBOARD / SCREEN</span>
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
            </div>
          )}

          {/* ---------------- 2. PROCTOR PODIUM (DYNAMICALLY ROTATED POSITION) ---------------- */}
          {(() => {
            const podiumPos = getRotatedPoint(50, 13, rotation);
            return (
              <div
                style={{
                  position: 'absolute',
                  left: `${podiumPos.x}%`,
                  top: `${podiumPos.y}%`,
                  transform: 'translate(-50%, -50%)',
                }}
                className="z-10 h-5 px-2 bg-neutral-100/90 border border-neutral-300 rounded-md shadow-2xs flex items-center gap-1.5 text-[8px] font-mono text-neutral-700 whitespace-nowrap"
              >
                <span className="flex items-center gap-1 font-bold">
                  <Shield size={9} className="text-neutral-700" />
                  <span>PODIUM</span>
                </span>
                <span className="flex items-center gap-0.5 text-neutral-500 text-[7px]">
                  <Laptop size={8} />
                  <span>Desk</span>
                </span>
              </div>
            );
          })()}

          {/* ---------------- 3. CAMERA SETUP: USER-PLACEABLE CCTV OR 4 CORNERS ---------------- */}
          {cameraMode === 'single' ? (
            customCamPos && visualCamPos ? (
              /* User-Placed CCTV Camera Component (Freely Draggable Anywhere) */
              <div
                onPointerDown={handlePointerDownCam}
                onClick={(e) => {
                  e.stopPropagation();
                  if (!isDraggingCam) {
                    playHapticSound('click');
                    setActiveCameraFilter(activeCameraFilter === 'CAM-01' ? null : 'CAM-01');
                  }
                }}
                style={{
                  position: 'absolute',
                  left: `${visualCamPos.x}%`,
                  top: `${visualCamPos.y}%`,
                  transform: 'translate(-50%, -50%)',
                  touchAction: 'none',
                }}
                className={`z-25 h-7 px-2.5 rounded-full border flex items-center gap-1.5 cursor-grab active:cursor-grabbing shadow-lg font-mono text-[8px] transition-shadow select-none whitespace-nowrap ${
                  isDraggingCam
                    ? 'bg-sky-950 text-white ring-2 ring-sky-400 scale-105 shadow-2xl'
                    : activeCameraFilter === 'CAM-01'
                    ? 'bg-neutral-900 text-white border-neutral-900 ring-2 ring-sky-400'
                    : 'bg-white/95 text-neutral-900 border-neutral-300 hover:border-neutral-500 hover:shadow-xl'
                }`}
                title="CAM-01: Drag to place anywhere in the room. Click to filter telemetry."
              >
                <Move size={9} className="opacity-60 text-sky-500" />
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <Camera size={10} className="text-sky-600" />
                <span className="font-black text-[8.5px]">CAM-01</span>
                <span className="text-[7px] font-bold px-1 py-0.2 rounded bg-sky-100 text-sky-800">
                  {customCamPos.x}%, {customCamPos.y}%
                </span>

                {/* Aim cycler button on badge */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    playHapticSound('click');
                    setCamAimAngle((prev) => {
                      if (prev === null) return 0;
                      if (prev === 270) return null;
                      return prev + 90;
                    });
                  }}
                  title={`Aim: ${camAimAngle !== null ? camAimAngle + '°' : 'Auto'}. Click to rotate.`}
                  className="px-1 py-0.2 rounded bg-neutral-100 hover:bg-neutral-200 text-neutral-700 text-[6.5px] font-bold cursor-pointer"
                >
                  {camAimAngle !== null ? `${camAimAngle}°` : 'Auto'}
                </button>

                {/* Remove button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    playHapticSound('snap');
                    setCustomCamPos(null);
                    setIsPlacingCam(false);
                    setActiveCameraFilter(null);
                  }}
                  title="Remove CCTV Camera from room"
                  className="w-3.5 h-3.5 rounded-full flex items-center justify-center text-neutral-400 hover:text-rose-600 hover:bg-rose-50 text-[8px] font-bold cursor-pointer"
                >
                  ✕
                </button>
              </div>
            ) : isPlacingCam ? (
              /* Placement Active Crosshair Marker */
              mousePos ? (
                <div
                  style={{
                    position: 'absolute',
                    left: `${mousePos.x}%`,
                    top: `${mousePos.y}%`,
                    transform: 'translate(-50%, -50%)',
                    pointerEvents: 'none',
                  }}
                  className="z-30 h-6 px-2 rounded-full bg-sky-600/90 text-white shadow-lg font-mono text-[7.5px] flex items-center gap-1 border border-white/50 animate-pulse select-none"
                >
                  <MapPin size={9} />
                  <span>Click to Mount CAM-01</span>
                </div>
              ) : (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-25 pointer-events-none bg-sky-950/80 text-white backdrop-blur-md px-3.5 py-1.5 rounded-xl border border-sky-400/50 shadow-2xl flex items-center gap-2 text-3xs font-mono animate-pulse">
                  <MapPin size={12} className="text-sky-400 animate-bounce" />
                  <span>Click anywhere on floorplan to mount CCTV Camera</span>
                </div>
              )
            ) : null
          ) : (
            /* 4 Distributed Corner CCTV Cameras */
            (() => {
              const currentCams = CAMERA_CORNER_MAP[rotation];
              const corners = [
                { key: 'NW', camId: currentCams.NW, posClass: 'top-2 left-2' },
                { key: 'NE', camId: currentCams.NE, posClass: 'top-2 right-2' },
                { key: 'SW', camId: currentCams.SW, posClass: 'bottom-2 left-2' },
                { key: 'SE', camId: currentCams.SE, posClass: 'bottom-2 right-2' },
              ];

              return corners.map(({ camId, posClass }) => (
                <div
                  key={camId}
                  onClick={() => {
                    playHapticSound('click');
                    setActiveCameraFilter(activeCameraFilter === camId ? null : camId);
                  }}
                  className={`absolute ${posClass} z-20 h-5 px-1.5 rounded-md border flex items-center gap-1 cursor-pointer shadow-2xs font-mono text-[8px] transition-all ${
                    activeCameraFilter === camId
                      ? 'bg-neutral-900 text-white border-neutral-900 ring-1 ring-sky-300'
                      : 'bg-white/95 text-neutral-800 border-neutral-300 hover:border-neutral-500'
                  }`}
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  <Camera size={9} />
                  <span className="font-bold">{camId}</span>
                </div>
              ));
            })()
          )}

          {/* ---------------- 4. CENTRAL PROCTOR WALKWAY ---------------- */}
          {!isPortrait ? (
            <div className="absolute top-[50%] left-1/2 -translate-x-1/2 -translate-y-1/2 z-5 w-3/4 py-0.5 px-3 rounded-md border border-dashed border-neutral-300 bg-neutral-50/70 flex items-center justify-center font-mono text-[7px] sm:text-[8px] font-bold text-neutral-400 tracking-wider pointer-events-none">
              <span>[ CENTRAL PROCTOR WALKWAY • 2.4M SEPARATION AISLE ]</span>
            </div>
          ) : (
            <div className="absolute left-[50%] top-1/2 -translate-x-1/2 -translate-y-1/2 z-5 h-3/4 w-6 py-3 px-0.5 rounded-md border border-dashed border-neutral-300 bg-neutral-50/70 flex items-center justify-center font-mono text-[7px] sm:text-[8px] font-bold text-neutral-400 tracking-wider pointer-events-none [writing-mode:vertical-rl] rotate-180">
              <span>[ CENTRAL PROCTOR WALKWAY • 2.4M AISLE ]</span>
            </div>
          )}

          {/* ---------------- 5. DYNAMIC WORKSTATION CARDS (ROTATED) ---------------- */}
          <div className="absolute inset-0 pointer-events-none">
            {resolvedSeats.map((seat) => {
              const rawX = seat.spatialPosition?.x ?? 50;
              const rawY = seat.spatialPosition?.y ?? 50;
              const rotated = getRotatedPoint(rawX, rawY, rotation);

              // Safe bounds clamp to prevent wall clipping
              const safeX = Math.max(isPortrait ? 10.0 : 6.0, Math.min(isPortrait ? 90.0 : 94.0, rotated.x));
              const safeY = Math.max(isPortrait ? 6.0 : 16.0, Math.min(isPortrait ? 94.0 : 88.0, rotated.y));

              return (
                <div
                  key={seat.id}
                  style={{
                    position: 'absolute',
                    left: `${safeX}%`,
                    top: `${safeY}%`,
                    transform: 'translate(-50%, -50%)',
                    pointerEvents: 'auto',
                  }}
                >
                  {renderWorkstationCard(seat)}
                </div>
              );
            })}
          </div>

          {/* ---------------- 6. ENTRANCE DOUBLE DOORS (ROTATES WITH ROOM) ---------------- */}
          {rotation === 0 && (
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 z-10 w-36 sm:w-48 h-4 bg-neutral-950 text-white rounded-t-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex items-center justify-center gap-1.5 border-b-0">
              <span>🚪 ENTRANCE DOUBLE DOORS</span>
            </div>
          )}
          {rotation === 90 && (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10 h-36 sm:h-48 w-4 bg-neutral-950 text-white rounded-r-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex items-center justify-center border-l-0">
              <span className="[writing-mode:vertical-rl] rotate-180">🚪 ENTRANCE DOORS</span>
            </div>
          )}
          {rotation === 180 && (
            <div className="absolute top-0 left-1/2 -translate-x-1/2 z-10 w-36 sm:w-48 h-4 bg-neutral-950 text-white rounded-b-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex items-center justify-center gap-1.5 border-t-0">
              <span>🚪 ENTRANCE DOUBLE DOORS</span>
            </div>
          )}
          {rotation === 270 && (
            <div className="absolute right-0 top-1/2 -translate-y-1/2 z-10 h-36 sm:h-48 w-4 bg-neutral-950 text-white rounded-l-md shadow-2xs text-center font-mono text-[8px] font-bold tracking-wider flex items-center justify-center border-r-0">
              <span className="[writing-mode:vertical-rl]">🚪 ENTRANCE DOORS</span>
            </div>
          )}
        </div>
      ) : (
        /* ================= VIEW 2: COMPACT LINEAR GRID ================= */
        !isPortrait ? (
          <div className="space-y-4">
            {/* North Desk: Row A */}
            <div className="space-y-2">
              <div className="w-full py-1.5 bg-sky-50 rounded-lg text-center text-3xs font-mono font-bold text-sky-900 uppercase tracking-wider flex items-center justify-center gap-1.5 border border-sky-200/70">
                <Laptop size={11} className="text-sky-700" />
                <span>NORTH WALL DESK — ROW A (STUDENTS FACING NORTH)</span>
                <Laptop size={11} className="text-sky-700" />
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {rowA.map((seat) => renderWorkstationCard(seat, true))}
              </div>
            </div>

            {/* Central Proctor Walkway */}
            <div className="w-full py-1.5 bg-white rounded-lg border border-dashed border-neutral-300 flex items-center justify-center gap-2 shadow-2xs font-mono text-3xs text-neutral-600">
              <Shield size={11} className="text-neutral-500" />
              <span className="font-bold">CENTRAL PROCTOR WALKWAY</span>
              <Shield size={11} className="text-neutral-500" />
            </div>

            {/* South Desk: Row B */}
            <div className="space-y-2">
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {rowB.map((seat) => renderWorkstationCard(seat, true))}
              </div>

              <div className="w-full py-1.5 bg-sky-50 rounded-lg text-center text-3xs font-mono font-bold text-sky-900 uppercase tracking-wider flex items-center justify-center gap-1.5 border border-sky-200/70">
                <Laptop size={11} className="text-sky-700" />
                <span>SOUTH WALL DESK — ROW B (STUDENTS FACING SOUTH)</span>
                <Laptop size={11} className="text-sky-700" />
              </div>
            </div>
          </div>
        ) : (
          /* Portrait 2-Column Vertical Deck */
          <div className="grid grid-cols-2 gap-4 max-w-[500px] mx-auto p-2 bg-white rounded-2xl border border-neutral-200">
            {/* Column 1: Row B */}
            <div className="space-y-2">
              <div className="py-1 bg-sky-50 rounded-lg text-center text-3xs font-mono font-bold text-sky-900 flex items-center justify-center gap-1 border border-sky-200/70">
                <Laptop size={10} className="text-sky-700" />
                <span>ROW B (WEST)</span>
              </div>
              <div className="space-y-2">
                {rowB.map((seat) => renderWorkstationCard(seat, true))}
              </div>
            </div>

            {/* Column 2: Row A */}
            <div className="space-y-2">
              <div className="py-1 bg-sky-50 rounded-lg text-center text-3xs font-mono font-bold text-sky-900 flex items-center justify-center gap-1 border border-sky-200/70">
                <Laptop size={10} className="text-sky-700" />
                <span>ROW A (EAST)</span>
              </div>
              <div className="space-y-2">
                {rowA.map((seat) => renderWorkstationCard(seat, true))}
              </div>
            </div>
          </div>
        )
      )}

      {/* Real-time Interactive Hover Dossier Strip */}
      <div className="min-h-[50px]">
        <AnimatePresence mode="wait">
          {hoveredSeat ? (
            <motion.div
              key={hoveredSeat.id}
              initial={{ opacity: 0, y: 2 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="bg-[#181716] text-white rounded-xl p-2.5 text-xs flex items-center justify-between border border-neutral-800 shadow-sm flex-wrap gap-2"
            >
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-white text-neutral-950 font-mono font-black text-xs flex items-center justify-center shadow-xs">
                  {hoveredSeat.id}
                </div>
                <div>
                  <div className="font-heading font-bold text-xs text-white flex items-center gap-2">
                    <span>Candidate #{hoveredSeat.candidateId}</span>
                    <StatusBadge status={hoveredSeat.state} size="sm" />
                    {hoveredSeat.spatialPosition?.zoneName && (
                      <span className="text-4xs font-mono px-1.5 py-0.5 rounded bg-white/10 text-neutral-300">
                        {hoveredSeat.spatialPosition.zoneName}
                      </span>
                    )}
                  </div>
                  <div className="text-3xs text-neutral-400 font-mono">
                    {hoveredSeat.telemetry.lastEventSummary}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 text-3xs font-mono">
                <div className="bg-white/10 px-2 py-0.5 rounded border border-white/10">
                  <span className="text-neutral-400">Continuity: </span>
                  <strong className="text-emerald-400 font-bold">{hoveredSeat.telemetry.trackContinuity}</strong>
                </div>
                <div className="bg-white/10 px-2 py-0.5 rounded border border-white/10">
                  <span className="text-neutral-400">Gaze: </span>
                  <strong className="text-white font-bold">{hoveredSeat.telemetry.headEvidence}</strong>
                </div>
                <div className="bg-white/10 px-2 py-0.5 rounded border border-white/10">
                  <span className="text-neutral-400">Camera: </span>
                  <strong className="text-sky-300 font-bold">
                    {cameraMode === 'single'
                      ? customCamPos
                        ? `CAM-01 (${customCamPos.x}%, ${customCamPos.y}%)`
                        : 'Unplaced'
                      : (hoveredSeat.cameraIds?.[0] || 'CAM-01')}
                  </strong>
                </div>
                <Button
                  variant="default"
                  size="xs"
                  onClick={() => {
                    playHapticSound('snap');
                    onSelectSeat(hoveredSeat.id);
                  }}
                  className="bg-white text-neutral-950 hover:bg-neutral-100 rounded-lg px-2 py-0.5 font-bold shadow-xs text-3xs h-6"
                >
                  <span>Inspect</span>
                  <ArrowRight size={10} />
                </Button>
              </div>
            </motion.div>
          ) : (
            <div className="p-2 text-center text-3xs text-neutral-500 bg-white rounded-lg border border-neutral-200 font-mono flex items-center justify-center gap-3">
              {customCamPos ? (
                <span>
                  💡 <strong>CAM-01</strong> mounted at ({customCamPos.x}%, {customCamPos.y}%). Drag it anywhere in the room, or click <strong>Reposition</strong> / <strong>Remove</strong>.
                </span>
              ) : (
                <span>
                  💡 <strong>CCTV Camera is unplaced.</strong> Click <strong>📍 Place Camera</strong> or drag the CAM-01 token to mount a CCTV camera anywhere in the hall.
                </span>
              )}
              {activeCameraFilter && (
                <button
                  onClick={() => setActiveCameraFilter(null)}
                  className="text-sky-600 font-bold underline cursor-pointer ml-1"
                >
                  Reset {activeCameraFilter} filter
                </button>
              )}
            </div>
          )}
        </AnimatePresence>
      </div>

      {/* LAB LAYOUT STUDIO MODAL */}
      <LabLayoutStudioModal
        isOpen={isStudioOpen}
        onClose={() => setIsStudioOpen(false)}
        currentSeats={seats}
        onSaveSeats={(newSeats) => {
          onUpdateSeats?.(newSeats);
        }}
        cameras={cameras}
      />
    </div>
  );
};

function roundTo1(val: number): number {
  return Math.round(val * 10) / 10;
}

// Mathematical Rotation Helper for 2D Coordinate Space around center (50, 50)
function getRotatedPoint(x: number, y: number, rotation: 0 | 90 | 180 | 270): { x: number; y: number } {
  switch (rotation) {
    case 90:
      // Clockwise 90 deg: x' = 100 - y, y' = x
      return { x: roundTo1(100 - y), y: roundTo1(x) };
    case 180:
      // 180 deg: x' = 100 - x, y' = 100 - y
      return { x: roundTo1(100 - x), y: roundTo1(100 - y) };
    case 270:
      // 270 deg (or 90 deg CCW): x' = y, y' = 100 - x
      return { x: roundTo1(y), y: roundTo1(100 - x) };
    case 0:
    default:
      return { x: roundTo1(x), y: roundTo1(y) };
  }
}

// Inverse Rotation Helper: Convert from rotated visual coordinate back to canonical room coordinate
function unrotatePoint(x: number, y: number, rotation: 0 | 90 | 180 | 270): { x: number; y: number } {
  switch (rotation) {
    case 90:
      // x' = 100 - y, y' = x ==> x = y', y = 100 - x'
      return { x: roundTo1(y), y: roundTo1(100 - x) };
    case 180:
      return { x: roundTo1(100 - x), y: roundTo1(100 - y) };
    case 270:
      // x' = y, y' = 100 - x ==> x = 100 - y', y = x'
      return { x: roundTo1(100 - y), y: roundTo1(x) };
    case 0:
    default:
      return { x: roundTo1(x), y: roundTo1(y) };
  }
}

// Rotation Helper for SVG ViewBox (1000 x 1000)
function rotateSvgPoint(px: number, py: number, rotation: 0 | 90 | 180 | 270): { x: number; y: number } {
  switch (rotation) {
    case 90:
      return { x: 1000 - py, y: px };
    case 180:
      return { x: 1000 - px, y: 1000 - py };
    case 270:
      return { x: py, y: 1000 - px };
    case 0:
    default:
      return { x: px, y: py };
  }
}

function rotatePolyPoints(ptsStr: string, rotation: 0 | 90 | 180 | 270): string {
  if (rotation === 0) return ptsStr;
  return ptsStr
    .trim()
    .split(/\s+/)
    .map((pair) => {
      const [x, y] = pair.split(',').map(Number);
      const p = rotateSvgPoint(x, y, rotation);
      return `${p.x},${p.y}`;
    })
    .join(' ');
}

// Facing Direction Rotation Helper
function rotateFacing(
  facing: 'North' | 'South' | 'East' | 'West',
  rotation: 0 | 90 | 180 | 270
): 'North' | 'South' | 'East' | 'West' {
  const directions: ('North' | 'East' | 'South' | 'West')[] = ['North', 'East', 'South', 'West'];
  const currentIndex = directions.indexOf(facing);
  if (currentIndex === -1) return facing;
  const shift = rotation / 90;
  const newIndex = (currentIndex + shift) % 4;
  return directions[newIndex];
}

// Camera mapping to the 4 physical corners of the screen based on rotation
const CAMERA_CORNER_MAP: Record<0 | 90 | 180 | 270, { NW: string; NE: string; SW: string; SE: string }> = {
  0: { NW: 'CAM-02', NE: 'CAM-01', SW: 'CAM-03', SE: 'CAM-04' },
  90: { NW: 'CAM-03', NE: 'CAM-02', SW: 'CAM-04', SE: 'CAM-01' },
  180: { NW: 'CAM-04', NE: 'CAM-03', SW: 'CAM-01', SE: 'CAM-02' },
  270: { NW: 'CAM-01', NE: 'CAM-04', SW: 'CAM-02', SE: 'CAM-03' },
};
