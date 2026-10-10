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
  // CCTV Camera is mounted on the central walkway aisle track (Left <-> Right rail)
  const [customCamPos, setCustomCamPos] = useState<{ x: number; y: number } | null>({ x: 18, y: 50 });
  const [camAimAngle, setCamAimAngle] = useState<90 | 270>(90); // 90 = Right (East), 270 = Left (West)
  const [coneRangePercent, setConeRangePercent] = useState<number>(78); // Visual Cone Reach (e.g. 78% of lab hall deck, ~15M)
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

  // Universal Camera Drag Handler: Constrained strictly along the central aisle rail (Left <-> Right)
  const startCameraDrag = () => {
    setIsDraggingCam(true);
    setIsPlacingCam(false);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();

      if (!isPortrait) {
        // Landscape (0° / 180°): Drag horizontally along central aisle (y = 50% fixed)
        const rawX = ((moveEvent.clientX - rect.left) / rect.width) * 100;
        const clampedX = Math.max(6, Math.min(94, Math.round(rawX * 10) / 10));
        const canonical = unrotatePoint(clampedX, 50, rotation);
        setCustomCamPos(canonical);
      } else {
        // Portrait (90° / 270°): Aisle runs vertically (x = 50% fixed), drag along track
        const rawY = ((moveEvent.clientY - rect.top) / rect.height) * 100;
        const clampedY = Math.max(6, Math.min(94, Math.round(rawY * 10) / 10));
        const canonical = unrotatePoint(50, clampedY, rotation);
        setCustomCamPos(canonical);
      }
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

  // Track mouse position over canvas for placement preview (locked to central aisle)
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPlacingCam || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (!isPortrait) {
      const rawX = ((e.clientX - rect.left) / rect.width) * 100;
      const clampedX = Math.max(6, Math.min(94, Math.round(rawX * 10) / 10));
      setMousePos({ x: clampedX, y: 50 });
    } else {
      const rawY = ((e.clientY - rect.top) / rect.height) * 100;
      const clampedY = Math.max(6, Math.min(94, Math.round(rawY * 10) / 10));
      setMousePos({ x: 50, y: clampedY });
    }
  };

  const handleCanvasMouseLeave = () => {
    setMousePos(null);
  };

  // Handle clicking on room canvas to place camera along the central aisle rail
  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPlacingCam || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (!isPortrait) {
      const rawX = ((e.clientX - rect.left) / rect.width) * 100;
      const clampedX = Math.max(6, Math.min(94, Math.round(rawX * 10) / 10));
      const canonical = unrotatePoint(clampedX, 50, rotation);
      setCustomCamPos(canonical);
    } else {
      const rawY = ((e.clientY - rect.top) / rect.height) * 100;
      const clampedY = Math.max(6, Math.min(94, Math.round(rawY * 10) / 10));
      const canonical = unrotatePoint(50, clampedY, rotation);
      setCustomCamPos(canonical);
    }
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

  const getSeatStyle = (
    state: SeatNodeData['state'],
    isSelected: boolean,
    isDimmed: boolean,
    isBlindSpot = false,
    isActivelyCovered = false
  ) => {
    let base = 'transition-all duration-150 cursor-pointer flex flex-col justify-between select-none font-sans rounded-xl shadow-2xs ';

    if (isDimmed) {
      base += 'opacity-25 scale-95 ';
    } else if (isBlindSpot) {
      base += 'opacity-65 border-dashed border-neutral-300 bg-neutral-50/70 ';
    } else if (isActivelyCovered && !isSelected) {
      base += 'ring-1 ring-sky-400/60 shadow-xs shadow-sky-500/10 ';
    }

    if (isSelected) {
      base += 'bg-[#181716] text-[#FAF7F3] ring-2 ring-neutral-900 shadow-md border-neutral-900 scale-[1.05] z-30 ';
      return base;
    }

    if (isBlindSpot && state === 'normal') {
      return base + 'text-neutral-500 hover:border-neutral-400 z-10';
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

  // Real-World Geometric Optics: Test if a workstation seat lies within camera's field of view
  const getSeatCoverage = (seat: SeatNodeData): { isCovered: boolean; cameraName: string } => {
    if (cameraMode === 'quad') {
      return {
        isCovered: true,
        cameraName: seat.cameraIds?.[0] || 'CAM-01',
      };
    }

    if (!customCamPos) {
      return { isCovered: false, cameraName: 'UNPLACED' };
    }

    const sx = seat.spatialPosition?.x ?? 50;
    const sy = seat.spatialPosition?.y ?? 50;

    // Optical Physics: Light travels along forward vectors from the lens aperture.
    // The camera has zero visibility behind its casing (backward hemisphere is in the blind spot).
    if (camAimAngle === 90) {
      // Aiming East (Right): only seats ahead in +X are illuminated
      if (sx < customCamPos.x - 2) {
        return { isCovered: false, cameraName: 'BLIND' };
      }
      const dist = Math.hypot(sx - customCamPos.x, (sy - 50) * 1.5);
      if (dist > coneRangePercent) {
        return { isCovered: false, cameraName: 'OUT_OF_RANGE' };
      }
      return { isCovered: true, cameraName: 'CAM-01' };
    } else {
      // Aiming West (Left): only seats ahead in -X are illuminated
      if (sx > customCamPos.x + 2) {
        return { isCovered: false, cameraName: 'BLIND' };
      }
      const dist = Math.hypot(customCamPos.x - sx, (sy - 50) * 1.5);
      if (dist > coneRangePercent) {
        return { isCovered: false, cameraName: 'OUT_OF_RANGE' };
      }
      return { isCovered: true, cameraName: 'CAM-01' };
    }
  };

  // Helper to format contiguous numbers (e.g. [2..10, 12..20] -> "#2–#10, #12–#20")
  const formatNumberRanges = (nums: number[]): string => {
    if (nums.length === 0) return 'None';
    const sorted = [...new Set(nums)].sort((a, b) => a - b);
    const ranges: string[] = [];
    let start = sorted[0];
    let prev = sorted[0];

    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] === prev + 1) {
        prev = sorted[i];
      } else {
        ranges.push(start === prev ? `#${start}` : `#${start}–#${prev}`);
        start = sorted[i];
        prev = sorted[i];
      }
    }
    ranges.push(start === prev ? `#${start}` : `#${start}–#${prev}`);
    return ranges.join(', ');
  };

  // Helper to format contiguous workstation IDs (e.g. cols 2..10 of Row A -> "A02–A10")
  const formatSeatRanges = (rowSeats: SeatNodeData[]): string => {
    if (rowSeats.length === 0) return 'None';
    const prefix = rowSeats[0].gridPosition.row;
    const cols = [...new Set(rowSeats.map((s) => s.gridPosition.col))].sort((a, b) => a - b);
    const ranges: string[] = [];
    let start = cols[0];
    let prev = cols[0];

    for (let i = 1; i < cols.length; i++) {
      if (cols[i] === prev + 1) {
        prev = cols[i];
      } else {
        ranges.push(
          start === prev
            ? `${prefix}${String(start).padStart(2, '0')}`
            : `${prefix}${String(start).padStart(2, '0')}–${prefix}${String(prev).padStart(2, '0')}`
        );
        start = cols[i];
        prev = cols[i];
      }
    }
    ranges.push(
      start === prev
        ? `${prefix}${String(start).padStart(2, '0')}`
        : `${prefix}${String(start).padStart(2, '0')}–${prefix}${String(prev).padStart(2, '0')}`
    );
    return ranges.join(', ');
  };

  // Live Optical Surveillance Coverage Analysis
  const coverageData = useMemo(() => {
    const total = resolvedSeats.length;
    if (cameraMode === 'quad') {
      const rowAAll = resolvedSeats.filter((s) => s.gridPosition.row === 'A');
      const rowBAll = resolvedSeats.filter((s) => s.gridPosition.row === 'B');
      const candidateNums = resolvedSeats
        .map((s) => parseInt(s.candidateId || '0', 10))
        .filter((n) => n > 0);

      return {
        coveredCount: total,
        totalCount: total,
        percent: 100,
        candidateRange: formatNumberRanges(candidateNums),
        rowARange: formatSeatRanges(rowAAll),
        rowBRange: formatSeatRanges(rowBAll),
        blindIds: [] as string[],
      };
    }

    if (!customCamPos) {
      return {
        coveredCount: 0,
        totalCount: total,
        percent: 0,
        candidateRange: 'None (Unplaced)',
        rowARange: 'None',
        rowBRange: 'None',
        blindIds: resolvedSeats.map((s) => s.id),
      };
    }

    const covered = resolvedSeats.filter((s) => getSeatCoverage(s).isCovered);
    const blind = resolvedSeats.filter((s) => !getSeatCoverage(s).isCovered);
    const percent = total > 0 ? Math.round((covered.length / total) * 100) : 0;

    const rowACovered = covered.filter((s) => s.gridPosition.row === 'A');
    const rowBCovered = covered.filter((s) => s.gridPosition.row === 'B');

    const candidateNums = covered
      .map((s) => parseInt(s.candidateId || '0', 10))
      .filter((n) => n > 0);

    return {
      coveredCount: covered.length,
      totalCount: total,
      percent,
      candidateRange: formatNumberRanges(candidateNums),
      rowARange: formatSeatRanges(rowACovered),
      rowBRange: formatSeatRanges(rowBCovered),
      blindIds: blind.map((s) => s.id),
    };
  }, [cameraMode, customCamPos, camAimAngle, coneRangePercent, resolvedSeats]);

  const coveredSeatsCount = coverageData.coveredCount;

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
    const coverage = getSeatCoverage(seat);
    const assignedCam = coverage.cameraName;
    const isBlindSpot = cameraMode === 'single' && Boolean(customCamPos) && !coverage.isCovered;
    const isActivelyCovered = cameraMode === 'single' && Boolean(customCamPos) && coverage.isCovered;
    const isDimmed = Boolean(
      activeCameraFilter &&
      (cameraMode === 'single'
        ? activeCameraFilter === 'CAM-01' && !coverage.isCovered
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
          title={`${seat.id} • Candidate #${seat.candidateId} (${seat.candidateName})${
            isBlindSpot
              ? ' • ⚠️ Blind Spot (Behind CAM-01 lens aperture)'
              : isActivelyCovered
              ? ' • 👁️ Inside CAM-01 Optical Vision Cone'
              : ''
          }`}
          className={`${dynamicLayout.padding} ${
            isCompactGrid ? 'min-h-[76px] w-full' : ''
          } ${getSeatStyle(seat.state, isSelected, isDimmed, isBlindSpot, isActivelyCovered)}`}
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

          {/* Bottom Bar: Zone & Alerts (No camera tags) */}
          <div className="pt-0.5 border-t border-black/[0.05] flex items-center justify-between text-[8px] font-mono leading-none opacity-80">
            <span className="text-[7px] text-neutral-500 font-bold tracking-tight">
              {seat.gridPosition.row === 'A' ? 'ROW A' : 'ROW B'}
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
              <span className="text-3xs font-mono font-bold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-900 border border-emerald-300 flex items-center gap-1.5 shadow-2xs">
                <span className={`w-2 h-2 rounded-full ${customCamPos ? 'bg-emerald-500 animate-pulse' : 'bg-neutral-400'}`} />
                {cameraMode === 'single' ? (
                  customCamPos ? (
                    <span>
                      <strong>CCTV Live:</strong> Students <span className="text-emerald-700 underline font-extrabold">{coverageData.candidateRange}</span> ({coverageData.coveredCount}/{seats.length} Covered • {coverageData.percent}% FOV)
                    </span>
                  ) : (
                    <span>CCTV: Unplaced (0/20 Covered)</span>
                  )
                ) : (
                  <span>4 Distributed Feeds (100% Coverage • Students #1–#20)</span>
                )}
              </span>
              {cameraMode === 'single' && customCamPos && coverageData.blindIds.length > 0 && (
                <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200 flex items-center gap-1">
                  <span>Blind ({coverageData.blindIds.length}): {coverageData.blindIds.join(', ')}</span>
                </span>
              )}
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

              {/* Camera Aim Orientation Toggle (Left vs Right down the aisle) */}
              {customCamPos && (
                <button
                  onClick={() => {
                    playHapticSound('click');
                    setCamAimAngle((prev) => (prev === 90 ? 270 : 90));
                  }}
                  title={`Aim: ${camAimAngle === 270 ? 'Left (West)' : 'Right (East)'}. Click to switch direction.`}
                  className="px-2 py-1 text-[7.5px] rounded bg-sky-50 text-sky-800 hover:bg-sky-100 border-l border-neutral-200 font-bold flex items-center gap-1 cursor-pointer transition-colors"
                >
                  <span>{camAimAngle === 270 ? '◀ Aim Left' : '▶ Aim Right'}</span>
                </button>
              )}

              {/* Quick Aisle Rail Snaps */}
              <div className="flex items-center pl-1 border-l border-neutral-200">
                <span className="text-[7px] text-neutral-400 font-bold uppercase mr-1">Aisle:</span>
                <button
                  onClick={() => {
                    playHapticSound('snap');
                    setCustomCamPos({ x: 10, y: 50 });
                    setCamAimAngle(90);
                    setIsPlacingCam(false);
                  }}
                  title="Snap camera to Aisle Left (West, pointing Right)"
                  className="px-1.5 py-0.5 text-[7.5px] rounded text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 font-medium cursor-pointer"
                >
                  Left
                </button>
                <button
                  onClick={() => {
                    playHapticSound('snap');
                    setCustomCamPos({ x: 50, y: 50 });
                    setIsPlacingCam(false);
                  }}
                  title="Snap camera to Aisle Center"
                  className="px-1.5 py-0.5 text-[7.5px] rounded text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 font-medium cursor-pointer"
                >
                  Center
                </button>
                <button
                  onClick={() => {
                    playHapticSound('snap');
                    setCustomCamPos({ x: 90, y: 50 });
                    setCamAimAngle(270);
                    setIsPlacingCam(false);
                  }}
                  title="Snap camera to Aisle Right (East, pointing Left)"
                  className="px-1.5 py-0.5 text-[7.5px] rounded text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 font-medium cursor-pointer"
                >
                  Right
                </button>
              </div>

              {/* Visual Cone Optical Range Selector */}
              {customCamPos && (
                <div className="flex items-center pl-1 border-l border-neutral-200">
                  <span className="text-[7px] text-neutral-400 font-bold uppercase mr-1">Range:</span>
                  <button
                    onClick={() => {
                      playHapticSound('click');
                      setConeRangePercent(45);
                    }}
                    title="Short Optical Range (45% of hall, ~10M)"
                    className={`px-1.5 py-0.5 text-[7.5px] rounded font-bold cursor-pointer transition-colors ${
                      coneRangePercent === 45
                        ? 'bg-sky-600 text-white shadow-2xs'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                    }`}
                  >
                    10M
                  </button>
                  <button
                    onClick={() => {
                      playHapticSound('click');
                      setConeRangePercent(78);
                    }}
                    title="Standard Optical Range (78% of hall, ~15M)"
                    className={`px-1.5 py-0.5 text-[7.5px] rounded font-bold cursor-pointer transition-colors ${
                      coneRangePercent === 78
                        ? 'bg-sky-600 text-white shadow-2xs'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                    }`}
                  >
                    15M
                  </button>
                  <button
                    onClick={() => {
                      playHapticSound('click');
                      setConeRangePercent(95);
                    }}
                    title="Max Deck Range (95% full hall reach)"
                    className={`px-1.5 py-0.5 text-[7.5px] rounded font-bold cursor-pointer transition-colors ${
                      coneRangePercent === 95
                        ? 'bg-sky-600 text-white shadow-2xs'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                    }`}
                  >
                    Max
                  </button>
                </div>
              )}

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

          {/* Tactical Student Coverage HUD: Instant On-Map Range Display */}
          {cameraMode === 'single' && (
            <div className="absolute top-2.5 left-2.5 z-20 pointer-events-auto bg-neutral-950/90 backdrop-blur-md text-white border border-neutral-700/80 rounded-xl p-2.5 shadow-xl font-mono text-[8.5px] max-w-[270px] sm:max-w-[320px] transition-all">
              <div className="flex items-center justify-between pb-1 mb-1.5 border-b border-white/10 text-[7.5px] text-neutral-400 uppercase tracking-wider font-bold">
                <span className="flex items-center gap-1.5 text-sky-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  CCTV Live Coverage
                </span>
                <span className="text-neutral-400 font-normal">
                  Aim: {camAimAngle === 90 ? 'East ▶' : 'West ◀'} ({customCamPos ? `${customCamPos.x}%` : 'Off'})
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-1.5 mb-1">
                <span className="text-neutral-400 text-[8px] font-medium">Students Covered:</span>
                <span className="font-heading font-black text-xs text-emerald-400 tracking-tight">
                  {coverageData.candidateRange}
                </span>
              </div>
              <div className="space-y-0.5 text-[8px] text-neutral-300">
                <div className="flex items-center justify-between">
                  <span className="text-neutral-400">Row A Desks:</span>
                  <span className="font-bold text-sky-300">{coverageData.rowARange}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-neutral-400">Row B Desks:</span>
                  <span className="font-bold text-sky-300">{coverageData.rowBRange}</span>
                </div>
              </div>
              <div className="mt-1.5 pt-1 border-t border-white/10 flex items-center justify-between text-[7.5px]">
                <span className="text-neutral-400">Visual Cone Range:</span>
                <span className="font-bold text-sky-400">
                  {coneRangePercent}% Hall Reach ({Math.round(coneRangePercent * 0.2)}M)
                </span>
              </div>
              <div className="flex items-center justify-between text-[7.5px]">
                <span className="text-neutral-400">FOV Monitored:</span>
                <span className="font-bold text-emerald-400">
                  {coverageData.coveredCount} of {seats.length} ({coverageData.percent}%)
                </span>
              </div>
              {coverageData.blindIds.length > 0 && (
                <div className="mt-1 pt-1 border-t border-white/10 text-[7.5px] text-amber-300 flex items-center justify-between">
                  <span className="text-amber-400 font-bold">Blind Spot ({coverageData.blindIds.length}):</span>
                  <span className="font-mono text-amber-200">{coverageData.blindIds.join(', ')}</span>
                </div>
              )}
            </div>
          )}

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

            {/* Vision Cones: 4-Corner Distributed CCTV Array when in quad mode */}
            {showFovCones && cameraMode === 'quad' && (
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
              /* Precision CCTV Camera Component: Aisle Track Mounted with Left/Right Orientation */
              <div
                style={{
                  position: 'absolute',
                  left: `${visualCamPos.x}%`,
                  top: `${visualCamPos.y}%`,
                  transform: 'translate(-50%, -50%)',
                  touchAction: 'none',
                }}
                className="relative z-25 select-none w-8 h-8 flex items-center justify-center pointer-events-auto"
              >
                {/* 1. CO-LOCATED OPTICAL VISION CONE: Originates directly from (0, 0) of the camera lens, covering student territory */}
                {showFovCones && (!activeCameraFilter || activeCameraFilter === 'CAM-01') && (() => {
                  const R = Math.round(520 * (coneRangePercent / 100)); // Calibrated screen-pixel reach
                  const xRay = Math.round(R * 0.866);
                  const yRay = Math.round(R * 0.5);
                  const R_near = Math.round(R * 0.38);
                  const xNear = Math.round(R_near * 0.866);
                  const yNear = Math.round(R_near * 0.5);
                  const R_mid = Math.round(R * 0.70);
                  const xMid = Math.round(R_mid * 0.866);
                  const yMid = Math.round(R_mid * 0.5);
                  const svgSize = (R + 60) * 2;
                  const half = R + 60;

                  return (
                    <svg
                      className="absolute pointer-events-none"
                      style={{
                        left: '50%',
                        top: '50%',
                        transform: 'translate(-50%, -50%)',
                        width: `${svgSize}px`,
                        height: `${svgSize}px`,
                        overflow: 'visible',
                        zIndex: 1,
                      }}
                      viewBox={`-${half} -${half} ${svgSize} ${svgSize}`}
                    >
                      <defs>
                        <radialGradient id="cctv-range-glow" cx="0" cy="0" r={R} gradientUnits="userSpaceOnUse">
                          <stop offset="0%" stopColor="#0284c7" stopOpacity="0.45" />
                          <stop offset="35%" stopColor="#38bdf8" stopOpacity="0.25" />
                          <stop offset="70%" stopColor="#0284c7" stopOpacity="0.14" />
                          <stop offset="100%" stopColor="#0284c7" stopOpacity="0.04" />
                        </radialGradient>
                      </defs>

                      <g transform={`rotate(${rotation}, 0, 0)`} style={{ transformOrigin: '0px 0px' }}>
                        {/* 1. Main Optical Volume Wedge */}
                        <path
                          d={camAimAngle === 90
                            ? `M 0 0 L ${xRay} ${-yRay} A ${R} ${R} 0 0 1 ${xRay} ${yRay} Z`
                            : `M 0 0 L ${-xRay} ${-yRay} A ${R} ${R} 0 0 0 ${-xRay} ${yRay} Z`
                          }
                          fill="url(#cctv-range-glow)"
                        />

                        {/* 2. Outer Frustum Boundary Rays */}
                        <line
                          x1="0"
                          y1="0"
                          x2={camAimAngle === 90 ? xRay : -xRay}
                          y2={-yRay}
                          stroke="#0284c7"
                          strokeWidth="1.8"
                          strokeDasharray="6 3"
                          opacity={0.8}
                        />
                        <line
                          x1="0"
                          y1="0"
                          x2={camAimAngle === 90 ? xRay : -xRay}
                          y2={yRay}
                          stroke="#0284c7"
                          strokeWidth="1.8"
                          strokeDasharray="6 3"
                          opacity={0.8}
                        />

                        {/* 3. Curved Range Limit Boundary Arc (At Outer Range R) */}
                        <path
                          d={camAimAngle === 90
                            ? `M ${xRay} ${-yRay} A ${R} ${R} 0 0 1 ${xRay} ${yRay}`
                            : `M ${-xRay} ${-yRay} A ${R} ${R} 0 0 0 ${-xRay} ${yRay}`
                          }
                          fill="none"
                          stroke="#0284c7"
                          strokeWidth="2.5"
                          strokeDasharray="6 3"
                          opacity={0.9}
                        />

                        {/* 4. Near Distance Range Arc (5M) */}
                        <path
                          d={camAimAngle === 90
                            ? `M ${xNear} ${-yNear} A ${R_near} ${R_near} 0 0 1 ${xNear} ${yNear}`
                            : `M ${-xNear} ${-yNear} A ${R_near} ${R_near} 0 0 0 ${-xNear} ${yNear}`
                          }
                          fill="none"
                          stroke="#0284c7"
                          strokeWidth="1"
                          strokeDasharray="4 3"
                          opacity={0.55}
                        />
                        <text
                          x={camAimAngle === 90 ? R_near + 4 : -R_near - 22}
                          y="-4"
                          fill="#0284c7"
                          fontSize="9"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          5M
                        </text>

                        {/* 5. Mid Distance Range Arc (10M) */}
                        <path
                          d={camAimAngle === 90
                            ? `M ${xMid} ${-yMid} A ${R_mid} ${R_mid} 0 0 1 ${xMid} ${yMid}`
                            : `M ${-xMid} ${-yMid} A ${R_mid} ${R_mid} 0 0 0 ${-xMid} ${yMid}`
                          }
                          fill="none"
                          stroke="#0284c7"
                          strokeWidth="1.2"
                          strokeDasharray="4 3"
                          opacity={0.6}
                        />
                        <text
                          x={camAimAngle === 90 ? R_mid + 4 : -R_mid - 26}
                          y="-4"
                          fill="#0284c7"
                          fontSize="9"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          10M
                        </text>

                        {/* 6. Optical Boresight Center Axis */}
                        <line
                          x1="0"
                          y1="0"
                          x2={camAimAngle === 90 ? R : -R}
                          y2="0"
                          stroke="#0284c7"
                          strokeWidth="1.2"
                          strokeDasharray="6 4"
                          opacity={0.5}
                        />

                        {/* 7. Tactical Range Limit Badge on the Outer Arc */}
                        <g transform={`translate(${camAimAngle === 90 ? R : -R}, 0)`}>
                          <line x1="0" y1="-10" x2="0" y2="10" stroke="#0284c7" strokeWidth="2.5" />
                          <rect
                            x="-38"
                            y="-18"
                            width="76"
                            height="16"
                            rx="3"
                            fill="#082f49"
                            stroke="#38bdf8"
                            strokeWidth="1"
                            opacity="0.95"
                          />
                          <text
                            x="0"
                            y="-7"
                            textAnchor="middle"
                            fill="#38bdf8"
                            fontSize="8"
                            fontFamily="monospace"
                            fontWeight="bold"
                          >
                            RANGE {coneRangePercent}%
                          </text>
                        </g>

                        {/* 8. Lens Emitter Optical Rings at (0, 0) */}
                        <circle cx="0" cy="0" r="18" fill="#0284c7" fillOpacity="0.22" stroke="#0284c7" strokeWidth="1.2" strokeDasharray="3 3" />
                        <circle cx="0" cy="0" r="10" fill="#0284c7" fillOpacity="0.55" stroke="#ffffff" strokeWidth="0.8" />
                        <circle cx="0" cy="0" r="4" fill="#ffffff" />
                      </g>
                    </svg>
                  );
                })()}

                {/* 2. Precision Circular CCTV Lens Turret: Zero Gap Emitter & Direction Controller */}
                <div
                  onPointerDown={handlePointerDownCam}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!isDraggingCam) {
                      playHapticSound('click');
                      // Quick flip aim direction on click
                      setCamAimAngle((prev) => (prev === 90 ? 270 : 90));
                    }
                  }}
                  className={`w-8 h-8 rounded-full flex items-center justify-center cursor-grab active:cursor-grabbing shadow-xl border-2 transition-all relative group ${
                    isDraggingCam
                      ? 'bg-sky-950 border-sky-400 scale-110 shadow-sky-500/40 ring-2 ring-sky-300'
                      : activeCameraFilter === 'CAM-01'
                      ? 'bg-neutral-950 border-sky-400 ring-2 ring-sky-300 shadow-sky-400/20'
                      : 'bg-neutral-900 border-white hover:border-sky-300 hover:scale-105'
                  }`}
                  title={`CCTV Camera: Aiming ${camAimAngle === 90 ? 'Right (East)' : 'Left (West)'}. Click to switch aim direction, or drag along track.`}
                >
                  {/* Subtle Overhead Track Bracket */}
                  <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-1 bg-neutral-600 rounded-t-xs" />

                  {/* Directional Camera Lens Icon pointing Left or Right */}
                  <div className={`transition-transform duration-300 flex items-center justify-center ${
                    camAimAngle === 270 ? 'rotate-180 text-sky-400' : 'text-sky-400'
                  }`}>
                    <Camera size={13} />
                  </div>

                  {/* Direction Aim Arrow Badge */}
                  <div className={`absolute ${camAimAngle === 90 ? '-right-1.5' : '-left-1.5'} top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-sky-500 text-white flex items-center justify-center shadow-xs text-[6.5px] pointer-events-none font-bold`}>
                    {camAimAngle === 90 ? '▶' : '◀'}
                  </div>
                </div>
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
                  className="z-30 select-none relative w-8 h-8 flex items-center justify-center"
                >
                  {/* Ghost Vision Cone Territory Preview */}
                  {(() => {
                    const R = Math.round(520 * (coneRangePercent / 100));
                    const xRay = Math.round(R * 0.866);
                    const yRay = Math.round(R * 0.5);
                    const svgSize = (R + 60) * 2;
                    const half = R + 60;

                    return (
                      <svg
                        className="absolute pointer-events-none -z-10 opacity-60"
                        style={{
                          left: '50%',
                          top: '50%',
                          transform: 'translate(-50%, -50%)',
                          width: `${svgSize}px`,
                          height: `${svgSize}px`,
                          overflow: 'visible',
                        }}
                        viewBox={`-${half} -${half} ${svgSize} ${svgSize}`}
                      >
                        <g transform={`rotate(${rotation}, 0, 0)`} style={{ transformOrigin: '0px 0px' }}>
                          <path
                            d={camAimAngle === 270
                              ? `M 0 0 L ${-xRay} ${-yRay} A ${R} ${R} 0 0 0 ${-xRay} ${yRay} Z`
                              : `M 0 0 L ${xRay} ${-yRay} A ${R} ${R} 0 0 1 ${xRay} ${yRay} Z`
                            }
                            fill="#0284c7"
                            fillOpacity="0.22"
                            stroke="#0284c7"
                            strokeWidth="1.5"
                            strokeDasharray="4 4"
                          />
                        </g>
                      </svg>
                    );
                  })()}
                  <div className="w-8 h-8 rounded-full bg-neutral-900 border-2 border-sky-400 flex items-center justify-center opacity-75 shadow-lg">
                    <Camera size={13} className="text-sky-400" />
                  </div>
                </div>
              ) : (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-25 pointer-events-none bg-sky-950/80 text-white backdrop-blur-md px-3.5 py-1.5 rounded-xl border border-sky-400/50 shadow-2xl flex items-center gap-2 text-3xs font-mono animate-pulse">
                  <MapPin size={12} className="text-sky-400 animate-bounce" />
                  <span>Click anywhere along central aisle to mount CCTV Camera</span>
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

          {/* ---------------- 4. CENTRAL PROCTOR WALKWAY & OVERHEAD CEILING CAMERA RAIL ---------------- */}
          {!isPortrait ? (
            <>
              {/* Overhead Motorized CCTV Ceiling Track Rail */}
              {cameraMode === 'single' && (
                <div className="absolute top-[50%] left-[8%] right-[8%] -translate-y-1/2 z-6 h-[2px] bg-neutral-300/80 pointer-events-none rounded-full flex items-center justify-between px-6 opacity-60">
                  <span className="w-1 h-2 bg-neutral-400 rounded-xs" />
                  <span className="w-1 h-2 bg-neutral-400 rounded-xs" />
                  <span className="w-1 h-2 bg-neutral-400 rounded-xs" />
                  <span className="w-1 h-2 bg-neutral-400 rounded-xs" />
                  <span className="w-1 h-2 bg-neutral-400 rounded-xs" />
                </div>
              )}
              <div className="absolute top-[50%] left-1/2 -translate-x-1/2 -translate-y-1/2 z-5 w-3/4 py-0.5 px-3 rounded-md border border-dashed border-neutral-300 bg-neutral-50/70 flex items-center justify-center font-mono text-[7px] sm:text-[8px] font-bold text-neutral-400 tracking-wider pointer-events-none">
                <span>[ CENTRAL PROCTOR WALKWAY • 2.4M SEPARATION AISLE ]</span>
              </div>
            </>
          ) : (
            <>
              {/* Overhead Motorized CCTV Ceiling Track Rail (Vertical) */}
              {cameraMode === 'single' && (
                <div className="absolute left-[50%] top-[8%] bottom-[8%] -translate-x-1/2 z-6 w-[2px] bg-neutral-300/80 pointer-events-none rounded-full flex flex-col items-center justify-between py-6 opacity-60">
                  <span className="h-1 w-2 bg-neutral-400 rounded-xs" />
                  <span className="h-1 w-2 bg-neutral-400 rounded-xs" />
                  <span className="h-1 w-2 bg-neutral-400 rounded-xs" />
                  <span className="h-1 w-2 bg-neutral-400 rounded-xs" />
                  <span className="h-1 w-2 bg-neutral-400 rounded-xs" />
                </div>
              )}
              <div className="absolute left-[50%] top-1/2 -translate-x-1/2 -translate-y-1/2 z-5 h-3/4 w-6 py-3 px-0.5 rounded-md border border-dashed border-neutral-300 bg-neutral-50/70 flex items-center justify-center font-mono text-[7px] sm:text-[8px] font-bold text-neutral-400 tracking-wider pointer-events-none [writing-mode:vertical-rl] rotate-180">
                <span>[ CENTRAL PROCTOR WALKWAY • 2.4M AISLE ]</span>
              </div>
            </>
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
            <div className="p-3 bg-white rounded-xl border border-neutral-200 font-mono text-3xs shadow-xs space-y-2">
              <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-neutral-100">
                <div className="flex items-center gap-2">
                  <span className="font-heading font-bold text-xs text-neutral-900 flex items-center gap-1.5">
                    <Camera size={13} className="text-sky-600" />
                    <span>CCTV Field-of-View Coverage Range</span>
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold text-[8px]">
                    {coverageData.coveredCount}/{seats.length} Students Monitored ({coverageData.percent}% FOV)
                  </span>
                </div>

                <div className="flex items-center gap-1.5 text-[8px]">
                  {customCamPos && (
                    <>
                      <span className="text-neutral-400 font-bold uppercase">Mount:</span>
                      <span className="px-1.5 py-0.5 rounded bg-neutral-100 text-neutral-800 font-bold">
                        Aisle Track ({customCamPos.x}%)
                      </span>
                      <span className="text-neutral-400 font-bold uppercase ml-1">Direction:</span>
                      <span className="px-1.5 py-0.5 rounded bg-sky-50 text-sky-800 border border-sky-200 font-bold">
                        {camAimAngle === 90 ? '▶ East (Right)' : '◀ West (Left)'}
                      </span>
                    </>
                  )}
                  {activeCameraFilter && (
                    <button
                      onClick={() => setActiveCameraFilter(null)}
                      className="text-sky-600 font-bold underline cursor-pointer ml-1"
                    >
                      Reset filter
                    </button>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[8px]">
                <div className="p-2 rounded-lg bg-emerald-50/70 border border-emerald-200/80">
                  <div className="text-emerald-700 font-bold uppercase tracking-wider text-[7px] mb-0.5">
                    Covered Students Range
                  </div>
                  <div className="font-heading font-black text-sm text-emerald-950">
                    {coverageData.candidateRange}
                  </div>
                  <div className="text-emerald-800/80 text-[7.5px] mt-0.5">
                    {coverageData.coveredCount} active candidates inside optical line-of-sight
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-sky-50/70 border border-sky-200/80">
                  <div className="text-sky-700 font-bold uppercase tracking-wider text-[7px] mb-0.5">
                    Covered Workstation Terminals
                  </div>
                  <div className="font-mono font-bold text-xs text-sky-950 flex flex-col gap-0.5">
                    <span>Row A: {coverageData.rowARange}</span>
                    <span>Row B: {coverageData.rowBRange}</span>
                  </div>
                  <div className="text-sky-800/80 text-[7.5px] mt-0.5">
                    Dual-row sweeping vision cone
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-neutral-50 border border-neutral-200">
                  <div className="text-neutral-500 font-bold uppercase tracking-wider text-[7px] mb-0.5">
                    Blind Spot (Behind Camera Casing)
                  </div>
                  <div className="font-mono font-bold text-xs text-neutral-800">
                    {coverageData.blindIds.length > 0 ? coverageData.blindIds.join(', ') : 'None (100% Coverage)'}
                  </div>
                  <div className="text-neutral-500 text-[7.5px] mt-0.5">
                    {coverageData.blindIds.length > 0
                      ? `${coverageData.blindIds.length} desks behind lens aperture (drag camera or switch aim)`
                      : 'Zero blind spots'}
                  </div>
                </div>
              </div>
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
