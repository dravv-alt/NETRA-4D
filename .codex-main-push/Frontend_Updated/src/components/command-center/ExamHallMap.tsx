import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SeatNodeData, CameraFeedInfo } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ArrowRight, Laptop, Shield, ArrowUp, ArrowDown, User, Sliders, Compass, Eye, Sparkles } from 'lucide-react';
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
  const [layoutColumns, setLayoutColumns] = useState<5 | 10>(5); // 5 cols (spacious 2-row) or 10 cols

  const rowA = seats.filter((s) => s.gridPosition.row === 'A');
  const rowB = seats.filter((s) => s.gridPosition.row === 'B');

  const getSeatStyle = (state: SeatNodeData['state'], isSelected: boolean) => {
    let base = 'relative p-3.5 rounded-2xl transition-all duration-200 cursor-pointer flex flex-col justify-between select-none font-sans min-h-[92px] ';

    if (isSelected) {
      base += 'bg-[#181716] text-[#FAF7F3] ring-4 ring-neutral-900/20 shadow-lg border-neutral-900 scale-[1.03] z-10 ';
      return base;
    }

    switch (state) {
      case 'escalated':
        return base + 'bg-rose-50/90 border border-rose-200 text-rose-950 hover:border-rose-400 hover:shadow-md';
      case 'review':
        return base + 'bg-amber-50/90 border border-amber-200 text-amber-950 hover:border-amber-400 hover:shadow-md';
      case 'observed':
        return base + 'bg-violet-50/90 border border-violet-200 text-violet-950 hover:border-violet-400 hover:shadow-md';
      case 'normal':
      default:
        return base + 'bg-white border border-neutral-200 text-neutral-900 hover:border-emerald-300 hover:shadow-md hover:bg-emerald-50/20';
    }
  };

  return (
    <div className="p-6 sm:p-7 rounded-3xl bg-neutral-50/70 border border-neutral-200/80 space-y-6 font-sans">
      {/* 1. Header with Controls */}
      <div className="flex items-center justify-between pb-3 border-b border-neutral-200 flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <IllustratedIcon icon={Compass} variant="sky" size="sm" />
          <div>
            <div className="font-heading font-extrabold text-base text-neutral-950 flex items-center gap-2">
              <span>Computer Lab Terminal Deck</span>
              <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded-full bg-white border border-neutral-200 text-neutral-800">
                20 Active
              </span>
            </div>
            <p className="text-2xs text-neutral-500 mt-0.5">
              Row A (North Wall) & Row B (South Wall) • Central Proctor Walkway
            </p>
          </div>
        </div>

        {/* Action Button & Layout View Toggle */}
        <div className="flex items-center gap-3">
          {/* 5-col / 10-col Toggle */}
          <div className="flex items-center bg-white p-1 rounded-xl border border-neutral-200 text-3xs font-mono">
            <button
              onClick={() => {
                playHapticSound('click');
                setLayoutColumns(5);
              }}
              className={`px-2.5 py-1 rounded-lg transition-all font-bold cursor-pointer ${
                layoutColumns === 5 ? 'bg-neutral-900 text-white' : 'text-neutral-500 hover:text-neutral-900'
              }`}
            >
              5x2 Grid
            </button>
            <button
              onClick={() => {
                playHapticSound('click');
                setLayoutColumns(10);
              }}
              className={`px-2.5 py-1 rounded-lg transition-all font-bold cursor-pointer ${
                layoutColumns === 10 ? 'bg-neutral-900 text-white' : 'text-neutral-500 hover:text-neutral-900'
              }`}
            >
              10x1 Grid
            </button>
          </div>

          <Button
            variant="default"
            size="xs"
            onClick={() => {
              playHapticSound('snap');
              setIsStudioOpen(true);
            }}
            className="gap-1.5 rounded-xl font-mono text-3xs shadow-xs"
          >
            <Sliders size={12} />
            <span>Design Studio</span>
          </Button>
        </div>
      </div>

      {/* ================= NORTH DESK: ROW A (FACING NORTH / LAPTOPS AT TOP) ================= */}
      <div className="space-y-2.5">
        <div className="w-full py-2 bg-sky-50 rounded-xl text-center text-3xs font-mono font-bold text-sky-900 uppercase tracking-wider flex items-center justify-center gap-2 border border-sky-200/70">
          <Laptop size={13} className="text-sky-700" />
          <span>NORTH WALL DESK — ROW A (STUDENTS FACING NORTH LAPTOPS)</span>
          <Laptop size={13} className="text-sky-700" />
        </div>

        {/* Row A Terminals Grid */}
        <div className={`grid gap-2.5 ${layoutColumns === 5 ? 'grid-cols-5' : 'grid-cols-5 sm:grid-cols-10'}`}>
          {rowA.map((seat) => {
            const isSelected = selectedSeatId === seat.id;

            return (
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
                whileHover={{ scale: 1.04, y: -2, transition: { type: 'spring', damping: 20, stiffness: 420 } }}
                whileTap={{ scale: 0.97 }}
                className={getSeatStyle(seat.state, isSelected)}
              >
                {/* Top Laptop Indicator */}
                <div className="flex items-center justify-between text-3xs pb-1 border-b border-black/[0.06] font-mono">
                  <span className="flex items-center gap-1 font-bold opacity-75">
                    <Laptop size={11} />
                    <span>LPT</span>
                  </span>
                  <span title="Facing Laptop (North)" className="opacity-75">
                    <ArrowUp size={11} />
                  </span>
                </div>

                <div className="my-1 flex items-center justify-between">
                  <div className={`font-mono font-extrabold text-sm sm:text-base ${isSelected ? 'text-white' : 'text-neutral-950'}`}>
                    {seat.id}
                  </div>
                  <div className={`text-3xs font-mono font-semibold px-1.5 py-0.5 rounded ${
                    isSelected ? 'bg-white/20 text-white' : 'bg-neutral-100 text-neutral-700'
                  }`}>
                    #{seat.candidateId}
                  </div>
                </div>

                {/* Bottom Back of Student Indicator */}
                <div className="pt-1 border-t border-black/[0.06] flex items-center justify-between text-3xs font-mono opacity-80">
                  <span className="flex items-center gap-0.5">
                    <User size={10} /> Back
                  </span>
                  {seat.telemetry.objectEvidence && (
                    <span className="font-bold px-1 py-0.2 rounded bg-rose-500 text-white text-3xs">
                      OBJ
                    </span>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* ================= CENTRAL WALKWAY / AISLE ================= */}
      <div className="w-full py-2.5 bg-white rounded-xl border border-dashed border-neutral-300 flex items-center justify-center gap-2 shadow-2xs font-mono text-3xs text-neutral-600">
        <Shield size={13} className="text-neutral-500" />
        <span className="font-bold">CENTRAL PROCTOR WALKWAY (STUDENTS SIT BACK-TO-BACK)</span>
        <Shield size={13} className="text-neutral-500" />
      </div>

      {/* ================= SOUTH DESK: ROW B (FACING SOUTH / LAPTOPS AT BOTTOM) ================= */}
      <div className="space-y-2.5">
        {/* Row B Terminals Grid */}
        <div className={`grid gap-2.5 ${layoutColumns === 5 ? 'grid-cols-5' : 'grid-cols-5 sm:grid-cols-10'}`}>
          {rowB.map((seat) => {
            const isSelected = selectedSeatId === seat.id;

            return (
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
                whileHover={{ scale: 1.04, y: 2, transition: { type: 'spring', damping: 20, stiffness: 420 } }}
                whileTap={{ scale: 0.97 }}
                className={getSeatStyle(seat.state, isSelected)}
              >
                {/* Top Back of Student Indicator */}
                <div className="flex items-center justify-between text-3xs pb-1 border-b border-black/[0.06] font-mono opacity-80">
                  <span className="flex items-center gap-0.5">
                    <User size={10} /> Back
                  </span>
                  <span title="Facing Laptop (South)">
                    <ArrowDown size={11} />
                  </span>
                </div>

                <div className="my-1 flex items-center justify-between">
                  <div className={`font-mono font-extrabold text-sm sm:text-base ${isSelected ? 'text-white' : 'text-neutral-950'}`}>
                    {seat.id}
                  </div>
                  <div className={`text-3xs font-mono font-semibold px-1.5 py-0.5 rounded ${
                    isSelected ? 'bg-white/20 text-white' : 'bg-neutral-100 text-neutral-700'
                  }`}>
                    #{seat.candidateId}
                  </div>
                </div>

                {/* Bottom Laptop Indicator */}
                <div className="pt-1 border-t border-black/[0.06] flex items-center justify-between text-3xs font-mono">
                  <span className="flex items-center gap-1 font-bold opacity-75">
                    <Laptop size={11} />
                    <span>LPT</span>
                  </span>
                  {seat.telemetry.objectEvidence && (
                    <span className="font-bold px-1 py-0.2 rounded bg-rose-500 text-white text-3xs">
                      OBJ
                    </span>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>

        <div className="w-full py-2 bg-sky-50 rounded-xl text-center text-3xs font-mono font-bold text-sky-900 uppercase tracking-wider flex items-center justify-center gap-2 border border-sky-200/70">
          <Laptop size={13} className="text-sky-700" />
          <span>SOUTH WALL DESK — ROW B (STUDENTS FACING SOUTH LAPTOPS)</span>
          <Laptop size={13} className="text-sky-700" />
        </div>
      </div>

      {/* Real-time Interactive Hover Dossier Strip */}
      <div className="min-h-[58px]">
        <AnimatePresence mode="wait">
          {hoveredSeat ? (
            <motion.div
              key={hoveredSeat.id}
              initial={{ opacity: 0, y: 3 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="bg-[#181716] text-white rounded-2xl p-3.5 text-xs flex items-center justify-between border border-neutral-800 shadow-md flex-wrap gap-3"
            >
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-white text-neutral-950 font-mono font-bold text-sm flex items-center justify-center shadow-sm">
                  {hoveredSeat.id}
                </div>
                <div>
                  <div className="font-heading font-bold text-xs text-white flex items-center gap-2">
                    <span>Candidate #{hoveredSeat.candidateId}</span>
                    <StatusBadge status={hoveredSeat.state} size="sm" />
                  </div>
                  <div className="text-3xs text-neutral-400 mt-0.5 font-mono">
                    {hoveredSeat.telemetry.lastEventSummary}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 text-3xs font-mono">
                <div className="bg-white/10 px-2.5 py-1 rounded-lg border border-white/10">
                  <span className="text-neutral-400">Continuity: </span>
                  <strong className="text-emerald-400 font-bold">{hoveredSeat.telemetry.trackContinuity}</strong>
                </div>
                <div className="bg-white/10 px-2.5 py-1 rounded-lg border border-white/10">
                  <span className="text-neutral-400">Gaze: </span>
                  <strong className="text-white font-bold">{hoveredSeat.telemetry.headEvidence}</strong>
                </div>
                <Button
                  variant="default"
                  size="xs"
                  onClick={() => {
                    playHapticSound('snap');
                    onSelectSeat(hoveredSeat.id);
                  }}
                  className="bg-white text-neutral-950 hover:bg-neutral-100 rounded-lg px-2.5 py-1 font-bold shadow-xs text-3xs"
                >
                  <span>Inspect</span>
                  <ArrowRight size={11} />
                </Button>
              </div>
            </motion.div>
          ) : (
            <div className="p-3 text-center text-3xs text-neutral-500 bg-white rounded-xl border border-neutral-200 font-mono">
              💡 Hover over any of the 20 lab terminals to inspect posture, gaze angle, and laptop interaction.
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
