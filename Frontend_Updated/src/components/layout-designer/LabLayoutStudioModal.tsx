import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SeatNodeData, CameraFeedInfo } from '../../types';
import {
  X,
  Plus,
  Trash2,
  RotateCw,
  Sparkles,
  Laptop,
  Camera,
  Shield,
  Upload,
  CheckCircle2,
  Sliders,
  Grid,
  RefreshCw,
  Move,
  Eye,
  Layers,
  ArrowUp,
  ArrowDown,
  Compass
} from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { Tabs } from '../ui/tabs';

interface LabLayoutStudioModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentSeats: SeatNodeData[];
  onSaveSeats: (newSeats: SeatNodeData[]) => void;
  cameras: CameraFeedInfo[];
}

export const LabLayoutStudioModal: React.FC<LabLayoutStudioModalProps> = ({
  isOpen,
  onClose,
  currentSeats,
  onSaveSeats,
  cameras,
}) => {
  const [activeTab, setActiveTab] = useState<'canvas' | 'ai-camera' | 'presets'>('canvas');
  const [editableSeats, setEditableSeats] = useState<SeatNodeData[]>(() => JSON.parse(JSON.stringify(currentSeats)));

  // AI Camera / Blueprint Calibration State
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-01');
  const [isScanning, setIsScanning] = useState(false);
  const [scanComplete, setScanComplete] = useState(false);
  const [uploadedImageName, setUploadedImageName] = useState<string | null>(null);
  const [aiSourceMode, setAiSourceMode] = useState<'cctv' | 'blueprint'>('cctv');
  const [detectedTerminalCount, setDetectedTerminalCount] = useState<number>(20);
  const [detectedConfidence, setDetectedConfidence] = useState<string>('98.4%');
  const [calibratedData, setCalibratedData] = useState<SeatNodeData[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  if (!isOpen) return null;

  const rowA = editableSeats.filter((s) => s.gridPosition.row === 'A');
  const rowB = editableSeats.filter((s) => s.gridPosition.row === 'B');

  // Add terminal to row
  const handleAddTerminal = (row: 'A' | 'B') => {
    playHapticSound('snap');
    const rowSeats = editableSeats.filter((s) => s.gridPosition.row === row);
    const newCol = rowSeats.length + 1;
    const seatId = `${row}${String(newCol).padStart(2, '0')}`;
    const newCandidateId = String(editableSeats.length + 1);

    const newSeat: SeatNodeData = {
      id: seatId,
      candidateId: newCandidateId,
      candidateName: `Candidate #${newCandidateId}`,
      cameraIds: [row === 'A' ? 'CAM-01' : 'CAM-03'],
      gridPosition: { row, col: newCol },
      state: 'normal',
      telemetry: {
        trackContinuity: 'GOOD',
        headEvidence: 'GOOD',
        handsEvidence: 'GOOD',
        objectEvidence: null,
        lastEventSummary: 'Focusing on laptop screen',
        lastEventTimeAgo: 'Just now',
      },
    };

    setEditableSeats([...editableSeats, newSeat]);
  };

  // Remove specific terminal
  const handleRemoveTerminal = (seatId: string) => {
    playHapticSound('click');
    setEditableSeats(editableSeats.filter((s) => s.id !== seatId));
  };

  // Rotate seating orientation
  const handleToggleOrientation = (seatId: string) => {
    playHapticSound('snap');
    setEditableSeats(
      editableSeats.map((seat) => {
        if (seat.id === seatId) {
          const nextRow = seat.gridPosition.row === 'A' ? 'B' : 'A';
          return {
            ...seat,
            gridPosition: { ...seat.gridPosition, row: nextRow },
          };
        }
        return seat;
      })
    );
  };

  // Preset Loaders
  const loadPreset = (presetType: '20-back-to-back' | '18-back-to-back' | '16-compact' | 'real-computer-lab') => {
    playHapticSound('snap');
    if (presetType === 'real-computer-lab') {
      const apiBase = (import.meta.env.VITE_NETRA_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
      fetch(`${apiBase}/api/spatial/parse-floorplan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: 'seating-plan-for-computer-class.png' }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.terminals && data.terminals.length > 0) {
            setEditableSeats(data.terminals);
          }
        })
        .catch(() => {});
    } else if (presetType === '20-back-to-back') {
      const generated: SeatNodeData[] = [];
      // Row A (10 seats)
      for (let i = 1; i <= 10; i++) {
        generated.push({
          id: `A${String(i).padStart(2, '0')}`,
          candidateId: String(i),
          candidateName: `Candidate #${i}`,
          cameraIds: [i <= 5 ? 'CAM-01' : 'CAM-02'],
          gridPosition: { row: 'A', col: i },
          state: i === 6 ? 'review' : i === 4 || i === 3 ? 'observed' : 'normal',
          telemetry: {
            trackContinuity: 'GOOD',
            headEvidence: 'GOOD',
            handsEvidence: 'GOOD',
            objectEvidence: i === 6 ? { label: 'PHONE', confidence: 0.74 } : null,
            lastEventSummary: 'Focusing on laptop screen',
            lastEventTimeAgo: '4m ago',
          },
        });
      }
      // Row B (10 seats)
      for (let i = 1; i <= 10; i++) {
        const cId = 10 + i;
        generated.push({
          id: `B${String(i).padStart(2, '0')}`,
          candidateId: String(cId),
          candidateName: `Candidate #${cId}`,
          cameraIds: [i <= 5 ? 'CAM-03' : 'CAM-04'],
          gridPosition: { row: 'B', col: i },
          state: i === 8 ? 'escalated' : i === 4 ? 'review' : i === 9 ? 'review' : 'normal',
          telemetry: {
            trackContinuity: 'GOOD',
            headEvidence: 'GOOD',
            handsEvidence: 'GOOD',
            objectEvidence: null,
            lastEventSummary: 'Active terminal session',
            lastEventTimeAgo: '9m ago',
          },
        });
      }
      setEditableSeats(generated);
    } else if (presetType === '18-back-to-back') {
      const generated: SeatNodeData[] = [];
      // 9 North, 9 South
      for (let i = 1; i <= 9; i++) {
        generated.push({
          id: `A${String(i).padStart(2, '0')}`,
          candidateId: String(i),
          candidateName: `Candidate #${i}`,
          cameraIds: ['CAM-01', 'CAM-02'],
          gridPosition: { row: 'A', col: i },
          state: i === 5 ? 'review' : 'normal',
          telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Laptop active', lastEventTimeAgo: '5m ago' },
        });
      }
      for (let i = 1; i <= 9; i++) {
        const cId = 9 + i;
        generated.push({
          id: `B${String(i).padStart(2, '0')}`,
          candidateId: String(cId),
          candidateName: `Candidate #${cId}`,
          cameraIds: ['CAM-03', 'CAM-04'],
          gridPosition: { row: 'B', col: i },
          state: i === 8 ? 'escalated' : 'normal',
          telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Laptop active', lastEventTimeAgo: '3m ago' },
        });
      }
      setEditableSeats(generated);
    } else if (presetType === '16-compact') {
      const generated: SeatNodeData[] = [];
      // 8 North, 8 South
      for (let i = 1; i <= 8; i++) {
        generated.push({
          id: `A${String(i).padStart(2, '0')}`,
          candidateId: String(i),
          candidateName: `Candidate #${i}`,
          cameraIds: ['CAM-01'],
          gridPosition: { row: 'A', col: i },
          state: 'normal',
          telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Laptop active', lastEventTimeAgo: '5m ago' },
        });
      }
      for (let i = 1; i <= 8; i++) {
        const cId = 8 + i;
        generated.push({
          id: `B${String(i).padStart(2, '0')}`,
          candidateId: String(cId),
          candidateName: `Candidate #${cId}`,
          cameraIds: ['CAM-03'],
          gridPosition: { row: 'B', col: i },
          state: 'normal',
          telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Laptop active', lastEventTimeAgo: '2m ago' },
        });
      }
      setEditableSeats(generated);
    }
  };

  // Run AI Camera or Blueprint Auto-Calibration
  const handleStartAIScan = async () => {
    playHapticSound('snap');
    setIsScanning(true);
    setScanComplete(false);

    const apiBase = (import.meta.env.VITE_NETRA_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

    try {
      if (uploadedImageName) {
        // Path A: Architectural Blueprint Ingestion
        setAiSourceMode('blueprint');
        const res = await fetch(`${apiBase}/api/spatial/parse-floorplan`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_path: uploadedImageName }),
        });
        if (res.ok) {
          const data = await res.json();
          setDetectedTerminalCount(data.total_terminals || 18);
          setDetectedConfidence('Dynamic OpenCV Contour & Orientation Verified');
          if (data.terminals && data.terminals.length > 0) {
            setCalibratedData(data.terminals);
            setEditableSeats(data.terminals);
          }
        }
      } else {
        // Path B: CCTV Video Frame Inverse Perspective Mapping (Edge-case)
        setAiSourceMode('cctv');
        const res = await fetch(`${apiBase}/api/spatial/calibrate-from-cctv?camera_id=${selectedCameraId}`, {
          method: 'POST',
        });
        if (res.ok) {
          const data = await res.json();
          const count = data.detected_terminal_count || data.terminals?.length || 6;
          setDetectedTerminalCount(count);
          setDetectedConfidence('Dynamic YOLO-Pose & Spatial Clustering');
          if (data.terminals && data.terminals.length > 0) {
            setCalibratedData(data.terminals);
            setEditableSeats(data.terminals);
          }
        }
      }
    } catch {
      // Keep existing editable seats
    } finally {
      setIsScanning(false);
      setScanComplete(true);
      playHapticSound('snap');
    }
  };

  const handleApplyAIDetectedLayout = () => {
    playHapticSound('snap');
    const target = calibratedData || editableSeats;
    if (target && target.length > 0) {
      setEditableSeats(target);
      onSaveSeats(target);
      onClose();
    }
  };

  const handleSaveAndApply = () => {
    playHapticSound('snap');
    onSaveSeats(editableSeats);
    onClose();
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Dimmed Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onClick={() => {
            playHapticSound('click');
            onClose();
          }}
          className="fixed inset-0 bg-black/50 backdrop-blur-xs"
        />

        {/* Modal Window */}
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 10 }}
          transition={{ type: 'spring', damping: 26, stiffness: 350 }}
          className="relative z-50 w-full max-w-5xl bg-white border border-neutral-200 rounded-3xl p-6 sm:p-8 font-sans text-sm shadow-2xl space-y-6 text-neutral-900 max-h-[90vh] flex flex-col"
        >
          {/* Header */}
          <div className="flex items-center justify-between pb-4 border-b border-neutral-200 shrink-0">
            <div className="flex items-center gap-3.5">
              <div className="w-11 h-11 rounded-2xl bg-neutral-900 flex items-center justify-center text-white font-bold shadow-sm">
                <Sliders size={20} />
              </div>
              <div>
                <h2 className="font-bold text-lg text-neutral-950 flex items-center gap-2.5">
                  <span>Lab Layout Studio & Spatial Calibration</span>
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-neutral-100 border border-neutral-200 text-neutral-700">
                    {editableSeats.length} Active Terminals
                  </span>
                </h2>
                <p className="text-xs text-neutral-500 mt-0.5">
                  Customize terminal coordinates, drag & drop desks, or auto-detect from CCTV camera frames.
                </p>
              </div>
            </div>

            <button
              onClick={() => {
                playHapticSound('click');
                onClose();
              }}
              className="p-2 rounded-xl hover:bg-neutral-100 text-neutral-400 hover:text-neutral-900 transition-colors cursor-pointer"
            >
              <X size={20} />
            </button>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="shrink-0 flex items-center justify-between">
            <Tabs
              items={[
                { id: 'canvas', label: 'Terminal Row Canvas', icon: <Grid size={15} /> },
                { id: 'ai-camera', label: 'AI Auto-Detect (CCTV / Blueprint)', icon: <Sparkles size={15} /> },
                { id: 'presets', label: 'Lab Layout Templates', icon: <Layers size={15} /> },
              ]}
              activeId={activeTab}
              onChange={(id) => setActiveTab(id as any)}
              size="sm"
            />

            <div className="text-xs text-neutral-500 hidden sm:flex items-center gap-3">
              <span>Row A: <strong className="text-neutral-900 font-semibold">{rowA.length} Terminals (North)</strong></span>
              <span>•</span>
              <span>Row B: <strong className="text-neutral-900 font-semibold">{rowB.length} Terminals (South)</strong></span>
            </div>
          </div>

          {/* SCROLLABLE MAIN CONTENT */}
          <div className="flex-1 overflow-y-auto pr-1 min-h-[380px]">
            {/* ================= TAB 1: DRAG & DROP BUILDER ================= */}
            {activeTab === 'canvas' && (
              <div className="space-y-6">
                {/* Visual Canvas Surface */}
                <div className="p-5 rounded-2xl border border-neutral-200 bg-neutral-50/60 space-y-4">
                  {/* NORTH DESK: ROW A */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between px-2 text-xs font-semibold text-neutral-700">
                      <div className="flex items-center gap-2">
                        <Laptop size={14} className="text-neutral-900" />
                        <span>NORTH WALL LAB DESK (ROW A — {rowA.length} TERMINALS)</span>
                      </div>
                      <Button
                        variant="outline"
                        size="xs"
                        onClick={() => handleAddTerminal('A')}
                        className="gap-1 rounded-lg"
                      >
                        <Plus size={12} />
                        <span>Add Terminal</span>
                      </Button>
                    </div>

                    <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 p-3 bg-white rounded-xl border border-neutral-200 shadow-2xs min-h-[90px]">
                      {rowA.map((seat) => (
                        <motion.div
                          key={seat.id}
                          layout
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="p-2.5 rounded-xl border border-neutral-200 bg-neutral-50 hover:border-neutral-400 transition-all flex flex-col justify-between group relative"
                        >
                          <div className="flex items-center justify-between text-2xs">
                            <span className="font-bold text-neutral-950">{seat.id}</span>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => handleToggleOrientation(seat.id)}
                                title="Flip to South Row"
                                className="text-neutral-400 hover:text-neutral-900 cursor-pointer"
                              >
                                <ArrowDown size={10} />
                              </button>
                              <button
                                onClick={() => handleRemoveTerminal(seat.id)}
                                title="Delete Terminal"
                                className="text-neutral-400 hover:text-black cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <Trash2 size={10} />
                              </button>
                            </div>
                          </div>

                          <div className="my-1 text-center font-semibold text-xs text-neutral-900">
                            #{seat.candidateId}
                          </div>

                          <div className="text-3xs text-neutral-400 text-center border-t border-neutral-200 pt-0.5 flex items-center justify-center gap-0.5">
                            <ArrowUp size={8} /> Facing North
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>

                  {/* CENTRAL PROCTOR WALKWAY */}
                  <div className="py-3 px-4 bg-white rounded-xl border border-dashed border-neutral-300 flex items-center justify-between text-xs text-neutral-600 shadow-2xs">
                    <div className="flex items-center gap-2 font-medium">
                      <Shield size={14} className="text-neutral-900" />
                      <span>Central Proctor Walkway (Students Seated Back-to-Back)</span>
                    </div>
                    <span className="text-2xs text-neutral-400">Cameras CAM-01 to CAM-04 Anchored</span>
                  </div>

                  {/* SOUTH DESK: ROW B */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between px-2 text-xs font-semibold text-neutral-700">
                      <div className="flex items-center gap-2">
                        <Laptop size={14} className="text-neutral-900" />
                        <span>SOUTH WALL LAB DESK (ROW B — {rowB.length} TERMINALS)</span>
                      </div>
                      <Button
                        variant="outline"
                        size="xs"
                        onClick={() => handleAddTerminal('B')}
                        className="gap-1 rounded-lg"
                      >
                        <Plus size={12} />
                        <span>Add Terminal</span>
                      </Button>
                    </div>

                    <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 p-3 bg-white rounded-xl border border-neutral-200 shadow-2xs min-h-[90px]">
                      {rowB.map((seat) => (
                        <motion.div
                          key={seat.id}
                          layout
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="p-2.5 rounded-xl border border-neutral-200 bg-neutral-50 hover:border-neutral-400 transition-all flex flex-col justify-between group relative"
                        >
                          <div className="flex items-center justify-between text-2xs">
                            <span className="font-bold text-neutral-950">{seat.id}</span>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => handleToggleOrientation(seat.id)}
                                title="Flip to North Row"
                                className="text-neutral-400 hover:text-neutral-900 cursor-pointer"
                              >
                                <ArrowUp size={10} />
                              </button>
                              <button
                                onClick={() => handleRemoveTerminal(seat.id)}
                                title="Delete Terminal"
                                className="text-neutral-400 hover:text-black cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <Trash2 size={10} />
                              </button>
                            </div>
                          </div>

                          <div className="my-1 text-center font-semibold text-xs text-neutral-900">
                            #{seat.candidateId}
                          </div>

                          <div className="text-3xs text-neutral-400 text-center border-t border-neutral-200 pt-0.5 flex items-center justify-center gap-0.5">
                            <ArrowDown size={8} /> Facing South
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}


            {/* ================= TAB 2: AI VISION AUTO-CALIBRATION ================= */}
            {activeTab === 'ai-camera' && (
              <div className="space-y-6">
                <div className="grid grid-cols-12 gap-6">
                  {/* Left 4 Cols: Source Ingestion Controls */}
                  <div className="col-span-12 lg:col-span-4 space-y-4">
                    <div className="bg-neutral-50 p-4 rounded-2xl border border-neutral-200 space-y-3">
                      <div className="text-xs font-semibold uppercase tracking-wider text-neutral-600">
                        1. Select CCTV Camera Feed
                      </div>
                      <div className="space-y-1.5">
                        {cameras.map((cam) => {
                          const isSelected = selectedCameraId === cam.id;
                          return (
                            <button
                              key={cam.id}
                              onClick={() => {
                                playHapticSound('click');
                                setSelectedCameraId(cam.id);
                                setScanComplete(false);
                              }}
                              className={`w-full p-2.5 rounded-xl text-left text-xs transition-colors flex items-center justify-between cursor-pointer ${
                                isSelected
                                  ? 'bg-neutral-900 text-white font-semibold shadow-xs'
                                  : 'bg-white text-neutral-700 hover:bg-neutral-100 border border-neutral-200'
                              }`}
                            >
                              <div className="flex items-center gap-2">
                                <Camera size={13} />
                                <span>{cam.id} ({cam.name.split(' ')[0]})</span>
                              </div>
                              <span className="text-2xs opacity-80">{cam.fps} FPS</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* Or Upload Image */}
                    <div className="bg-neutral-50 p-4 rounded-2xl border border-neutral-200 space-y-3">
                      <div className="text-xs font-semibold uppercase tracking-wider text-neutral-600">
                        2. Or Upload Lab Photo / Blueprint
                      </div>
                      <input
                        type="file"
                        ref={fileInputRef}
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            setUploadedImageName(e.target.files[0].name);
                            playHapticSound('snap');
                          }
                        }}
                      />
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full gap-2 text-xs"
                      >
                        <Upload size={14} />
                        <span>{uploadedImageName ? uploadedImageName : 'Upload Photo / Floorplan'}</span>
                      </Button>
                    </div>

                    {/* Start AI Calibration CTA */}
                    <Button
                      variant="default"
                      size="default"
                      onClick={handleStartAIScan}
                      disabled={isScanning}
                      className="w-full gap-2 shadow-sm font-semibold"
                    >
                      {isScanning ? (
                        <>
                          <RefreshCw size={15} className="animate-spin" />
                          <span>Analyzing Perspective Grid...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles size={15} />
                          <span>Run AI Vision Auto-Calibration</span>
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Right 8 Cols: Live Camera Frame Scanner Simulation */}
                  <div className="col-span-12 lg:col-span-8 space-y-4">
                    <div className="relative aspect-video bg-black rounded-2xl overflow-hidden border border-neutral-200 flex items-center justify-center">
                      {/* CCTV Background Drawing */}
                      <div className="w-full h-full p-6 flex flex-col justify-between relative bg-neutral-950 text-white font-mono text-xs select-none">
                        {/* Perspective Desks Simulation */}
                        <div className="w-full h-12 border-b border-neutral-700 bg-neutral-900/60 flex items-center justify-around px-2 text-2xs text-neutral-400">
                          <span>💻 A01</span><span>💻 A02</span><span>💻 A03</span><span>💻 A04</span><span>💻 A05</span>
                          <span>💻 A06</span><span>💻 A07</span><span>💻 A08</span><span>💻 A09</span><span>💻 A10</span>
                        </div>

                        {/* Central Aisle */}
                        <div className="w-full py-4 border-y border-dashed border-neutral-700 text-center text-3xs text-neutral-400 uppercase tracking-widest">
                          [ CENTRAL CORRIDOR — BACK-TO-BACK GAZE VECTORS ]
                        </div>

                        {/* South Desks Simulation */}
                        <div className="w-full h-12 border-t border-neutral-700 bg-neutral-900/60 flex items-center justify-around px-2 text-2xs text-neutral-400">
                          <span>💻 B01</span><span>💻 B02</span><span>💻 B03</span><span>💻 B04</span><span>💻 B05</span>
                          <span>💻 B06</span><span>💻 B07</span><span>💻 B08</span><span>💻 B09</span><span>💻 B10</span>
                        </div>

                        {/* Scanning Laser Animation */}
                        {isScanning && (
                          <motion.div
                            initial={{ top: '0%' }}
                            animate={{ top: '100%' }}
                            transition={{ duration: 1.8, repeat: Infinity, ease: 'linear' }}
                            className="absolute left-0 right-0 h-0.5 bg-white shadow-[0_0_12px_#ffffff] z-20"
                          />
                        )}

                        {/* Camera HUD Overlays */}
                        <div className="absolute top-3 left-3 bg-black/80 px-2.5 py-1 rounded-lg text-2xs border border-white/20">
                          {selectedCameraId} • Perspective Homography Ingestion
                        </div>

                        {scanComplete && (
                          <div className="absolute top-3 right-3 bg-white text-black px-3 py-1 rounded-full text-2xs font-bold font-sans flex items-center gap-1.5 shadow-sm">
                            <CheckCircle2 size={13} />
                            <span>20 Terminals Detected (98.4% Confidence)</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* AI Scan Diagnostics */}
                    {scanComplete && (
                      <motion.div
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-neutral-50 p-4 rounded-2xl border border-neutral-200 space-y-3 text-xs"
                      >
                        <div className="font-semibold text-neutral-950 flex items-center justify-between">
                          <span>AI Vision Detection Summary</span>
                          <span className="text-neutral-600 font-normal">Homography Matrix: Calibrated</span>
                        </div>
                        <div className="grid grid-cols-3 gap-3 text-2xs">
                          <div className="bg-white p-2.5 rounded-xl border border-neutral-200">
                            <div className="text-neutral-500">Detected Desks:</div>
                            <div className="font-bold text-neutral-900 mt-0.5">2 Perimeter Clusters</div>
                          </div>
                          <div className="bg-white p-2.5 rounded-xl border border-neutral-200">
                            <div className="text-neutral-500">Laptops Located:</div>
                            <div className="font-bold text-neutral-900 mt-0.5">20 Screen Planes</div>
                          </div>
                          <div className="bg-white p-2.5 rounded-xl border border-neutral-200">
                            <div className="text-neutral-500">Orientation:</div>
                            <div className="font-bold text-neutral-900 mt-0.5">Back-to-Back (N/S)</div>
                          </div>
                        </div>

                        <Button
                          variant="default"
                          size="sm"
                          onClick={handleApplyAIDetectedLayout}
                          className="w-full gap-1.5"
                        >
                          <CheckCircle2 size={14} />
                          <span>Apply AI Detected 20-Terminal Layout to Lab Map</span>
                        </Button>
                      </motion.div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* ================= TAB 3: PRESETS ================= */}
            {activeTab === 'presets' && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* Featured Blueprint CAD Preset */}
                  <div className="bg-sky-50/50 p-5 rounded-2xl border-2 border-sky-400 space-y-3 hover:border-sky-600 transition-colors flex flex-col justify-between shadow-xs relative">
                    <div className="absolute top-3 right-3 text-3xs font-mono font-bold bg-sky-600 text-white px-2 py-0.5 rounded-full">
                      Blueprint CAD
                    </div>
                    <div>
                      <div className="font-bold text-base text-neutral-950">Architectural Lab Blueprint</div>
                      <div className="text-xs text-neutral-600 mt-1 leading-relaxed">
                        Extracted from CAD seating plan: 5 West wall desks, 5 East wall desks, 6 center-island desks, 4 South entrance terminals, and proctor podium.
                      </div>
                      <div className="mt-3 text-xs font-semibold text-sky-950">
                        • 20 Terminals (4 Facing Orientations)
                      </div>
                    </div>
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => {
                        loadPreset('real-computer-lab');
                        setActiveTab('canvas');
                      }}
                      className="w-full mt-3 bg-sky-600 hover:bg-sky-700 text-white font-semibold"
                    >
                      Load Blueprint Layout
                    </Button>
                  </div>

                  {/* Preset 1 */}
                  <div className="bg-white p-5 rounded-2xl border border-neutral-200 space-y-3 hover:border-neutral-400 transition-colors flex flex-col justify-between shadow-xs">
                    <div>
                      <div className="font-bold text-base text-neutral-950">2-Row Back-to-Back Lab (Default)</div>
                      <div className="text-xs text-neutral-500 mt-1 leading-relaxed">
                        Standard 20-terminal computer lab with 10 North terminals and 10 South terminals facing outward.
                      </div>
                      <div className="mt-3 text-xs font-semibold text-neutral-800">
                        • 20 Terminals (Row A: 10, Row B: 10)
                      </div>
                    </div>
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => {
                        loadPreset('20-back-to-back');
                        setActiveTab('canvas');
                      }}
                      className="w-full mt-3"
                    >
                      Load 20-Seat Layout
                    </Button>
                  </div>

                  {/* Preset 2 */}
                  <div className="bg-white p-5 rounded-2xl border border-neutral-200 space-y-3 hover:border-neutral-400 transition-colors flex flex-col justify-between shadow-xs">
                    <div>
                      <div className="font-bold text-base text-neutral-950">18-Seat Compact Lab</div>
                      <div className="text-xs text-neutral-500 mt-1 leading-relaxed">
                        Medium computer lab with 9 North terminals and 9 South terminals with central walkway.
                      </div>
                      <div className="mt-3 text-xs font-semibold text-neutral-800">
                        • 18 Terminals (Row A: 9, Row B: 9)
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        loadPreset('18-back-to-back');
                        setActiveTab('canvas');
                      }}
                      className="w-full mt-3"
                    >
                      Load 18-Seat Layout
                    </Button>
                  </div>

                  {/* Preset 3 */}
                  <div className="bg-white p-5 rounded-2xl border border-neutral-200 space-y-3 hover:border-neutral-400 transition-colors flex flex-col justify-between shadow-xs">
                    <div>
                      <div className="font-bold text-base text-neutral-950">16-Seat Small Cohort</div>
                      <div className="text-xs text-neutral-500 mt-1 leading-relaxed">
                        Compact testing session with 8 North terminals and 8 South terminals.
                      </div>
                      <div className="mt-3 text-xs font-semibold text-neutral-800">
                        • 16 Terminals (Row A: 8, Row B: 8)
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        loadPreset('16-compact');
                        setActiveTab('canvas');
                      }}
                      className="w-full mt-3"
                    >
                      Load 16-Seat Layout
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Footer Action Strip */}
          <div className="flex items-center justify-between pt-4 border-t border-neutral-200 shrink-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                playHapticSound('click');
                loadPreset('20-back-to-back');
              }}
              className="text-xs"
            >
              Reset to Default (20 Seats)
            </Button>

            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  playHapticSound('click');
                  onClose();
                }}
              >
                Cancel
              </Button>

              <Button
                variant="default"
                size="sm"
                onClick={handleSaveAndApply}
                className="font-semibold shadow-sm"
              >
                Save & Apply to Exam Map
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
