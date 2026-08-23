import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Video, Play, Pause, RotateCcw, ShieldCheck, Maximize2, Sparkles, ZoomIn } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';

export const MultiCamForensicMatrix: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [timelineSec, setTimelineSec] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);

  const cameras = [
    { id: 'CAM-01', name: 'North Desk Ceiling Angle', fov: 'Covers Row A (Terminals A01-A10)', color: 'border-sky-500/40' },
    { id: 'CAM-02', name: 'North 45° Low Lateral', fov: 'Wrist & Lap Occlusion Inspection', color: 'border-amber-500/40' },
    { id: 'CAM-03', name: 'South Desk Ceiling Angle', fov: 'Covers Row B (Terminals B01-B10)', color: 'border-emerald-500/40' },
    { id: 'CAM-04', name: 'Central Walkway Surveillance', fov: 'Proctor Corridor & Peer Exchanges', color: 'border-purple-500/40' },
  ];

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
              <Video size={18} />
            </div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Synchronized 4-Camera Forensic Matrix
            </h3>
            <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-neutral-900 text-white">
              FRAME-LOCKED RTSP REPLAY
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">
            Simultaneous 4-feed spatial correlation for verified incident triaging at exact timestamp 15:42:18 IST (Event EV-2038).
          </p>
        </div>

        {/* Speed Controls */}
        <div className="flex items-center gap-2">
          {[0.25, 0.5, 1].map((spd) => (
            <button
              key={spd}
              onClick={() => {
                playHapticSound('click');
                setPlaybackSpeed(spd);
              }}
              className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold cursor-pointer transition-all ${
                playbackSpeed === spd
                  ? 'bg-neutral-900 text-white shadow-xs'
                  : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200 hover:text-neutral-950'
              }`}
            >
              {spd}x {spd === 0.25 ? 'Slow-Mo' : ''}
            </button>
          ))}
        </div>
      </div>

      {/* 4-QUADRANT SYNCHRONIZED MATRIX */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {cameras.map((cam, idx) => (
          <div
            key={cam.id}
            className={`relative aspect-16/10 rounded-2xl bg-neutral-950 border ${cam.color} overflow-hidden shadow-sm flex flex-col justify-between p-3.5 select-none`}
          >
            {/* Camera Identifier Header */}
            <div className="flex items-center justify-between z-10">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded bg-black/60 text-white font-mono text-3xs font-bold border border-white/10">
                  {cam.id}
                </span>
                <span className="text-3xs text-neutral-300 font-sans font-semibold">
                  {cam.name}
                </span>
              </div>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </div>

            {/* Target Candidate Cross-hair Bounding Box */}
            <div className="my-auto mx-auto relative w-36 h-28 border-2 border-dashed border-amber-400/80 rounded-xl flex flex-col justify-between p-1.5 bg-amber-500/10">
              <span className="text-3xs font-mono font-bold text-amber-300 bg-black/70 px-1.5 py-0.5 rounded self-start">
                Seat A06 (Track #106)
              </span>
              <span className="text-3xs font-mono text-white bg-black/70 px-1 rounded self-end">
                Conf: 0.74
              </span>
            </div>

            {/* Camera FOV Footer */}
            <div className="z-10 text-3xs font-mono text-neutral-400 bg-black/60 px-2 py-1 rounded backdrop-blur-xs flex items-center justify-between">
              <span className="truncate">{cam.fov}</span>
              <span className="text-neutral-500 shrink-0 ml-2">1080p @ 30fps</span>
            </div>
          </div>
        ))}
      </div>

      {/* Synchronized Replay Timeline Scrubber */}
      <div className="p-4 rounded-2xl bg-neutral-50 border border-neutral-200/80 flex items-center gap-4 flex-wrap">
        <Button
          variant="default"
          size="sm"
          onClick={() => {
            playHapticSound('snap');
            setIsPlaying(!isPlaying);
          }}
          className="rounded-xl gap-2 font-mono text-xs"
        >
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          <span>{isPlaying ? 'PAUSE' : 'PLAY SYNC'}</span>
        </Button>

        <div className="flex-1 flex items-center gap-3">
          <span className="text-xs font-mono text-neutral-500 shrink-0">-15.0s</span>
          <input
            type="range"
            min="-15"
            max="15"
            step="0.1"
            value={timelineSec}
            onChange={(e) => setTimelineSec(Number(e.target.value))}
            className="flex-1 accent-neutral-900 cursor-pointer"
          />
          <span className="text-xs font-mono font-bold text-neutral-900 shrink-0">
            {timelineSec > 0 ? `+${timelineSec.toFixed(1)}s` : `${timelineSec.toFixed(1)}s`}
          </span>
        </div>

        <span className="text-2xs font-mono text-neutral-500 bg-white px-3 py-1.5 rounded-lg border border-neutral-200">
          Timestamp: 15:42:{18 + Math.floor(timelineSec)} IST
        </span>
      </div>
    </div>
  );
};
