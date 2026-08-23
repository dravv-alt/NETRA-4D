import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { Play, Pause, RotateCcw } from 'lucide-react';
import { AIProvider } from '../../types';
import { AILayersState } from './LayerControlWidget';
import { MOCK_TELEMETRY_FRAMES } from '../../data/demoData';

interface CCTVVideoPlayerProps {
  cameraName: string;
  resolution: string;
  fps: number;
  activeProvider: AIProvider;
  layers: AILayersState;
  onTimeUpdate?: (currentTime: number) => void;
  selectedEventTimeSec?: number;
  liveStreamUrl?: string;
  recordedVideoUrl?: string;
  liveStatus?: string;
  onLiveStreamError?: () => void;
}

export const CCTVVideoPlayer: React.FC<CCTVVideoPlayerProps> = ({
  cameraName,
  resolution,
  fps,
  activeProvider,
  layers,
  onTimeUpdate,
  selectedEventTimeSec,
  liveStreamUrl,
  recordedVideoUrl,
  liveStatus,
  onLiveStreamError,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentTimeSec, setCurrentTimeSec] = useState(selectedEventTimeSec || 14.8);
  const totalDurationSec = 120; // 2 minutes sample loop

  useEffect(() => {
    if (selectedEventTimeSec !== undefined) {
      setCurrentTimeSec(selectedEventTimeSec);
    }
  }, [selectedEventTimeSec]);

  // Video Playback Simulation Loop
  useEffect(() => {
    let interval: any = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentTimeSec((prev) => {
          const next = prev >= totalDurationSec ? 0 : prev + 0.1;
          if (onTimeUpdate) onTimeUpdate(next);
          return next;
        });
      }, 100);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, onTimeUpdate]);

  // Canvas Overlay 2D Rendering Engine (Monochrome Minimalist)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Fixed internal resolution
    canvas.width = 1280;
    canvas.height = 720;

    // 1. Clean Monochrome CCTV Dark Studio Environment
    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Subtle Perspective Desk Lines
    ctx.strokeStyle = '#1F1F1F';
    ctx.lineWidth = 1.5;
    
    ctx.beginPath();
    ctx.moveTo(100, 680); ctx.lineTo(450, 200);
    ctx.moveTo(1180, 680); ctx.lineTo(830, 200);
    ctx.moveTo(0, 520); ctx.lineTo(1280, 520);
    ctx.moveTo(0, 340); ctx.lineTo(1280, 340);
    ctx.stroke();

    // Desk surfaces
    ctx.fillStyle = '#141414';
    ctx.fillRect(320, 360, 340, 260); // Seat A12 Desk
    ctx.fillRect(700, 360, 340, 260); // Seat A13 Desk
    ctx.strokeStyle = '#262626';
    ctx.strokeRect(320, 360, 340, 260);
    ctx.strokeRect(700, 360, 340, 260);

    // CBT Computer Monitors
    ctx.fillStyle = '#1F1F1F';
    ctx.fillRect(430, 290, 120, 80);
    ctx.fillRect(810, 290, 120, 80);
    ctx.fillStyle = '#262626';
    ctx.fillRect(475, 370, 30, 20);
    ctx.fillRect(855, 370, 30, 20);

    // Monitor screen
    ctx.fillStyle = '#2A2A2A';
    ctx.fillRect(435, 295, 110, 70);
    ctx.fillRect(815, 295, 110, 70);

    // Candidate Silhouettes
    // Candidate A12
    ctx.fillStyle = '#1E1E1E';
    ctx.beginPath();
    ctx.arc(490, 240, 45, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(420, 280, 140, 180);

    // Candidate A13
    ctx.fillStyle = '#1A1A1A';
    ctx.beginPath();
    ctx.arc(870, 240, 45, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(800, 280, 140, 180);

    // 2. RENDER ACTIVE AI OVERLAYS IN MONOCHROME
    const mock = MOCK_TELEMETRY_FRAMES[0];

    const boxColor = '#FFFFFF';
    const skeletonColor = '#E5E5E5';
    const keypointColor = '#FFFFFF';

    mock.persons.forEach((person) => {
      const isTarget = person.seatId === 'A12';

      // A. SEAT MAPPING ANCHORS
      if (layers.seatMapping) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(person.box.x - 20, person.box.y - 10, person.box.w + 40, person.box.h + 50);
        ctx.setLineDash([]);

        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(person.box.x - 20, person.box.y - 30, 80, 20);
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 11px -apple-system, "SF Mono", monospace';
        ctx.fillText(`SEAT ${person.seatId}`, person.box.x - 12, person.box.y - 16);
      }

      // B. PERSON BOUNDING BOX
      if (layers.personBoxes) {
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(person.box.x, person.box.y, person.box.w, person.box.h);

        // Header Label Tag
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(person.box.x, person.box.y - 22, person.box.w, 22);

        ctx.fillStyle = '#000000';
        ctx.font = 'bold 11px -apple-system, "SF Mono", monospace';
        const labelText = layers.trackingIds
          ? `${person.seatId} • Track #${person.trackId} | PERSON 0.94`
          : `${person.seatId} | PERSON 0.94`;
        ctx.fillText(labelText, person.box.x + 8, person.box.y - 7);
      }

      // C. POSE SKELETON
      if (layers.poseSkeleton && person.keypoints.length > 0) {
        const kps = person.keypoints;
        const findKp = (name: string) => kps.find((k) => k.name === name);

        const drawBone = (k1Name: string, k2Name: string) => {
          const k1 = findKp(k1Name);
          const k2 = findKp(k2Name);
          if (k1 && k2 && k1.score > 0.5 && k2.score > 0.5) {
            ctx.strokeStyle = skeletonColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(k1.x, k1.y);
            ctx.lineTo(k2.x, k2.y);
            ctx.stroke();
          }
        };

        drawBone('left_shoulder', 'right_shoulder');
        drawBone('left_shoulder', 'left_elbow');
        drawBone('left_elbow', 'left_wrist');
        drawBone('right_shoulder', 'right_elbow');
        drawBone('right_elbow', 'right_wrist');
        drawBone('left_shoulder', 'left_hip');
        drawBone('right_shoulder', 'right_hip');
        drawBone('left_hip', 'right_hip');

        kps.forEach((kp) => {
          if (kp.score > 0.5) {
            ctx.fillStyle = keypointColor;
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      }

      // D. ATTENTION / GAZE RAYS
      if (layers.attentionRays && person.gazeRay) {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(person.gazeRay.originX, person.gazeRay.originY);
        ctx.lineTo(person.gazeRay.targetX, person.gazeRay.targetY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(person.gazeRay.targetX, person.gazeRay.targetY, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = '10px "SF Mono", monospace';
        ctx.fillText(`Yaw: ${person.headYawDeg}°`, person.gazeRay.originX + 15, person.gazeRay.originY - 10);
      }

      // E. 3D VECTOR OVERLAY
      if (layers.threeDVector && isTarget) {
        ctx.strokeStyle = '#D4D4D4';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(480, 320); ctx.lineTo(480, 200);
        ctx.moveTo(480, 320); ctx.lineTo(560, 350);
        ctx.moveTo(480, 320); ctx.lineTo(420, 360);
        ctx.stroke();

        ctx.fillStyle = '#FFFFFF';
        ctx.font = '10px "SF Mono", monospace';
        ctx.fillText('Multi-HMR 3D Vector (+18° Pitch, -28° Yaw)', 380, 180);
      }
    });

    // F. OBJECT DETECTION
    if (layers.objects) {
      mock.objects.forEach((obj) => {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.strokeRect(obj.x, obj.y, obj.w, obj.h);

        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(obj.x, obj.y - 18, obj.w + 35, 18);

        ctx.fillStyle = '#000000';
        ctx.font = 'bold 10px "SF Mono", monospace';
        ctx.fillText(obj.label, obj.x + 4, obj.y - 5);
      });
    }

    // G. BEHAVIOR HUD
    if (layers.behaviorEvents) {
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(380, 490, 240, 44);
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.strokeRect(380, 490, 240, 44);

      ctx.fillStyle = '#000000';
      ctx.font = 'bold 11px -apple-system, "SF Mono", monospace';
      ctx.fillText('A12 • ANOMALOUS INTERACTION', 390, 508);

      ctx.fillStyle = '#52525B';
      ctx.font = '10px "SF Mono", monospace';
      ctx.fillText('Duration: 4.8s | Confidence: 0.76', 390, 524);
    }

    // CCTV Timestamp HUD top-left
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(16, 16, 260, 40);
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '12px -apple-system, "SF Mono", monospace';
    const timeStr = new Date(1771669338000 + currentTimeSec * 1000).toISOString().substr(11, 8);
    ctx.fillText(`${cameraName.toUpperCase()}`, 26, 34);
    ctx.fillStyle = '#A1A1AA';
    ctx.fillText(`${timeStr} IST • ${fps} FPS`, 26, 48);

  }, [currentTimeSec, activeProvider, layers, cameraName, fps]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="bg-white overflow-hidden flex flex-col rounded-2xl border border-neutral-200 shadow-2xs">
      {/* CCTV Screen Surface */}
      <div className="relative w-full aspect-video bg-black flex items-center justify-center overflow-hidden">
        {liveStreamUrl ? (
          <img src={liveStreamUrl} onError={onLiveStreamError} alt="Live annotated NETRA-4D stream" className="w-full h-full object-contain" />
        ) : recordedVideoUrl ? (
          <video src={recordedVideoUrl} controls autoPlay muted loop playsInline className="w-full h-full object-contain" />
        ) : (
          <canvas ref={canvasRef} className="w-full h-full object-contain cursor-crosshair" />
        )}

        {/* Top-Right Feed Status Overlay */}
        <div className="absolute top-3 right-3 flex items-center gap-2 font-mono text-2xs">
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white text-neutral-900 font-bold border border-neutral-200">
            <span className="w-1.5 h-1.5 rounded-full bg-neutral-900" />
            {liveStatus || (liveStreamUrl ? 'LIVE MODEL STREAM' : recordedVideoUrl ? 'VERIFIED PROOF-OF-WORK' : 'DEMO PREVIEW')}
          </span>
          <span className="px-2.5 py-1 rounded-full bg-black/60 text-white border border-white/20">
            {resolution}
          </span>
        </div>
      </div>

      {/* Scrubber Bar */}
      {!liveStreamUrl && !recordedVideoUrl && <div className="p-4 bg-white border-t border-neutral-200 font-mono space-y-2.5">
        {/* Playback Controls & Time Display */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <motion.button
              whileTap={{ scale: 0.94 }}
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 rounded-lg bg-neutral-900 hover:bg-black text-white transition-colors cursor-pointer"
            >
              {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            </motion.button>
            <motion.button
              whileTap={{ scale: 0.94 }}
              onClick={() => setCurrentTimeSec(0)}
              className="p-2 rounded-lg bg-neutral-100 hover:bg-neutral-200 text-neutral-700 transition-colors cursor-pointer"
            >
              <RotateCcw size={14} />
            </motion.button>
            <span className="text-neutral-900 tabular-nums ml-2 font-bold">
              {formatTime(currentTimeSec)} <span className="text-neutral-400 font-normal">/ {formatTime(totalDurationSec)}</span>
            </span>
          </div>

          {/* Anomaly Jump Points */}
          <div className="flex items-center gap-2 text-2xs">
            <span className="text-neutral-500">Jump to Flag:</span>
            <button
              onClick={() => setCurrentTimeSec(14.8)}
              className="px-2.5 py-1 rounded-lg bg-neutral-100 border border-neutral-300 text-neutral-900 hover:bg-neutral-200 transition-colors font-bold cursor-pointer"
            >
              15:42 (Seat A12)
            </button>
            <button
              onClick={() => setCurrentTimeSec(32.4)}
              className="px-2.5 py-1 rounded-lg bg-neutral-100 border border-neutral-300 text-neutral-900 hover:bg-neutral-200 transition-colors font-bold cursor-pointer"
            >
              15:41 (Seat B06)
            </button>
          </div>
        </div>

        {/* Timeline Slider */}
        <div className="relative pt-2 pb-1">
          <div className="relative h-2 bg-neutral-100 rounded-full overflow-hidden border border-neutral-200">
            {/* Playhead Progress */}
            <div
              className="h-full bg-neutral-900 rounded-full transition-all duration-75"
              style={{ width: `${(currentTimeSec / totalDurationSec) * 100}%` }}
            />

            {/* Event Markers on Track */}
            <div className="absolute top-0 bottom-0 left-[12.3%] w-2 bg-neutral-900 rounded-full" title="15:42 Flag" />
            <div className="absolute top-0 bottom-0 left-[27.0%] w-2 bg-neutral-600 rounded-full" title="15:41 Flag" />
            <div className="absolute top-0 bottom-0 left-[56.6%] w-2 bg-neutral-400 rounded-full" title="15:39 Flag" />
          </div>

          <input
            type="range"
            min={0}
            max={totalDurationSec}
            step={0.1}
            value={currentTimeSec}
            onChange={(e) => setCurrentTimeSec(parseFloat(e.target.value))}
            className="absolute inset-0 w-full opacity-0 cursor-pointer"
          />
        </div>
      </div>}
    </div>
  );
};
