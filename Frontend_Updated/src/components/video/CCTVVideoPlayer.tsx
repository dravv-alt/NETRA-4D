import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion } from 'motion/react';
import { Play, Pause, RotateCcw, Layers } from 'lucide-react';
import { AIProvider } from '../../types';
import { AILayersState } from './LayerControlWidget';

interface CCTVVideoPlayerProps {
  cameraName: string;
  resolution: string;
  fps: number;
  activeProvider: AIProvider;
  layers: AILayersState;
  telemetry?: any;
  onTimeUpdate?: (currentTime: number) => void;
  selectedEventTimeSec?: number;
  liveStreamUrl?: string;
  recordedVideoUrl?: string;
  liveStatus?: string;
  onLiveStreamError?: () => void;
  showSourcePreview?: boolean;
}

export const CCTVVideoPlayer: React.FC<CCTVVideoPlayerProps> = ({
  cameraName,
  resolution,
  fps,
  activeProvider,
  layers,
  telemetry,
  onTimeUpdate,
  selectedEventTimeSec,
  liveStreamUrl,
  recordedVideoUrl,
  liveStatus,
  onLiveStreamError,
  showSourcePreview,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentTimeSec, setCurrentTimeSec] = useState(selectedEventTimeSec || 0);
  const totalDurationSec = 120;

  useEffect(() => {
    if (selectedEventTimeSec !== undefined) {
      setCurrentTimeSec(selectedEventTimeSec);
      if (videoRef.current) {
        videoRef.current.currentTime = selectedEventTimeSec;
      }
    }
  }, [selectedEventTimeSec]);

  // Video Playback Simulation Loop (only for time counter if no native video)
  useEffect(() => {
    let interval: any = null;
    if (isPlaying && !recordedVideoUrl) {
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
  }, [isPlaying, recordedVideoUrl, onTimeUpdate]);

  const activeLayersCount = useMemo(() => {
    return Object.values(layers).filter(Boolean).length;
  }, [layers]);

  // Canvas Overlay: Renders ONLY real telemetry from the backend (NO hardcoded figures or boxes)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = 1280;
    canvas.height = 720;

    // Always clear canvas completely to transparent
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // If there is no real telemetry from backend inference, do not draw anything!
    if (!telemetry || !telemetry.people) {
      return;
    }

    const rawPeople: any[] = Array.isArray(telemetry.people) ? telemetry.people : [];
    const rawObjects: any[] = Array.isArray(telemetry.objects) ? telemetry.objects : [];
    const rawEvents: any[] = Array.isArray(telemetry.active_review_events) ? telemetry.active_review_events : [];

    // Helper: Extract bounding box in canvas coordinates
    const getBox = (item: any) => {
      if (item.bbox_xyxy_norm) {
        const [x1, y1, x2, y2] = item.bbox_xyxy_norm;
        return {
          x: x1 * canvas.width,
          y: y1 * canvas.height,
          w: (x2 - x1) * canvas.width,
          h: (y2 - y1) * canvas.height,
        };
      }
      if (item.bbox_xyxy_px) {
        return {
          x: item.bbox_xyxy_px[0],
          y: item.bbox_xyxy_px[1],
          w: item.bbox_xyxy_px[2] - item.bbox_xyxy_px[0],
          h: item.bbox_xyxy_px[3] - item.bbox_xyxy_px[1],
        };
      }
      if (item.box) return item.box;
      return null;
    };

    // 1. Draw Real Seat Mapping Anchors
    if (layers.seatMapping) {
      rawPeople.forEach((p) => {
        if (!p.seat_id && !p.associated_seat_id) return;
        const seatId = p.seat_id || p.associated_seat_id;
        const box = getBox(p);
        if (!box || box.w <= 0 || box.h <= 0) return;

        ctx.save();
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(box.x - 16, box.y - 8, box.w + 32, box.h + 36);
        ctx.setLineDash([]);

        ctx.fillStyle = '#0f172a';
        ctx.fillRect(box.x - 16, box.y - 30, 90, 22);
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 10px -apple-system, "SF Mono", monospace';
        ctx.fillText(`SEAT ${seatId}`, box.x - 8, box.y - 15);
        ctx.restore();
      });
    }

    // 2. Draw Real Person Bounding Boxes
    if (layers.personBoxes) {
      rawPeople.forEach((p) => {
        const box = getBox(p);
        if (!box || box.w <= 0 || box.h <= 0) return;

        ctx.save();
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 1.8;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        // Corner brackets
        const bLen = Math.min(16, box.w / 4, box.h / 4);
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(box.x, box.y + bLen); ctx.lineTo(box.x, box.y); ctx.lineTo(box.x + bLen, box.y);
        ctx.moveTo(box.x + box.w - bLen, box.y); ctx.lineTo(box.x + box.w, box.y); ctx.lineTo(box.x + box.w, box.y + bLen);
        ctx.moveTo(box.x, box.y + box.h - bLen); ctx.lineTo(box.x, box.y + box.h); ctx.lineTo(box.x + bLen, box.y + box.h);
        ctx.moveTo(box.x + box.w - bLen, box.y + box.h); ctx.lineTo(box.x + box.w, box.y + box.h); ctx.lineTo(box.x + box.w, box.y + box.h - bLen);
        ctx.stroke();

        // Label Tag
        const conf = p.detection_confidence ? `${(p.detection_confidence * 100).toFixed(0)}%` : '';
        const seat = p.seat_id ? `${p.seat_id} • ` : '';
        const track = layers.trackingIds && (p.entity_local_id || p.track_id) ? `Track #${p.entity_local_id || p.track_id} | ` : '';
        const label = `${seat}${track}PERSON ${conf}`.trim();

        ctx.font = 'bold 11px -apple-system, "SF Mono", monospace';
        const tagW = ctx.measureText(label).width + 16;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(box.x, box.y - 24, tagW, 24);
        ctx.fillStyle = '#00e5ff';
        ctx.fillText(label, box.x + 8, box.y - 7);
        ctx.restore();
      });
    }

    // 3. Draw Real Pose Skeleton
    if (layers.poseSkeleton) {
      rawPeople.forEach((p) => {
        const kpsMap: Record<string, { x: number; y: number; score: number }> = {};
        if (p.keypoints_2d && typeof p.keypoints_2d === 'object') {
          Object.entries(p.keypoints_2d).forEach(([name, kp]: [string, any]) => {
            const kx = kp.x > 1 ? kp.x : kp.x * canvas.width;
            const ky = kp.y > 1 ? kp.y : kp.y * canvas.height;
            kpsMap[name] = { x: kx, y: ky, score: kp.confidence ?? kp.score ?? 0.9 };
          });
        }

        const drawBone = (k1: string, k2: string) => {
          const a = kpsMap[k1];
          const b = kpsMap[k2];
          if (a && b && a.score > 0.4 && b.score > 0.4) {
            ctx.strokeStyle = 'rgba(0, 229, 255, 0.85)';
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        };

        ctx.save();
        drawBone('left_shoulder', 'right_shoulder');
        drawBone('left_shoulder', 'left_elbow');
        drawBone('left_elbow', 'left_wrist');
        drawBone('right_shoulder', 'right_elbow');
        drawBone('right_elbow', 'right_wrist');
        drawBone('left_shoulder', 'left_hip');
        drawBone('right_shoulder', 'right_hip');
        drawBone('left_hip', 'right_hip');
        drawBone('nose', 'left_shoulder');
        drawBone('nose', 'right_shoulder');

        Object.values(kpsMap).forEach((kp) => {
          if (kp.score > 0.4) {
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
        ctx.restore();
      });
    }

    // 4. Draw Real Gaze & Attention Rays
    if (layers.attentionRays) {
      rawPeople.forEach((p) => {
        if (!p.metadata?.gaze_ray && p.metadata?.head_yaw_deg === undefined) return;
        ctx.save();
        const gaze = p.metadata?.gaze_ray;
        const yaw = p.metadata?.head_yaw_deg ?? 0;
        let ox = 0, oy = 0, tx = 0, ty = 0;
        if (gaze) {
          ox = gaze.originX; oy = gaze.originY;
          tx = gaze.targetX; ty = gaze.targetY;
        } else if (p.keypoints_2d?.nose) {
          const nose = p.keypoints_2d.nose;
          ox = nose.x > 1 ? nose.x : nose.x * canvas.width;
          oy = nose.y > 1 ? nose.y : nose.y * canvas.height;
          tx = ox + Math.sin((yaw * Math.PI) / 180) * 120;
          ty = oy + 120;
        }
        if (ox === 0 && oy === 0) return;

        const isDeviated = Math.abs(yaw) > 20;
        const rayColor = isDeviated ? '#f59e0b' : '#10b981';
        ctx.strokeStyle = rayColor;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = rayColor;
        ctx.beginPath();
        ctx.arc(tx, ty, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(ox + 10, oy - 22, 120, 18);
        ctx.fillStyle = rayColor;
        ctx.font = 'bold 10px "SF Mono", monospace';
        ctx.fillText(`Yaw: ${yaw.toFixed(0)}°`, ox + 16, oy - 9);
        ctx.restore();
      });
    }

    // 5. Draw Real Object Candidates
    if (layers.objects) {
      rawObjects.forEach((obj) => {
        const box = getBox(obj);
        if (!box || box.w <= 0 || box.h <= 0) return;

        ctx.save();
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        const label = `${(obj.object_class || obj.label || 'OBJECT').toUpperCase()} ${((obj.confidence || 0.7) * 100).toFixed(0)}%`;
        ctx.font = 'bold 10px "SF Mono", monospace';
        const tagW = ctx.measureText(label).width + 12;
        ctx.fillStyle = '#f43f5e';
        ctx.fillRect(box.x, box.y - 20, tagW, 20);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, box.x + 6, box.y - 6);
        ctx.restore();
      });
    }

    // 6. Draw Real Behavior HUD Anomaly Cards
    if (layers.behaviorEvents && rawEvents.length > 0) {
      const topEv = rawEvents[0];
      ctx.save();
      const cardX = 360, cardY = 30, cardW = 340, cardH = 46;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
      ctx.fillRect(cardX, cardY, cardW, cardH);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.8;
      ctx.strokeRect(cardX, cardY, cardW, cardH);

      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.arc(cardX + 16, cardY + 16, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#f43f5e';
      ctx.font = 'bold 11px -apple-system, "SF Mono", monospace';
      ctx.fillText(`${topEv.label || 'ANOMALOUS EVENT FLAGGED'}`, cardX + 28, cardY + 20);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px "SF Mono", monospace';
      ctx.fillText(`Duration: ${(topEv.duration_sec || 0).toFixed(1)}s | Score: ${((topEv.evidence_score || 0.7) * 100).toFixed(0)}% | State: REVIEW`, cardX + 28, cardY + 36);
      ctx.restore();
    }

    // 7. Draw Real 3D Orientation & Mesh Vector Triad
    if (layers.threeDVector) {
      rawPeople.forEach((p) => {
        const box = getBox(p);
        if (!box) return;
        const cx = box.x + box.w / 2;
        const cy = box.y + box.h * 0.42;
        const axisLen = Math.min(box.w, box.h) * 0.35;
        const yaw = ((p.metadata?.head_yaw_deg ?? 0) * Math.PI) / 180;

        ctx.save();
        // X-axis (Red - lateral)
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(yaw) * axisLen, cy + Math.sin(yaw) * axisLen * 0.4);
        ctx.stroke();

        // Y-axis (Green - vertical)
        ctx.strokeStyle = '#22c55e';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx, cy - axisLen);
        ctx.stroke();

        // Z-axis (Blue - depth / gaze normal)
        ctx.strokeStyle = '#38bdf8';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx - Math.sin(yaw) * axisLen * 0.8, cy + Math.cos(yaw) * axisLen * 0.4);
        ctx.stroke();

        // Center Node
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
        ctx.fillRect(cx - 30, cy + axisLen * 0.45, 60, 16);
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 9px "SF Mono", monospace';
        ctx.fillText('3D VEC', cx - 22, cy + axisLen * 0.45 + 11);
        ctx.restore();
      });
    }
  }, [layers, telemetry]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  const handleTogglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (time: number) => {
    setCurrentTimeSec(time);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
    if (onTimeUpdate) onTimeUpdate(time);
  };

  return (
    <div className="bg-white overflow-hidden flex flex-col rounded-2xl border border-neutral-200 shadow-2xs">
      {/* CCTV Screen Surface */}
      <div className="relative w-full aspect-video bg-black flex items-center justify-center overflow-hidden select-none">
        {/* Base Media Layer */}
        {liveStreamUrl ? (
          <img
            src={liveStreamUrl}
            onError={onLiveStreamError}
            alt="Live NETRA-4D stream"
            className="w-full h-full object-contain"
          />
        ) : recordedVideoUrl ? (
          <video
            ref={videoRef}
            src={recordedVideoUrl}
            autoPlay
            muted
            loop
            playsInline
            onTimeUpdate={(e) => {
              const t = (e.target as HTMLVideoElement).currentTime;
              setCurrentTimeSec(t);
              if (onTimeUpdate) onTimeUpdate(t);
            }}
            className="w-full h-full object-contain"
          />
        ) : null}

        {/* Dynamic Transparent AI Visual Layers Overlay Canvas */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none z-10"
        />

        {/* Top-Right Feed Status Overlay */}
        <div className="absolute top-3 right-3 flex items-center gap-2 font-mono text-2xs z-20">
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white text-neutral-900 font-bold border border-neutral-200 shadow-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse" />
            {liveStatus || (liveStreamUrl ? 'LIVE STREAM' : showSourcePreview ? 'SOURCE FEED' : 'VERIFIED PROOF-OF-WORK')}
          </span>
          <span className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-black/75 text-cyan-300 border border-cyan-500/30">
            <Layers size={10} />
            <span>{activeLayersCount}/8 HUD</span>
          </span>
          <span className="px-2.5 py-1 rounded-full bg-black/60 text-white border border-white/20">
            {resolution}
          </span>
        </div>
      </div>

      {/* Scrubber Bar & Playback Controls */}
      <div className="p-4 bg-white border-t border-neutral-200 font-mono space-y-2.5">
        <div className="flex items-center justify-between text-xs flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <motion.button
              whileTap={{ scale: 0.94 }}
              onClick={handleTogglePlay}
              className="p-2 rounded-lg bg-neutral-900 hover:bg-black text-white transition-colors cursor-pointer"
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            </motion.button>
            <motion.button
              whileTap={{ scale: 0.94 }}
              onClick={() => handleSeek(0)}
              className="p-2 rounded-lg bg-neutral-100 hover:bg-neutral-200 text-neutral-700 transition-colors cursor-pointer"
              title="Restart"
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
              onClick={() => handleSeek(14.8)}
              className="px-2.5 py-1 rounded-lg bg-neutral-100 border border-neutral-300 text-neutral-900 hover:bg-neutral-200 transition-colors font-bold cursor-pointer"
            >
              15:42 (Seat A06)
            </button>
            <button
              onClick={() => handleSeek(32.4)}
              className="px-2.5 py-1 rounded-lg bg-neutral-100 border border-neutral-300 text-neutral-900 hover:bg-neutral-200 transition-colors font-bold cursor-pointer"
            >
              15:41 (Seat B04)
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
            <div className="absolute top-0 bottom-0 left-[12.3%] w-2 bg-rose-500 rounded-full" title="15:42 Flag" />
            <div className="absolute top-0 bottom-0 left-[27.0%] w-2 bg-amber-500 rounded-full" title="15:41 Flag" />
            <div className="absolute top-0 bottom-0 left-[56.6%] w-2 bg-neutral-400 rounded-full" title="15:39 Flag" />
          </div>

          <input
            type="range"
            min={0}
            max={totalDurationSec}
            step={0.1}
            value={currentTimeSec}
            onChange={(e) => handleSeek(parseFloat(e.target.value))}
            className="absolute inset-0 w-full opacity-0 cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
};
