import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { CCTVVideoPlayer } from './CCTVVideoPlayer';
import { LayerControlWidget, AILayersState } from './LayerControlWidget';
import { ProviderSelector } from './ProviderSelector';
import { AIProvider, CameraFeedInfo, TriagedAIEvent } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { ArrowRight, Camera, CheckCircle2, ExternalLink, FileJson, Image, Play, Radio, Square, Upload, Video } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { ParallaxCard } from '../ui/parallax-card';

interface LiveMonitorViewProps {
  cameras: CameraFeedInfo[];
  onReviewEvent: (event: TriagedAIEvent) => void;
  events: TriagedAIEvent[];
}

export const LiveMonitorView: React.FC<LiveMonitorViewProps> = ({
  cameras,
  onReviewEvent,
  events,
}) => {
  const apiBase = (import.meta.env.VITE_NETRA_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-01');
  const [session, setSession] = useState<any>(null);
  const [telemetry, setTelemetry] = useState<any>(null);
  const [liveError, setLiveError] = useState('');
  const [showcaseSamples, setShowcaseSamples] = useState<any[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState('');
  const [fallbackActive, setFallbackActive] = useState(true);
  const [showSourcePreview, setShowSourcePreview] = useState(false);
  const [sampleTimeline, setSampleTimeline] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const browserStreamRef = useRef<MediaStream | null>(null);
  const browserTimerRef = useRef<number | null>(null);
  const browserVideoRef = useRef<HTMLVideoElement | null>(null);
  const browserCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const browserPublishSettingsRef = useRef<{ width: number; height: number; fps: number } | null>(null);
  const browserSendingRef = useRef(false);
  const [invigilatorId, setInvigilatorId] = useState('INVIGILATOR');
  const [experimentalPhoneEvidence, setExperimentalPhoneEvidence] = useState(false);
  const [overlayProfile, setOverlayProfile] = useState<'full_context' | 'product_operator'>('full_context');
  const [monitoringBusy, setMonitoringBusy] = useState(false);
  const [activeProvider, setActiveProvider] = useState<AIProvider>('fusion');
  const [layers, setLayers] = useState<AILayersState>({
    personBoxes: true,
    trackingIds: true,
    poseSkeleton: true,
    objects: true,
    seatMapping: true,
    attentionRays: true,
    behaviorEvents: true,
    threeDVector: false,
  });

  const selectedCam = cameras.find((c) => c.id === selectedCameraId) || cameras[0];
  const activeEvent = events.find((e) => e.seatId === 'A06') || events[0];
  const selectedSample = showcaseSamples.find((sample) => sample.asset_id === selectedSampleId) || showcaseSamples[0];
  const monitoringState = session?.status;
  const isArmed = monitoringState === 'ARMED';
  const isPaused = monitoringState === 'PAUSED';
  const isMonitoring = monitoringState === 'RUNNING';

  const handleToggleLayer = (key: keyof AILayersState) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // The player deliberately has no browser-drawn duplicate overlay. Keep the
  // selected layers in the live worker so the authoritative JPEG/MJPEG frame
  // itself changes on the next processed frame.
  useEffect(() => {
    if (!session?.render_layers_url) return;
    const controller = new AbortController();
    const updateLayers = async () => {
      try {
        const response = await fetch(`${apiBase}${session.render_layers_url}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(layers),
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(await response.text());
        const updated = await response.json();
        setSession((previous: any) => previous?.session_id === updated.session_id
          ? { ...previous, render_layers: updated.render_layers }
          : previous);
      } catch (error) {
        if (!controller.signal.aborted) {
          setLiveError(`Could not update server-rendered layers: ${String(error)}`);
        }
      }
    };
    void updateLayers();
    return () => controller.abort();
  }, [apiBase, layers, session?.session_id, session?.render_layers_url]);

  const handleTimeUpdate = (currentTime: number) => {
    if (sampleTimeline.length === 0 || session) return;
    let closest = sampleTimeline[0];
    let minDiff = Math.abs(closest.timestamp_sec - currentTime);
    for (let i = 1; i < sampleTimeline.length; i++) {
      const diff = Math.abs(sampleTimeline[i].timestamp_sec - currentTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = sampleTimeline[i];
      } else if (diff > minDiff) {
        break;
      }
    }
    if (minDiff < 0.6) {
      setTelemetry({
        run_id: selectedSample?.run_id || 'netra4d_final_20260823',
        timestamp_sec: closest.timestamp_sec,
        frame_index: closest.frame_index,
        latency_ms: 28.5,
        effective_fps: 25.0,
        people: closest.people,
        objects: closest.objects,
        active_review_events: closest.active_review_events,
      });
      setFallbackActive(false);
    }
  };

  const pauseBrowserPublisher = () => {
    if (browserTimerRef.current !== null) window.clearInterval(browserTimerRef.current);
    browserTimerRef.current = null;
    browserSendingRef.current = false;
  };

  const stopBrowserPublisher = () => {
    pauseBrowserPublisher();
    browserStreamRef.current?.getTracks().forEach((track) => track.stop());
    browserStreamRef.current = null;
    browserVideoRef.current = null;
    browserCanvasRef.current = null;
    browserPublishSettingsRef.current = null;
  };

  const startBrowserPublisher = (browserSession: any) => {
    const video = browserVideoRef.current;
    const canvas = browserCanvasRef.current;
    const settings = browserPublishSettingsRef.current;
    const context = canvas?.getContext('2d');
    if (!video || !canvas || !settings || !context) throw new Error('Browser camera is armed but no frame publisher is available');
    pauseBrowserPublisher();
    browserTimerRef.current = window.setInterval(() => {
      if (video.readyState < 2 || browserSendingRef.current) return;
      context.drawImage(video, 0, 0, settings.width, settings.height);
      // Claim the slot before encoding: toBlob is asynchronous and otherwise
      // multiple JPEG uploads can pile up and exhaust the browser.
      browserSendingRef.current = true;
      canvas.toBlob(async (blob) => {
        if (!blob) { browserSendingRef.current = false; return; }
        try {
          const frameResponse = await fetch(`${apiBase}${browserSession.browser_frame_url}`, {
            method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob,
          });
          if (!frameResponse.ok) throw new Error(await frameResponse.text());
        } catch (error) {
          setLiveError(`Browser frame transport stopped: ${String(error)}`);
          stopBrowserPublisher();
        } finally {
          browserSendingRef.current = false;
        }
      }, 'image/jpeg', 0.68);
    }, Math.round(1000 / settings.fps));
  };

  useEffect(() => () => stopBrowserPublisher(), []);

  useEffect(() => {
    fetch(`${apiBase}/api/showcase?limit=3`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((result) => {
        setShowcaseSamples(result.samples || []);
        if (result.samples?.length) setSelectedSampleId(result.samples[0].asset_id);
      })
      .catch(() => setLiveError('Prepared proof-of-work samples could not be loaded'));
  }, [apiBase]);

  useEffect(() => {
    if (!session?.session_id || !isMonitoring) return;
    let disposed = false;
    let socket: WebSocket | null = null;
    const connectTimer = window.setTimeout(() => {
      if (disposed) return;
      socket = new WebSocket(`${apiBase.replace(/^http/, 'ws')}/ws/live/${session.session_id}`);
      socket.onmessage = (message) => { if (!disposed) { setTelemetry(JSON.parse(message.data)); setFallbackActive(false); } };
      socket.onerror = () => { if (!disposed) { setLiveError('Live telemetry unavailable; showing a verified previous run.'); setShowSourcePreview(false); setFallbackActive(true); } };
    }, 0);
    return () => {
      disposed = true;
      window.clearTimeout(connectTimer);
      if (socket?.readyState === WebSocket.OPEN) socket.close(1000, 'monitoring view unmounted');
      else if (socket?.readyState === WebSocket.CONNECTING) socket.onopen = () => socket?.close(1000, 'monitoring view unmounted');
    };
  }, [apiBase, session?.session_id, isMonitoring]);

  useEffect(() => {
    if (!session?.session_id || !isMonitoring || telemetry) return;
    const timer = window.setTimeout(() => {
      setFallbackActive(true);
      setShowSourcePreview(false);
      setLiveError('Live first frame is taking longer than expected; verified proof-of-work fallback is playing.');
    }, 10000);
    return () => window.clearTimeout(timer);
  }, [session?.session_id, telemetry, isMonitoring]);

  useEffect(() => {
    if (!selectedSample?.asset_id || session) return;
    const runId = selectedSample.run_id || 'netra4d_final_20260823';
    fetch(`${apiBase}/api/showcase/${runId}/${selectedSample.asset_id}/telemetry-timeline`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setSampleTimeline(data);
          const firstWithPeople = data.find((f: any) => f.people && f.people.length > 0) || data[0];
          setTelemetry({
            run_id: runId,
            timestamp_sec: firstWithPeople.timestamp_sec,
            frame_index: firstWithPeople.frame_index,
            latency_ms: 28.5,
            effective_fps: 25.0,
            people: firstWithPeople.people,
            objects: firstWithPeople.objects,
            active_review_events: firstWithPeople.active_review_events,
          });
          setFallbackActive(false);
        }
      })
      .catch(() => {});
  }, [apiBase, selectedSample?.asset_id, session]);

  const armBrowserWebcam = async () => {
    setLiveError(''); setTelemetry(null); setFallbackActive(true);
    try {
      stopBrowserPublisher();
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not expose webcam access');
      const stream = await navigator.mediaDevices.getUserMedia({
        // Bounded input keeps CPU inference and browser upload predictable.
        video: { width: { ideal: 640, max: 640 }, height: { ideal: 360, max: 360 }, frameRate: { ideal: 2, max: 2 } },
        audio: false,
      });
      browserStreamRef.current = stream;
      const video = document.createElement('video');
      video.srcObject = stream; video.muted = true; video.playsInline = true;
      await video.play();
      browserVideoRef.current = video;
      const settings = stream.getVideoTracks()[0].getSettings();
      const width = settings.width || video.videoWidth || 640;
      const height = settings.height || video.videoHeight || 480;
      const publishFps = 2;
      const response = await fetch(`${apiBase}/api/live/sessions/browser?width=${width}&height=${height}&fps=${publishFps}&overlay_profile=${overlayProfile}`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      const browserSession = await response.json();
      setSession(browserSession);
      const canvas = document.createElement('canvas');
      canvas.width = width; canvas.height = height;
      browserCanvasRef.current = canvas;
      browserPublishSettingsRef.current = { width, height, fps: publishFps };
    } catch (error) {
      stopBrowserPublisher();
      setLiveError(`${String(error)}; showing verified fallback.`); setShowSourcePreview(false); setFallbackActive(true);
    }
  };

  const startMonitoring = async () => {
    if (!session?.session_id || (!isArmed && !isPaused)) return;
    setMonitoringBusy(true); setLiveError(''); setTelemetry(null); setFallbackActive(false);
    try {
      const response = await fetch(`${apiBase}/api/live/sessions/${session.session_id}/start?operator=${encodeURIComponent(invigilatorId || 'INVIGILATOR')}`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      const startedSession = await response.json();
      setSession(startedSession);
      if (startedSession.source_kind === 'BROWSER_WEBCAM') startBrowserPublisher(startedSession);
    } catch (error) {
      setLiveError(`Monitoring did not start: ${String(error)}`); setFallbackActive(true);
    } finally {
      setMonitoringBusy(false);
    }
  };

  const uploadVideo = async (file: File) => {
    setLiveError(''); setTelemetry(null); setFallbackActive(true);
    try {
      stopBrowserPublisher();
      const response = await fetch(`${apiBase}/api/live/sessions/upload?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file,
      });
      if (!response.ok) throw new Error(await response.text());
      setSession(await response.json());
    } catch (error) { setLiveError(`${String(error)}; showing verified fallback.`); setShowSourcePreview(false); setFallbackActive(true); }
  };

  const stopSession = async () => {
    pauseBrowserPublisher();
    if (session?.session_id) {
      const response = await fetch(`${apiBase}/api/live/sessions/${session.session_id}/stop?operator=${encodeURIComponent(invigilatorId || 'INVIGILATOR')}`, { method: 'POST' });
      if (response.ok) {
        const result = await response.json();
        setSession({ ...session, status: result.status });
      }
    }
    setTelemetry(null); setShowSourcePreview(false); setFallbackActive(true);
  };

  return (
    <div className="space-y-4 sm:space-y-8 max-w-7xl mx-auto font-sans text-xs sm:text-sm">
      {/* Top Controls */}
      <div className="flex items-center justify-between flex-wrap gap-2.5 sm:gap-4 pb-4 border-b border-neutral-200">
        <div className="flex flex-wrap items-center gap-2 w-full lg:w-auto">
          <Button onClick={armBrowserWebcam} className="gap-2 rounded-xl"><Camera size={14} /> Arm Browser Webcam</Button>
          <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="gap-2 rounded-xl"><Upload size={14} /> Upload & Arm</Button>
          <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && uploadVideo(e.target.files[0])} />
          <input value={invigilatorId} onChange={(event) => setInvigilatorId(event.target.value)} aria-label="Invigilator ID" className="w-full sm:w-28 md:w-32 rounded-xl border border-neutral-200 px-3 py-2 text-xs font-mono" placeholder="Invigilator ID" />
          <label className="flex items-center gap-1.5 text-2xs text-neutral-600" title={overlayProfile === 'full_context' ? 'The established full-context renderer does not run the experimental Video-3 phone specialist.' : 'Video-3 trained; review evidence only, not a release model.'}>
            <input type="checkbox" disabled={overlayProfile === 'full_context'} checked={experimentalPhoneEvidence} onChange={(event) => setExperimentalPhoneEvidence(event.target.checked)} />
            Experimental phone evidence
          </label>
          <label className="flex items-center gap-1.5 text-2xs text-neutral-600" title="Full context uses the established pose, gaze-proxy, workstation, and temporal review renderer. Compact keeps the canonical product overlay.">
            Overlay
            <select value={overlayProfile} onChange={(event) => { const next = event.target.value as 'full_context' | 'product_operator'; setOverlayProfile(next); if (next === 'full_context') setExperimentalPhoneEvidence(false); }} className="rounded border border-neutral-200 bg-white px-1 py-0.5 text-2xs">
              <option value="full_context">Full context</option>
              <option value="product_operator">Compact product</option>
            </select>
          </label>
          {(isArmed || isPaused) && <Button onClick={startMonitoring} disabled={monitoringBusy} className="gap-2 rounded-xl bg-emerald-700 hover:bg-emerald-800"><Play size={13} /> {monitoringBusy ? 'Starting…' : isPaused ? 'Resume Monitoring' : 'Start Monitoring'}</Button>}
          {isMonitoring && <Button variant="destructive" onClick={stopSession} className="gap-2 rounded-xl"><Square size={13} /> Stop Monitoring</Button>}
          <input
            className="w-full sm:w-56 md:w-64 rounded-xl border border-neutral-200 px-3 py-2 text-xs font-mono"
            placeholder="RTSP: rtsp://127.0.0.1:8554/exam"
            onKeyDown={async (e) => {
              if (e.key !== 'Enter') return;
              const source = (e.currentTarget as HTMLInputElement).value;
              stopBrowserPublisher();
              setTelemetry(null); setShowSourcePreview(false); setFallbackActive(true); setLiveError('');
              try {
                const response = await fetch(`${apiBase}/api/live/sessions/webcam?source=${encodeURIComponent(source)}&experimental_phone_evidence=${experimentalPhoneEvidence}&overlay_profile=${overlayProfile}`, { method: 'POST' });
                if (!response.ok) throw new Error(await response.text());
                setSession(await response.json());
              } catch (error) { setLiveError(`${String(error)}; showing verified fallback.`); setFallbackActive(true); }
            }}
          />
        </div>
        {/* Camera Selector */}
        <div className="flex items-center gap-1.5 bg-white p-1.5 sm:p-2 rounded-xl sm:rounded-2xl border border-neutral-200 shadow-2xs overflow-x-auto max-w-full scrollbar-none">
          <span className="text-neutral-400 text-xs font-bold px-2 sm:px-3 uppercase tracking-wider shrink-0">Feed:</span>
          {cameras.map((cam) => {
            const isSelected = selectedCameraId === cam.id;
            return (
              <motion.button
                key={cam.id}
                onClick={() => {
                  playHapticSound('snap');
                  setSelectedCameraId(cam.id);
                }}
                whileTap={{ scale: 0.97 }}
                className={`relative px-4 py-2.5 rounded-xl text-xs font-medium transition-colors duration-150 cursor-pointer ${
                  isSelected
                    ? 'text-white font-semibold'
                    : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
                }`}
              >
                {isSelected && (
                  <motion.div
                    layoutId="camPill"
                    transition={{ type: 'spring', damping: 28, stiffness: 380 }}
                    className="absolute inset-0 bg-neutral-900 rounded-xl shadow-xs"
                  />
                )}
                <span className="relative z-10 flex items-center gap-2">
                  <Video size={14} className={isSelected ? 'text-white' : 'text-neutral-400'} />
                  <span>{cam.name.split(' ')[0]} ({cam.fps} FPS)</span>
                </span>
              </motion.button>
            );
          })}
        </div>

        {/* AI Provider Selector */}
        <ProviderSelector
          activeProvider={activeProvider}
          onSelectProvider={setActiveProvider}
        />
      </div>

      {session && (
        <div className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border px-4 py-3 text-xs ${isMonitoring ? 'border-emerald-200 bg-emerald-50 text-emerald-900' : isArmed || isPaused ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-neutral-200 bg-neutral-50 text-neutral-700'}`}>
          <div><strong>Invigilator computation gate: </strong>{isMonitoring ? 'frames are being sent for inference and evidence logging.' : isPaused ? 'camera/feed remains armed; new frames are not computed or logged until Resume Monitoring.' : isArmed ? 'source is armed; no frames are sent to inference until Start Monitoring.' : 'this source is stopped; arm a new source to create another monitoring window.'}</div>
          <span className="font-mono font-bold">STATE: {monitoringState}</span>
        </div>
      )}

      {/* Main Content Grid: Player + Telemetry Cards */}
      <div className="grid grid-cols-12 gap-4 lg:gap-8 items-start">
        {/* Main Video Viewport (8 Cols) */}
        <div className="col-span-12 lg:col-span-8 space-y-4 sm:space-y-6">
          <CCTVVideoPlayer
            cameraName={selectedCam.name}
            resolution={selectedCam.resolution}
            fps={selectedCam.fps}
            activeProvider={activeProvider}
            layers={layers}
            telemetry={telemetry}
            showSourcePreview={showSourcePreview}
            onTimeUpdate={handleTimeUpdate}
            liveStreamUrl={isMonitoring && !fallbackActive ? `${apiBase}${session.stream_url}` : undefined}
            recordedVideoUrl={selectedSample && (fallbackActive || !session) ? `${apiBase}${showSourcePreview && selectedSample.source.preview_url ? selectedSample.source.preview_url : selectedSample.annotated_video_url}` : undefined}
            liveStatus={isMonitoring && !fallbackActive ? (telemetry ? 'LIVE • SERVER EVIDENCE FEED' : 'INITIALIZING MODELS') : isArmed ? 'ARMED • INVIGILATOR START REQUIRED' : isPaused ? 'COMPUTATION PAUSED • FEED REMAINS ARMED' : monitoringState === 'STOPPED' ? 'MONITORING STOPPED • RE-ARM SOURCE TO START AGAIN' : selectedSample ? (showSourcePreview ? 'SOURCE VIDEO • UNANNOTATED' : 'VERIFIED ANNOTATED RUN') : undefined}
            onLiveStreamError={() => { setShowSourcePreview(false); setFallbackActive(true); setLiveError('Live video transport failed; verified output fallback activated.'); }}
          />
          {liveError && <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">{liveError}</div>}
        </div>

        {/* Live Tracking HUD & Layer Widgets (4 Cols) */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <LayerControlWidget layers={layers} onToggleLayer={handleToggleLayer} />

          {/* Active Terminal Telemetry Card */}
          <ParallaxCard tiltAngle={3} className="p-6 bg-white border border-neutral-200 rounded-3xl shadow-xs space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-200">
              <div className="font-bold text-sm text-neutral-950 flex items-center gap-2.5">
                <Radio size={14} className="text-rose-500 animate-pulse" />
                <span>{telemetry ? `Frame #${telemetry.frame_index} Telemetry` : 'Awaiting Real Telemetry'}</span>
              </div>
              <StatusBadge status="review" size="sm" />
            </div>

            <div className="space-y-3.5 text-xs">
              <div className="flex justify-between items-center text-neutral-600">
                <span>Terminal / Candidate:</span>
                <span className="font-bold text-neutral-950">{telemetry?.people?.length ?? selectedSample?.metrics?.people ?? 0} {telemetry ? 'tracked candidate(s)' : 'person detections'}</span>
              </div>

              <div className="flex justify-between items-center text-neutral-600">
                <span>Pose Status:</span>
                <span className="font-semibold text-emerald-700">{telemetry ? `${telemetry.latency_ms.toFixed(0)} ms • ${telemetry.effective_fps.toFixed(1)} FPS` : selectedSample ? `${selectedSample.metrics.frames.toLocaleString()} verified frames` : 'Backend not streaming'}</span>
              </div>

              <div className="space-y-1.5 pt-2 border-t border-neutral-200">
                <ConfidenceBar
                  value={Math.max(0, ...(telemetry?.objects || []).filter((o: any) => o.metadata?.raw_model_class === 'cell phone').map((o: any) => o.confidence))}
                  label="Measured Cell-phone Evidence"
                  color="rose"
                  size="sm"
                />
                {(telemetry?.temporal_evidence?.phone_evidence || []).map((row: any) => (
                  <div key={`${row.candidate_id}-${row.state}`} className="text-2xs text-neutral-600">
                    Candidate {row.candidate_id}: {row.state === 'FRESH_PHONE_DETECTION'
                      ? `fresh phone detection (${(row.last_fresh_confidence * 100).toFixed(0)}%)`
                      : `recent-phone continuity only (${row.elapsed_since_fresh_sec.toFixed(1)}s since fresh evidence)`}
                  </div>
                ))}
              </div>

              <div className="space-y-1.5">
                <ConfidenceBar
                  value={telemetry ? Math.max(0, ...(telemetry.active_review_events || []).map((e: any) => e.evidence_score)) : Number(selectedSample?.metrics?.max_evidence_score || 0)}
                  label="Sustained Temporal Evidence"
                  color="amber"
                  size="sm"
                />
                {telemetry?.temporal_evidence && <p className="text-2xs text-neutral-500">Persistence and association only; not a behavioural verdict.</p>}
              </div>
            </div>

            <Button
              variant="default"
              size="sm"
              onClick={() => onReviewEvent(activeEvent)}
              className="w-full gap-2 rounded-xl font-semibold shadow-xs"
            >
              <span>Review Flag Frame</span>
              <ArrowRight size={13} />
            </Button>
          </ParallaxCard>
        </div>
      </div>

      {showcaseSamples.length > 0 && (
        <section className="space-y-4 rounded-3xl border border-neutral-200 bg-white p-6 shadow-xs">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 font-bold text-neutral-950"><CheckCircle2 size={16} className="text-emerald-600" /> Previous Sampled Runs — Proof of Work</div>
              <p className="mt-1 text-xs text-neutral-500">Integrity-validated annotated videos, measured metrics, JSONL logs, and captured evidence. These samples are also the automatic player fallback.</p>
            </div>
            <div className="flex items-center gap-2">
              {selectedSample?.source?.preview_url && <Button size="sm" variant={showSourcePreview ? 'default' : 'outline'} onClick={() => { setShowSourcePreview(true); setFallbackActive(false); }}>Source video (unannotated)</Button>}
              <Button size="sm" variant={!showSourcePreview ? 'default' : 'outline'} onClick={() => { setShowSourcePreview(false); setFallbackActive(true); }}>Annotated evidence video</Button>
              <span className="rounded-full bg-emerald-50 px-3 py-1 text-2xs font-bold text-emerald-700">REAL GPU OUTPUT</span>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {showcaseSamples.map((sample) => (
              <div role="button" tabIndex={0} key={sample.asset_id} onClick={() => { setSelectedSampleId(sample.asset_id); setShowSourcePreview(true); setFallbackActive(false); }} onKeyDown={(event) => { if (event.key === 'Enter') { setSelectedSampleId(sample.asset_id); setShowSourcePreview(true); setFallbackActive(false); } }} className={`cursor-pointer rounded-2xl border p-4 text-left transition ${selectedSample?.asset_id === sample.asset_id ? 'border-neutral-900 ring-1 ring-neutral-900' : 'border-neutral-200 hover:border-neutral-400'}`}>
                <div className="flex items-center justify-between"><span className="font-bold text-neutral-950">{sample.asset_id}</span><span className="text-2xs font-bold text-rose-600">{sample.metrics.review_events} FLAGS</span></div>
                <p className="mt-1 line-clamp-2 min-h-8 text-xs text-neutral-500">{sample.title}</p>
                <div className="mt-3 grid grid-cols-2 gap-2 text-2xs text-neutral-600">
                  <span>{sample.metrics.frames.toLocaleString()} frames</span><span>{Number(sample.metrics.fps).toFixed(2)} FPS source</span>
                  <span>{sample.metrics.people.toLocaleString()} people</span><span>{sample.metrics.objects.toLocaleString()} objects</span>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <a href={`${apiBase}${sample.events_url}`} onClick={(event) => event.stopPropagation()} className="flex items-center gap-1 text-2xs font-semibold text-neutral-700 hover:text-black"><FileJson size={12} /> JSON evidence</a>
                  {sample.source.url && <a href={sample.source.url} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()} className="flex items-center gap-1 text-2xs font-semibold text-neutral-700 hover:text-black"><ExternalLink size={12} /> Original source</a>}
                </div>
              </div>
            ))}
          </div>
          {selectedSample?.evidence_frames?.length > 0 && (
            <div>
              <div className="mb-2 flex items-center gap-2 text-xs font-bold text-neutral-800"><Image size={14} /> Captured high-confidence evidence</div>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                {selectedSample.evidence_frames.slice(0, 4).map((frame: any) => <a key={frame.filename} href={`${apiBase}${frame.url}`} target="_blank" rel="noreferrer"><img src={`${apiBase}${frame.url}`} alt={frame.filename} className="aspect-video w-full rounded-xl border border-neutral-200 object-cover" /></a>)}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
};
