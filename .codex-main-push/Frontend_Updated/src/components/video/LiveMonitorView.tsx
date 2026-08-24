import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { CCTVVideoPlayer } from './CCTVVideoPlayer';
import { LayerControlWidget, AILayersState } from './LayerControlWidget';
import { ProviderSelector } from './ProviderSelector';
import { AIProvider, CameraFeedInfo, TriagedAIEvent } from '../../types';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { ArrowRight, Camera, CheckCircle2, ExternalLink, FileJson, Image, Radio, Square, Upload, Video } from 'lucide-react';
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
  const fileInputRef = useRef<HTMLInputElement>(null);
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

  const handleToggleLayer = (key: keyof AILayersState) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

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
    if (!session?.session_id) return;
    const socket = new WebSocket(`${apiBase.replace(/^http/, 'ws')}/ws/live/${session.session_id}`);
    socket.onmessage = (message) => { setTelemetry(JSON.parse(message.data)); setFallbackActive(false); };
    socket.onerror = () => { setLiveError('Live telemetry unavailable; showing a verified previous run.'); setShowSourcePreview(false); setFallbackActive(true); };
    return () => socket.close();
  }, [apiBase, session?.session_id]);

  useEffect(() => {
    if (!session?.session_id || telemetry) return;
    const timer = window.setTimeout(() => {
      setFallbackActive(true);
      setShowSourcePreview(false);
      setLiveError('Live first frame is taking longer than expected; verified proof-of-work fallback is playing.');
    }, 10000);
    return () => window.clearTimeout(timer);
  }, [session?.session_id, telemetry]);

  const startWebcam = async () => {
    setLiveError(''); setTelemetry(null); setFallbackActive(false);
    try {
      const response = await fetch(`${apiBase}/api/live/sessions/webcam?source=0`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      setSession(await response.json());
    } catch (error) { setLiveError(`${String(error)}; showing verified fallback.`); setShowSourcePreview(false); setFallbackActive(true); }
  };

  const uploadVideo = async (file: File) => {
    setLiveError(''); setTelemetry(null); setFallbackActive(false);
    try {
      const response = await fetch(`${apiBase}/api/live/sessions/upload?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file,
      });
      if (!response.ok) throw new Error(await response.text());
      setSession(await response.json());
    } catch (error) { setLiveError(`${String(error)}; showing verified fallback.`); setShowSourcePreview(false); setFallbackActive(true); }
  };

  const stopSession = async () => {
    if (session?.session_id) await fetch(`${apiBase}/api/live/sessions/${session.session_id}/stop`, { method: 'POST' });
    setSession(null); setTelemetry(null); setShowSourcePreview(false); setFallbackActive(true);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm">
      {/* Top Controls */}
      <div className="flex items-center justify-between flex-wrap gap-4 pb-4 border-b border-neutral-200">
        <div className="flex items-center gap-2">
          <Button onClick={startWebcam} className="gap-2 rounded-xl"><Camera size={14} /> Start Webcam</Button>
          <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="gap-2 rounded-xl"><Upload size={14} /> Upload & Stream</Button>
          <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && uploadVideo(e.target.files[0])} />
          {session && <Button variant="destructive" onClick={stopSession} className="gap-2 rounded-xl"><Square size={13} /> Stop</Button>}
          <input
            className="w-64 rounded-xl border border-neutral-200 px-3 py-2 text-xs font-mono"
            placeholder="RTSP via MediaMTX: rtsp://127.0.0.1:8554/exam"
            onKeyDown={async (e) => {
              if (e.key !== 'Enter') return;
              const source = (e.currentTarget as HTMLInputElement).value;
              setTelemetry(null); setShowSourcePreview(false); setFallbackActive(false); setLiveError('');
              try {
                const response = await fetch(`${apiBase}/api/live/sessions/webcam?source=${encodeURIComponent(source)}`, { method: 'POST' });
                if (!response.ok) throw new Error(await response.text());
                setSession(await response.json());
              } catch (error) { setLiveError(`${String(error)}; showing verified fallback.`); setFallbackActive(true); }
            }}
          />
        </div>
        {/* Camera Selector */}
        <div className="flex items-center gap-1.5 bg-white p-2 rounded-2xl border border-neutral-200 shadow-2xs">
          <span className="text-neutral-400 text-xs font-bold px-3 uppercase tracking-wider">Feed:</span>
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

      {/* Main Content Grid: Player + Telemetry Cards */}
      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Main Video Viewport (8 Cols) */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <CCTVVideoPlayer
            cameraName={selectedCam.name}
            resolution={selectedCam.resolution}
            fps={selectedCam.fps}
            activeProvider={activeProvider}
            layers={layers}
            liveStreamUrl={session && !fallbackActive ? `${apiBase}${session.stream_url}` : undefined}
            recordedVideoUrl={selectedSample && (fallbackActive || !session) ? `${apiBase}${showSourcePreview && selectedSample.source.preview_url ? selectedSample.source.preview_url : selectedSample.annotated_video_url}` : undefined}
            liveStatus={session && !fallbackActive ? (telemetry ? 'LIVE • FRAME-BY-FRAME' : 'INITIALIZING MODELS') : selectedSample ? (showSourcePreview ? 'SOURCE INPUT • NO ANNOTATIONS' : 'VERIFIED PREVIOUS RUN') : undefined}
            onLiveStreamError={() => { setShowSourcePreview(false); setFallbackActive(true); setLiveError('Live video transport failed; verified output fallback activated.'); }}
          />
          {liveError && <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">{liveError}</div>}
        </div>

        {/* Live Tracking HUD & Layer Widgets (4 Cols) */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          {/* AI Layers Toggle Widget */}
          <LayerControlWidget
            layers={layers}
            onToggleLayer={handleToggleLayer}
          />

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
              </div>

              <div className="space-y-1.5">
                <ConfidenceBar
                  value={telemetry ? Math.max(0, ...(telemetry.active_review_events || []).map((e: any) => e.evidence_score)) : Number(selectedSample?.metrics?.max_evidence_score || 0)}
                  label="Sustained Temporal Evidence"
                  color="amber"
                  size="sm"
                />
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
              {selectedSample?.source?.preview_url && <Button size="sm" variant={showSourcePreview ? 'default' : 'outline'} onClick={() => { setShowSourcePreview(true); setFallbackActive(true); }}>Source Input</Button>}
              <Button size="sm" variant={!showSourcePreview ? 'default' : 'outline'} onClick={() => { setShowSourcePreview(false); setFallbackActive(true); }}>Annotated Output</Button>
              <span className="rounded-full bg-emerald-50 px-3 py-1 text-2xs font-bold text-emerald-700">REAL GPU OUTPUT</span>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {showcaseSamples.map((sample) => (
              <div role="button" tabIndex={0} key={sample.asset_id} onClick={() => { setSelectedSampleId(sample.asset_id); setShowSourcePreview(false); setFallbackActive(true); }} onKeyDown={(event) => { if (event.key === 'Enter') { setSelectedSampleId(sample.asset_id); setShowSourcePreview(false); setFallbackActive(true); } }} className={`cursor-pointer rounded-2xl border p-4 text-left transition ${selectedSample?.asset_id === sample.asset_id ? 'border-neutral-900 ring-1 ring-neutral-900' : 'border-neutral-200 hover:border-neutral-400'}`}>
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
