import React, { useEffect, useState } from 'react';
import { Activity, Camera, Database, FileCheck2, Radio, ShieldCheck } from 'lucide-react';

interface PipelineStatus {
  sessions: number;
  running: number;
  events: number;
  latestRun: string;
}

export const LiveAIAmbientSoundscapeMonitor: React.FC = () => {
  const apiBase = (import.meta.env.VITE_NETRA_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const [status, setStatus] = useState<PipelineStatus>({ sessions: 0, running: 0, events: 0, latestRun: 'Loading…' });
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const [healthResponse, sessionsResponse] = await Promise.all([
          fetch(`${apiBase}/api/health`), fetch(`${apiBase}/api/live/sessions`),
        ]);
        if (!healthResponse.ok || !sessionsResponse.ok) throw new Error('pipeline unavailable');
        const health = await healthResponse.json();
        const sessions = await sessionsResponse.json();
        if (!active) return;
        setStatus({
          sessions: sessions.length,
          running: sessions.filter((session: any) => session.status === 'RUNNING').length,
          events: Number(health.events?.TOTAL || 0),
          latestRun: health.latest_run_id || 'No verified run',
        });
        setConnected(true);
      } catch {
        if (active) setConnected(false);
      }
    };
    void refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [apiBase]);

  const stages = [
    { label: 'Frame ingest', detail: 'Webcam / upload / RTSP', icon: Camera },
    { label: 'Model inference', detail: 'Pose + objects + tracking', icon: Activity },
    { label: 'Temporal evidence', detail: 'Persistence + candidate/PC context', icon: ShieldCheck },
    { label: 'Durable output', detail: 'JSON, snapshots, clips, annotated MP4', icon: Database },
  ];

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-100"><Radio size={18} /></div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">Live Frame & Evidence Pipeline</h3>
            <span className={`text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full border ${connected ? 'bg-emerald-50 text-emerald-800 border-emerald-200' : 'bg-rose-50 text-rose-800 border-rose-200'}`}>
              {connected ? 'BACKEND CONNECTED' : 'BACKEND OFFLINE'}
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">Actual processing stages and persisted review evidence from the NETRA backend—no acoustic or synthetic animation claims.</p>
        </div>
        <div className="text-right"><div className="text-2xl font-extrabold font-mono text-neutral-950">{status.running}</div><div className="text-3xs uppercase tracking-wider text-neutral-500">active sessions</div></div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {stages.map((stage, index) => {
          const Icon = stage.icon;
          return <div key={stage.label} className="flex items-center gap-3 rounded-2xl border border-neutral-200 bg-[#F7F6F2] p-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white border border-neutral-200 text-neutral-800"><Icon size={16} /></div>
            <div className="min-w-0"><div className="text-xs font-bold text-neutral-950">{index + 1}. {stage.label}</div><div className="text-2xs text-neutral-500">{stage.detail}</div></div>
          </div>;
        })}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-2xl bg-neutral-950 p-4 text-white"><div className="text-xl font-extrabold font-mono">{status.sessions}</div><div className="text-3xs text-neutral-400">Recorded sessions</div></div>
        <div className="rounded-2xl bg-neutral-950 p-4 text-white"><div className="text-xl font-extrabold font-mono">{status.events}</div><div className="text-3xs text-neutral-400">Evidence events</div></div>
        <div className="rounded-2xl bg-neutral-950 p-4 text-white"><FileCheck2 size={15} className="mb-1 text-emerald-400" /><div className="truncate text-xs font-bold font-mono" title={status.latestRun}>{status.latestRun}</div><div className="text-3xs text-neutral-400">Latest valid run</div></div>
      </div>
    </div>
  );
};
