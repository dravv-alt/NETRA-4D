import React, { useState } from 'react';
import { Activity, CheckCircle2, Database, FileImage, ShieldCheck, Users } from 'lucide-react';
import { VALIDATED_RUN, VALIDATED_VIDEO_RUNS } from '../../data/benchmarkData';
import { ParallaxCard } from '../ui/parallax-card';

const number = new Intl.NumberFormat('en-IN');

export const BenchmarkLabView: React.FC = () => {
  const [selectedVideo, setSelectedVideo] = useState(VALIDATED_VIDEO_RUNS[0].assetId);
  const active = VALIDATED_VIDEO_RUNS.find((video) => video.assetId === selectedVideo) ?? VALIDATED_VIDEO_RUNS[0];

  const summaryCards = [
    { label: 'Real processed videos', value: VALIDATED_RUN.videos, icon: Database },
    { label: 'Frames rendered', value: number.format(VALIDATED_RUN.frames), icon: Activity },
    { label: 'Event-involved track IDs', value: VALIDATED_RUN.eventCandidateTrackIds, icon: Users },
    { label: 'Temporal review events', value: VALIDATED_RUN.reviewEvents, icon: CheckCircle2 },
    { label: 'Evidence snapshots', value: VALIDATED_RUN.evidenceStills, icon: FileImage },
    { label: 'Privacy masks', value: number.format(VALIDATED_RUN.privacyMasks), icon: ShieldCheck },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between pb-5 border-b border-neutral-200">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-700">Validated proof of work</div>
          <h1 className="mt-2 text-2xl font-bold text-neutral-950">Measured Pipeline Run</h1>
          <p className="mt-1 text-sm text-neutral-500 max-w-3xl">
            Five genuine CCTV source files processed end to end. Every value below is read from the completed run manifest and JSONL evidence artifacts.
          </p>
        </div>
        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-900 max-w-sm">
          <strong>Not an accuracy benchmark.</strong> No reviewed ground truth exists, so F1, precision, recall, and false-positive claims are intentionally omitted.
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
        {summaryCards.map(({ label, value, icon: Icon }) => (
          <div key={label} className="rounded-2xl border border-neutral-200 bg-white p-4 shadow-2xs">
            <Icon size={16} className="text-emerald-700" />
            <div className="mt-3 text-xl font-bold tabular-nums text-neutral-950">{value}</div>
            <div className="mt-1 text-2xs font-semibold uppercase tracking-wide text-neutral-500">{label}</div>
          </div>
        ))}
      </div>

      <div className="rounded-3xl border border-neutral-200 bg-white overflow-hidden shadow-xs">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between p-5 border-b border-neutral-200 bg-neutral-50/80">
          <div>
            <div className="font-bold text-neutral-950">Per-video measured outputs</div>
            <div className="text-xs text-neutral-500">Run ID: {VALIDATED_RUN.runId}</div>
          </div>
          <div className="text-xs text-neutral-600">{VALIDATED_RUN.sourceDurationSec.toFixed(2)} s source footage · {VALIDATED_RUN.device}</div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-xs">
            <thead className="bg-neutral-50 text-neutral-500 uppercase tracking-wide">
              <tr>
                <th className="px-5 py-3 text-left">Video</th><th className="px-4 py-3 text-right">Frames</th>
                <th className="px-4 py-3 text-right">Duration</th><th className="px-4 py-3 text-right">Event track IDs</th>
                <th className="px-4 py-3 text-right">Events</th><th className="px-4 py-3 text-right">Evidence</th>
                <th className="px-4 py-3 text-right">Runtime</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-100">
              {VALIDATED_VIDEO_RUNS.map((video) => (
                <tr key={video.assetId} className="hover:bg-neutral-50">
                  <td className="px-5 py-4"><button className="text-left" onClick={() => setSelectedVideo(video.assetId)}><strong className="text-neutral-950">{video.assetId}</strong><span className="block max-w-[360px] truncate text-neutral-500">{video.filename}</span></button></td>
                  <td className="px-4 py-4 text-right font-mono">{number.format(video.frames)}</td>
                  <td className="px-4 py-4 text-right font-mono">{video.durationSec.toFixed(2)}s</td>
                  <td className="px-4 py-4 text-right font-mono">{video.eventCandidateTrackIds}</td>
                  <td className="px-4 py-4 text-right font-mono font-bold">{video.reviewEvents}</td>
                  <td className="px-4 py-4 text-right font-mono font-bold text-emerald-700">{video.evidenceStills}</td>
                  <td className="px-4 py-4 text-right font-mono">{video.inferenceRuntimeSec.toFixed(1)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-7 items-start">
        <div className="col-span-12 lg:col-span-4 space-y-2">
          <div className="px-1 text-2xs uppercase tracking-widest font-bold text-neutral-400">Validated CCTV sources (5)</div>
          {VALIDATED_VIDEO_RUNS.map((video) => (
            <button key={video.assetId} onClick={() => setSelectedVideo(video.assetId)} className={`w-full rounded-2xl border p-4 text-left transition-colors ${selectedVideo === video.assetId ? 'border-neutral-900 bg-neutral-900 text-white' : 'border-neutral-200 bg-white hover:bg-neutral-50'}`}>
              <div className="flex justify-between text-2xs font-bold"><span>{video.assetId}</span><span>{video.sourceFps.toFixed(2)} FPS · {video.durationSec.toFixed(1)}s</span></div>
              <div className="mt-1 truncate text-xs font-semibold">{video.filename}</div>
            </button>
          ))}
        </div>

        <div className="col-span-12 lg:col-span-8 min-w-0">
          <ParallaxCard tiltAngle={1.5} className="rounded-3xl border border-neutral-200 bg-white p-7 shadow-xs">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between pb-5 border-b border-neutral-200">
            <div><div className="text-lg font-bold text-neutral-950">{active.assetId}</div><div className="mt-1 text-xs text-neutral-500">{active.filename}</div></div>
            <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-800">{active.calibration.replaceAll('_', ' ')}</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 py-5">
            {[
              ['Frame integrity', `${number.format(active.decodedFrames)} decoded / ${number.format(active.frames)} rendered`],
              ['Person observations', number.format(active.personObservations)],
              ['Object observations', number.format(active.objectObservations)],
              ['Gaze proxy known', number.format(active.gazeProxyObservations)],
              ['Privacy masks', number.format(active.privacyMasks)],
              ['Temporal persistence', `${active.temporalDurationSec.toFixed(2)} s summed`],
              ['Longest event', `${active.maxEventDurationSec.toFixed(2)} s`],
              ['Evidence frames', String(active.evidenceStills)],
            ].map(([label, value]) => <div key={label} className="rounded-xl bg-neutral-50 p-3 border border-neutral-100"><div className="text-2xs uppercase font-bold tracking-wide text-neutral-400">{label}</div><div className="mt-1 text-xs font-bold text-neutral-950">{value}</div></div>)}
            </div>

            <div className="rounded-2xl border border-neutral-200 p-5">
            <div className="text-xs font-bold uppercase tracking-wide text-neutral-500">Observed temporal event types</div>
            {active.eventTypes.length ? <div className="mt-3 flex flex-wrap gap-2">{active.eventTypes.map((event) => <span key={event} className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1.5 text-xs font-semibold text-neutral-700">{event}</span>)}</div> : <p className="mt-3 text-sm text-neutral-500">No review event crossed the configured temporal activation rules in this video.</p>}
            </div>
            <p className="mt-4 text-2xs leading-relaxed text-neutral-500">
            Candidate values are session-local ByteTrack IDs involved in review events, not biometric identity counts. {active.decodeFailures > 0 ? `${active.assetId} contains ${active.decodeFailures} red diagnostic decode-failure frames` : `${active.assetId} completed with no decode failures`}; source pixels were not fabricated or duplicated.
            </p>
          </ParallaxCard>
        </div>
      </div>

      <div className="rounded-3xl border border-neutral-200 bg-neutral-950 p-6 text-neutral-100">
        <div className="font-bold">Scientific boundary</div>
        <p className="mt-2 text-xs leading-relaxed text-neutral-300">The run proves source-to-output execution, temporal logging, privacy masking, candidate/PC context, evidence capture, and artifact integrity. It does not prove misconduct-classification accuracy. At the strict COCO cell-phone threshold, the run retained {VALIDATED_RUN.phoneStillsAtStrictThreshold} raw phone stills; possible-phone and possible-chit labels remain review proxies.</p>
        <div className="mt-3 text-2xs text-neutral-400">Runtime: {VALIDATED_RUN.device} · Torch {VALIDATED_RUN.torch} · Python {VALIDATED_RUN.python}</div>
      </div>
    </div>
  );
};
