import React, { useState } from 'react';
import { motion } from 'motion/react';
import { PROVIDER_BENCHMARKS, PER_VIDEO_BENCHMARKS } from '../../data/benchmarkData';
import { StatusBadge } from '../ui/StatusBadge';
import { FlaskConical, Gauge, Activity, Video } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { ParallaxCard } from '../ui/parallax-card';

export const BenchmarkLabView: React.FC = () => {
  const [selectedVideo, setSelectedVideo] = useState<string>(PER_VIDEO_BENCHMARKS[2].videoId);
  const activeVideo = PER_VIDEO_BENCHMARKS.find((v) => v.videoId === selectedVideo) || PER_VIDEO_BENCHMARKS[0];

  const pipelineStages = [
    { name: 'Video Decode (NVDEC)', latency: '4.2 ms', status: 'HEALTHY' },
    { name: 'Person Detection (YOLOv8x)', latency: '12.8 ms', status: 'HEALTHY' },
    { name: 'Multi-Object Tracking (ByteTrack)', latency: '2.1 ms', status: 'HEALTHY' },
    { name: '2D Pose Estimation (YOLO/RTM)', latency: '8.4 ms', status: 'HEALTHY' },
    { name: 'Object & Phone Classifier', latency: '4.6 ms', status: 'HEALTHY' },
    { name: 'Spatial Context & Seat Map', latency: '0.8 ms', status: 'HEALTHY' },
    { name: 'Temporal Rolling Buffer (12s)', latency: '1.2 ms', status: 'HEALTHY' },
    { name: 'Multi-Modal Fusion & Triage', latency: '0.6 ms', status: 'HEALTHY' },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto font-sans text-sm text-neutral-900">
      {/* Top Banner */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
        <div>
          <div className="font-bold text-lg text-neutral-950 flex items-center gap-3">
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
              <FlaskConical size={18} />
            </div>
            <span>Multi-Provider Model Benchmark Lab</span>
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            Empirical evaluation across 8 standardized CCTV clips with real-world ground truth comparison.
          </div>
        </div>
        <div className="text-xs text-neutral-500 text-right">
          <div>Dataset: <strong className="text-neutral-900">8 CCTV Benchmarks</strong></div>
          <div>Hardware: <strong className="text-neutral-900">NVIDIA T4 Tensor Core</strong></div>
        </div>
      </div>

      {/* 1. MULTI-PROVIDER COMPARISON MATRIX TABLE */}
      <div className="bg-white rounded-3xl border border-neutral-200 overflow-hidden shadow-xs">
        <div className="p-6 bg-neutral-50/80 border-b border-neutral-200 flex items-center justify-between">
          <div className="font-bold text-base text-neutral-950 flex items-center gap-2.5">
            <Gauge size={16} className="text-neutral-900" />
            <span>Architecture & Provider Performance Matrix</span>
          </div>
          <span className="text-xs text-neutral-500 font-medium">Single NVIDIA T4 GPU (16GB VRAM)</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-neutral-50/50 text-neutral-500 uppercase border-b border-neutral-200 font-semibold text-2xs tracking-wider">
              <tr>
                <th className="py-3.5 px-6">Provider Architecture</th>
                <th className="py-3.5 px-6">Pipeline Role</th>
                <th className="py-3.5 px-6">Test Coverage</th>
                <th className="py-3.5 px-6">Event F1</th>
                <th className="py-3.5 px-6">FP / Min</th>
                <th className="py-3.5 px-6">Inference FPS</th>
                <th className="py-3.5 px-6">VRAM Load</th>
                <th className="py-3.5 px-6">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200/80">
              {PROVIDER_BENCHMARKS.map((row, idx) => (
                <tr
                  key={idx}
                  className={`hover:bg-neutral-50 transition-colors ${
                    row.providerName.includes('Fusion') ? 'bg-neutral-50/90 font-bold text-neutral-950' : ''
                  }`}
                >
                  <td className="py-4 px-6 font-bold text-neutral-950">
                    {row.providerName}
                  </td>
                  <td className="py-4 px-6 text-neutral-600 font-medium">{row.role}</td>
                  <td className="py-4 px-6 text-neutral-500 font-mono tabular-nums">{row.videosTested}</td>
                  <td className="py-4 px-6 text-neutral-950 font-bold font-mono tabular-nums">{row.eventF1}</td>
                  <td className="py-4 px-6 text-neutral-900 font-mono tabular-nums">{row.fpPerMin}</td>
                  <td className="py-4 px-6 text-emerald-700 font-bold font-mono tabular-nums">{row.fps}</td>
                  <td className="py-4 px-6 text-neutral-500 font-mono tabular-nums">{row.vramGb}</td>
                  <td className="py-4 px-6">
                    <StatusBadge
                      status={row.status === 'PRIMARY' ? 'HEALTHY' : row.status === 'REFINEMENT' ? 'AVAILABLE' : 'ON_DEMAND'}
                      label={row.status}
                      size="sm"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. PER-VIDEO DETAILED BENCHMARK INSPECTOR */}
      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Left 4 Cols: Video Test Cases Selector */}
        <div className="col-span-12 lg:col-span-4 space-y-3">
          <div className="text-2xs uppercase text-neutral-400 font-bold tracking-wider px-1">
            Standardized CCTV Clips (8 Benchmarks)
          </div>
          <div className="space-y-2">
            {PER_VIDEO_BENCHMARKS.map((vid) => {
              const isSelected = selectedVideo === vid.videoId;
              return (
                <button
                  key={vid.videoId}
                  onClick={() => {
                    playHapticSound('click');
                    setSelectedVideo(vid.videoId);
                  }}
                  className={`w-full p-4 rounded-2xl text-left transition-all border cursor-pointer ${
                    isSelected
                      ? 'bg-neutral-900 text-white border-neutral-900 shadow-md font-semibold'
                      : 'bg-white text-neutral-600 hover:text-neutral-950 hover:bg-neutral-50 border-neutral-200 shadow-2xs'
                  }`}
                >
                  <div className="flex items-center justify-between text-2xs mb-1">
                    <span className="font-bold">{vid.videoId}</span>
                    <span className={`font-mono tabular-nums ${isSelected ? 'text-neutral-300' : 'text-neutral-400'}`}>{vid.fps} FPS • {vid.durationSec}s</span>
                  </div>
                  <div className={`text-sm truncate font-bold ${isSelected ? 'text-white' : 'text-neutral-950'}`}>
                    {vid.filename}
                  </div>
                  <div className={`text-2xs truncate mt-1 ${isSelected ? 'text-neutral-300' : 'text-neutral-500'}`}>
                    {vid.notes}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right 8 Cols: Active Video Metric Deep-Dive */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <ParallaxCard tiltAngle={2} className="p-8 space-y-6 bg-white border border-neutral-200 shadow-xs rounded-3xl">
            {/* Header of Active Video */}
            <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
              <div>
                <span className="font-bold text-base text-neutral-950">{activeVideo.videoId}: {activeVideo.filename}</span>
                <div className="text-xs text-neutral-500 mt-1">
                  Resolution: {activeVideo.resolution} • Base FPS: {activeVideo.fps} • Duration: {activeVideo.durationSec}s
                </div>
              </div>
              <span className="px-3.5 py-1.5 rounded-full bg-neutral-100 text-neutral-900 text-xs font-bold border border-neutral-200">
                {activeVideo.groundTruthEvents}
              </span>
            </div>

            {/* Provider Breakdown Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-neutral-50 p-4 rounded-2xl border border-neutral-200 space-y-1.5">
                <div className="font-bold text-xs text-neutral-800 uppercase">YOLO-Pose</div>
                <div className="text-sm text-neutral-950 font-mono font-bold">Frames: {activeVideo.yoloMetrics.frames}</div>
                <div className="text-2xs text-neutral-500 font-mono">Tracks: {activeVideo.yoloMetrics.tracks} | VRAM: {activeVideo.yoloMetrics.vram}</div>
                <div className="text-xs text-emerald-700 font-bold font-mono">{activeVideo.yoloMetrics.fps} FPS Avg</div>
              </div>

              <div className="bg-neutral-50 p-4 rounded-2xl border border-neutral-200 space-y-1.5">
                <div className="font-bold text-xs text-neutral-800 uppercase">RTMPose Refine</div>
                <div className="text-sm text-neutral-950 font-mono font-bold">Frames: {activeVideo.rtmposeMetrics.frames}</div>
                <div className="text-2xs text-neutral-500 font-mono">Tracks: {activeVideo.rtmposeMetrics.tracks} | VRAM: {activeVideo.rtmposeMetrics.vram}</div>
                <div className="text-xs text-emerald-700 font-bold font-mono">{activeVideo.rtmposeMetrics.fps} FPS Avg</div>
              </div>

              <div className="bg-neutral-50 p-4 rounded-2xl border border-neutral-200 space-y-1.5">
                <div className="font-bold text-xs text-neutral-800 uppercase">Multi-HMR 3D</div>
                <div className="text-sm text-neutral-950 font-mono font-bold">Frames: {activeVideo.multiHmrMetrics.frames}</div>
                <div className="text-2xs text-neutral-500 font-mono">Tracks: {activeVideo.multiHmrMetrics.tracks} | VRAM: {activeVideo.multiHmrMetrics.vram}</div>
                <div className="text-xs text-neutral-900 font-bold font-mono">{activeVideo.multiHmrMetrics.fps > 0 ? `${activeVideo.multiHmrMetrics.fps} FPS` : 'On-Demand'}</div>
              </div>
            </div>

            {/* Event Accuracy & Overlap Card */}
            <div className="bg-neutral-50 p-5 rounded-2xl space-y-3 border border-neutral-200">
              <div className="font-bold text-xs text-neutral-950 uppercase tracking-wider flex items-center justify-between">
                <span>Evaluation Results & Ground Truth Match</span>
                <span className="text-xs text-neutral-900 font-bold px-3 py-1 rounded-full bg-white border border-neutral-300">
                  {activeVideo.overlap}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs pt-3 border-t border-neutral-200">
                <div>
                  <span className="text-neutral-500">Predicted: </span>
                  <strong className="text-neutral-950 font-bold font-mono">{activeVideo.predictedEvents}</strong>
                </div>
                <div>
                  <span className="text-neutral-500">Ground Truth: </span>
                  <strong className="text-neutral-950 font-bold font-mono">{activeVideo.groundTruthEvents}</strong>
                </div>
                <div>
                  <span className="text-neutral-500">False Alerts: </span>
                  <strong className="text-neutral-950 font-bold font-mono">{activeVideo.falseAlerts}</strong>
                </div>
              </div>
            </div>
          </ParallaxCard>
        </div>
      </div>

      {/* 3. PIPELINE LATENCY DAG */}
      <div className="bg-white p-8 space-y-5 rounded-3xl border border-neutral-200 shadow-xs">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-200">
          <div className="font-bold text-base text-neutral-950 flex items-center gap-2.5">
            <Activity size={16} className="text-neutral-900" />
            <span>End-to-End Pipeline DAG Latency Profile</span>
          </div>
          <span className="text-xs text-emerald-800 font-bold px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200">
            Total Latency: 34.7 ms
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          {pipelineStages.map((stage, idx) => (
            <div key={idx} className="bg-neutral-50 p-4 rounded-2xl border border-neutral-200 space-y-1.5 shadow-2xs">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-neutral-400 font-mono">0{idx + 1}</span>
                <StatusBadge status={stage.status as any} size="sm" />
              </div>
              <div className="text-xs font-bold text-neutral-950 truncate">{stage.name}</div>
              <div className="text-xs text-neutral-500 font-mono tabular-nums pt-2 border-t border-neutral-200 flex items-center justify-between">
                <span>Latency:</span>
                <strong className="text-neutral-950 font-bold">{stage.latency}</strong>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
