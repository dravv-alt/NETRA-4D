import React, { useState, useEffect, useRef } from 'react';
import { animate, stagger } from 'animejs';
import { Volume2, Mic, Activity, ShieldCheck, AlertTriangle, Sparkles } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { StatusBadge } from '../ui/StatusBadge';

export const LiveAIAmbientSoundscapeMonitor: React.FC = () => {
  const [decibels, setDecibels] = useState<number>(34);
  const barsContainerRef = useRef<HTMLDivElement>(null);
  const soundBars = [14, 28, 42, 56, 38, 22, 18, 32, 64, 48, 30, 22, 36, 44, 26, 18, 12, 24];

  // Animate audio spectrum with animejs
  useEffect(() => {
    if (!barsContainerRef.current) return;

    const bars = barsContainerRef.current.querySelectorAll('.sound-bar-item');
    if (!bars || bars.length === 0) return;

    const anim = animate(bars, {
      scaleY: [0.3, 1.6],
      ease: 'easeInOutQuad',
      duration: 600,
      delay: stagger(40),
      loop: true,
      alternate: true,
    });

    const decibelInterval = setInterval(() => {
      setDecibels(Math.floor(30 + Math.random() * 12));
    }, 1200);

    return () => {
      anim.revert();
      clearInterval(decibelInterval);
    };
  }, []);

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80 flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700 border border-sky-100">
              <Volume2 size={18} />
            </div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Acoustic Soundscape & Whisper Anomaly Monitor
            </h3>
            <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-800 border border-sky-200">
              ANIME.JS SPECTRUM ENGINE
            </span>
          </div>
          <p className="text-xs text-neutral-500 mt-1">
            Directional acoustic decibel tracking, verbal peer exchange alerts, and paper rustling pattern classification.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-neutral-500">Hall Noise Level:</span>
          <span className="px-3.5 py-1 rounded-xl bg-neutral-900 text-white font-mono text-sm font-bold shadow-xs">
            {decibels} dB SPL
          </span>
        </div>
      </div>

      {/* Main Sound Waveform Visualizer Powered by Anime.js */}
      <div className="p-6 rounded-3xl bg-neutral-950 text-white border border-neutral-900 space-y-4 select-none">
        <div className="flex items-center justify-between text-2xs font-mono text-neutral-400">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>4-ARRAY DIRECTIONAL MICROPHONE SENSORS</span>
          </div>
          <span>Ambient Baseline: 32 dB (Nominal Silence)</span>
        </div>

        {/* Real-time Anime.js Staggered Waveform Bars */}
        <div
          ref={barsContainerRef}
          className="h-24 flex items-end gap-1.5 sm:gap-2 justify-between py-2 px-1"
        >
          {soundBars.map((baseHeight, idx) => (
            <div
              key={idx}
              style={{ height: `${baseHeight}px` }}
              className={`sound-bar-item flex-1 rounded-full origin-bottom transition-colors ${
                baseHeight > 40
                  ? 'bg-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.5)]'
                  : 'bg-sky-400/90'
              }`}
            />
          ))}
        </div>

        {/* Waveform Frequency Legend */}
        <div className="flex items-center justify-between text-3xs font-mono text-neutral-500 pt-2 border-t border-white/10">
          <span>60 Hz (HVAC Ambient)</span>
          <span>1.2 kHz (Whisper & Vocal Band)</span>
          <span>8 kHz (Keyboard / Mouse Clicks)</span>
        </div>
      </div>

      {/* Acoustic Anomaly Event Callout */}
      <div className="p-4 rounded-2xl bg-amber-50/70 border border-amber-200 flex items-center justify-between text-xs text-amber-900">
        <div className="flex items-center gap-2.5">
          <AlertTriangle size={16} className="text-amber-700 shrink-0" />
          <span className="font-medium">
            <strong>15:33:58 IST:</strong> Transient 48 dB acoustic burst correlated with Peer Paper Transfer at Seat B08 & B09.
          </span>
        </div>
        <span className="font-mono text-2xs font-bold bg-white px-2.5 py-1 rounded-lg border border-amber-200 shrink-0 ml-3">
          CONFIDENCE: 88%
        </span>
      </div>
    </div>
  );
};
