import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Terminal, Search, Sparkles, ArrowRight, CornerDownLeft, Play, User } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';
import { StatusBadge } from '../ui/StatusBadge';

interface SurveillanceAIQueryConsoleProps {
  onSelectCandidate?: (seatId: string) => void;
}

export const SurveillanceAIQueryConsole: React.FC<SurveillanceAIQueryConsoleProps> = ({
  onSelectCandidate,
}) => {
  const [query, setQuery] = useState('');
  const [activeQuery, setActiveQuery] = useState<string | null>(null);

  const quickPrompts = [
    'Show candidates with off-laptop hands >5s',
    'List gaze deviations >30° yaw in Row A',
    'Display mutual peer lookups in Row B',
    'Show suppressed wall-timer lookups',
  ];

  const handleRunQuery = (text: string) => {
    playHapticSound('snap');
    setActiveQuery(text);
  };

  return (
    <div className="p-8 rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-neutral-200/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-amber-50 text-amber-700 border border-amber-100">
            <Sparkles size={18} />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
              Natural Language Surveillance AI Query Console
            </h3>
            <p className="text-xs text-neutral-500 mt-0.5">
              Execute real-time semantic queries across the multi-camera kinematic temporal stream.
            </p>
          </div>
        </div>
        <span className="text-3xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-neutral-900 text-white">
          LLM-VISION EMBEDDINGS
        </span>
      </div>

      {/* Query Input Bar */}
      <div className="relative">
        <Terminal size={16} className="absolute left-4 top-3.5 text-neutral-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && query && handleRunQuery(query)}
          placeholder="Ask AI: e.g. 'Show candidates who looked down at lap for >5s'..."
          className="w-full bg-neutral-50 border border-neutral-300 rounded-2xl pl-11 pr-24 py-3 text-xs text-neutral-900 placeholder-neutral-400 font-mono focus:outline-none focus:border-neutral-900 focus:bg-white transition-all shadow-2xs"
        />
        <Button
          variant="default"
          size="xs"
          onClick={() => query && handleRunQuery(query)}
          className="absolute right-2.5 top-2 rounded-xl text-2xs font-mono font-bold"
        >
          <span>EXECUTE</span>
          <CornerDownLeft size={11} />
        </Button>
      </div>

      {/* Quick Prompts */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-2xs font-mono text-neutral-400 font-bold">Suggested:</span>
        {quickPrompts.map((prompt, idx) => (
          <button
            key={idx}
            onClick={() => {
              setQuery(prompt);
              handleRunQuery(prompt);
            }}
            className="px-3 py-1.5 rounded-xl bg-neutral-100 hover:bg-neutral-200 text-neutral-700 font-sans text-2xs font-semibold transition-colors cursor-pointer"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Query Result Card */}
      {activeQuery && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-5 rounded-2xl bg-neutral-900 text-white space-y-4 font-mono text-xs"
        >
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-emerald-300 font-bold">QUERY EXECUTED: "{activeQuery}"</span>
            </div>
            <span className="text-2xs text-neutral-400">2 Matches Identified</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* Match 1 */}
            <div className="p-3.5 rounded-xl bg-white/5 border border-white/10 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white">Terminal A06 (Candidate #6)</span>
                <StatusBadge status="review" size="sm" />
              </div>
              <p className="text-2xs text-neutral-300 font-sans">
                Hands below desk plane for 6.2s with concurrent 34.2° downward gaze vector.
              </p>
              <Button
                variant="outline"
                size="xs"
                onClick={() => {
                  playHapticSound('snap');
                  onSelectCandidate?.('A06');
                }}
                className="w-full bg-white/10 hover:bg-white text-white hover:text-neutral-950 border-white/20 text-2xs font-bold rounded-lg mt-1"
              >
                <span>Jump to Candidate Dossier</span>
                <ArrowRight size={11} />
              </Button>
            </div>

            {/* Match 2 */}
            <div className="p-3.5 rounded-xl bg-white/5 border border-white/10 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white">Terminal B04 (Candidate #14)</span>
                <StatusBadge status="review" size="sm" />
              </div>
              <p className="text-2xs text-neutral-300 font-sans">
                Repeated turn back toward central aisle (+38° yaw deflection over 3 bursts).
              </p>
              <Button
                variant="outline"
                size="xs"
                onClick={() => {
                  playHapticSound('snap');
                  onSelectCandidate?.('B04');
                }}
                className="w-full bg-white/10 hover:bg-white text-white hover:text-neutral-950 border-white/20 text-2xs font-bold rounded-lg mt-1"
              >
                <span>Jump to Candidate Dossier</span>
                <ArrowRight size={11} />
              </Button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};
