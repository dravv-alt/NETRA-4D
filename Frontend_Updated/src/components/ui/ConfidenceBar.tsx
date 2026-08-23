import React from 'react';
import { motion } from 'motion/react';

interface ConfidenceBarProps {
  value: number; // 0.0 - 1.0
  label?: string;
  showPercent?: boolean;
  color?: 'amber' | 'emerald' | 'red' | 'sky' | 'slate' | 'lavender' | 'rose' | 'default';
  size?: 'sm' | 'md';
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({
  value,
  label,
  showPercent = true,
  color = 'default',
  size = 'md',
}) => {
  const clamped = Math.max(0, Math.min(1, value));
  const percent = Math.round(clamped * 100);

  const heightClass = size === 'sm' ? 'h-1.5' : 'h-2';

  const colorMap: Record<string, string> = {
    amber: 'bg-amber-400',
    emerald: 'bg-emerald-400',
    red: 'bg-rose-400',
    rose: 'bg-rose-400',
    sky: 'bg-sky-400',
    lavender: 'bg-violet-400',
    slate: 'bg-neutral-500',
    default: 'bg-neutral-900',
  };

  const fillClass = colorMap[color] || (
    clamped >= 0.8
      ? 'bg-rose-400'
      : clamped >= 0.65
      ? 'bg-amber-400'
      : 'bg-emerald-400'
  );

  return (
    <div className="w-full font-sans">
      {(label || showPercent) && (
        <div className="flex items-center justify-between text-xs text-neutral-600 mb-1.5 font-normal">
          {label && <span className="truncate">{label}</span>}
          {showPercent && (
            <span className="text-neutral-900 font-semibold tabular-nums ml-auto">
              {clamped.toFixed(2)} <span className="text-neutral-400 font-normal">({percent}%)</span>
            </span>
          )}
        </div>
      )}
      <div className={`w-full bg-neutral-100 rounded-full overflow-hidden border border-neutral-200 ${heightClass}`}>
        <motion.div
          className={`h-full rounded-full ${fillClass}`}
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ type: 'spring', damping: 28, stiffness: 220 }}
        />
      </div>
    </div>
  );
};
