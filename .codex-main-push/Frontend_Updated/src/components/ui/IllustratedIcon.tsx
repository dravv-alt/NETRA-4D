import React from 'react';
import { LucideIcon } from 'lucide-react';

export type IllustratedVariant = 'sky' | 'emerald' | 'amber' | 'rose' | 'violet' | 'neutral' | 'graphite';

interface IllustratedIconProps {
  icon: LucideIcon;
  variant?: IllustratedVariant;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const IllustratedIcon: React.FC<IllustratedIconProps> = ({
  icon: Icon,
  variant = 'neutral',
  size = 'md',
  className = '',
}) => {
  const sizeMap = {
    sm: { box: 'w-8 h-8 rounded-xl p-1.5', icon: 14 },
    md: { box: 'w-10 h-10 rounded-2xl p-2.5', icon: 18 },
    lg: { box: 'w-12 h-12 rounded-2xl p-3', icon: 22 },
  };

  const variantStyles: Record<IllustratedVariant, string> = {
    sky: 'bg-sky-50 text-sky-700 border border-sky-200/90 shadow-[0_2px_8px_rgba(2,132,199,0.08)]',
    emerald: 'bg-emerald-50 text-emerald-800 border border-emerald-200/90 shadow-[0_2px_8px_rgba(16,185,129,0.08)]',
    amber: 'bg-amber-50 text-amber-800 border border-amber-200/90 shadow-[0_2px_8px_rgba(245,158,11,0.08)]',
    rose: 'bg-rose-50 text-rose-800 border border-rose-200/90 shadow-[0_2px_8px_rgba(244,63,94,0.08)]',
    violet: 'bg-violet-50 text-violet-800 border border-violet-200/90 shadow-[0_2px_8px_rgba(139,92,246,0.08)]',
    neutral: 'bg-neutral-100 text-neutral-800 border border-neutral-200/90 shadow-[0_2px_8px_rgba(0,0,0,0.03)]',
    graphite: 'bg-[#181716] text-white border border-white/10 shadow-[0_4px_12px_rgba(0,0,0,0.15)]',
  };

  const currentSize = sizeMap[size];

  return (
    <div
      className={`inline-flex items-center justify-center shrink-0 transition-transform ${currentSize.box} ${variantStyles[variant]} ${className}`}
    >
      <Icon size={currentSize.icon} strokeWidth={2} />
    </div>
  );
};
