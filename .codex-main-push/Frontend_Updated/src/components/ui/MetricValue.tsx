import React from 'react';
import { ParallaxCard } from './parallax-card';
import { cn } from '../../lib/utils';

interface MetricValueProps {
  value: string | number;
  label: string;
  sublabel?: string;
  icon?: React.ReactNode;
  trend?: string;
  className?: string;
}

export const MetricValue: React.FC<MetricValueProps> = ({
  value,
  label,
  sublabel,
  icon,
  trend,
  className,
}) => {
  return (
    <ParallaxCard tiltAngle={4} className={cn('p-5 border-neutral-200 group bg-white shadow-xs rounded-2xl', className)}>
      <div className="flex items-center justify-between text-xs uppercase font-semibold text-neutral-400 tracking-wider mb-2 font-sans">
        <span className="text-neutral-500 group-hover:text-neutral-900 transition-colors truncate">
          {label}
        </span>
        {icon && (
          <span className="text-neutral-400 group-hover:text-neutral-900 transition-all duration-150">
            {icon}
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-2.5 my-1">
        <span className="text-3xl font-bold tracking-tight text-neutral-950 font-sans">
          {value}
        </span>
        {trend && (
          <span className="text-xs font-medium text-neutral-700 px-2 py-0.5 rounded-full bg-neutral-100 border border-neutral-200">
            {trend}
          </span>
        )}
      </div>

      {sublabel && (
        <div className="text-xs text-neutral-500 truncate mt-1">
          {sublabel}
        </div>
      )}
    </ParallaxCard>
  );
};
