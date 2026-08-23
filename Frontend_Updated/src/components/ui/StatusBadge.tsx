import React from 'react';
import { EventSeverity, OperationalState } from '../../types';

interface StatusBadgeProps {
  status: EventSeverity | OperationalState | 'HEALTHY' | 'DEGRADED' | 'AVAILABLE' | 'ON_DEMAND' | 'OFFLINE' | 'CRITICAL' | 'ELEVATED' | 'ROUTINE' | 'ONLINE' | 'OPTIMAL';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showDot?: boolean;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  size = 'md',
  showDot = true,
  className = '',
}) => {
  const normalized = String(status).toLowerCase();
  
  let bg = 'bg-neutral-100 border-neutral-200 text-neutral-800';
  let dotColor = 'bg-neutral-500';
  let displayLabel = label || String(status).toUpperCase();

  if (normalized === 'escalated' || normalized === 'critical') {
    bg = 'bg-rose-50/90 border-rose-200 text-rose-800 font-semibold';
    dotColor = 'bg-rose-500';
    displayLabel = label || 'ESCALATED';
  } else if (normalized === 'review' || normalized === 'elevated' || normalized === 'degraded' || normalized === 'warning') {
    bg = 'bg-amber-50/90 border-amber-200 text-amber-800 font-semibold';
    dotColor = 'bg-amber-500';
    displayLabel = label || (normalized === 'degraded' ? 'DEGRADED' : 'NEEDS REVIEW');
  } else if (normalized === 'observe' || normalized === 'routine' || normalized === 'observed') {
    bg = 'bg-violet-50/90 border-violet-200 text-violet-800 font-medium';
    dotColor = 'bg-violet-500';
    displayLabel = label || 'OBSERVE';
  } else if (normalized === 'normal' || normalized === 'healthy' || normalized === 'optimal' || normalized === 'online') {
    bg = 'bg-emerald-50/90 border-emerald-200 text-emerald-800 font-semibold';
    dotColor = 'bg-emerald-500';
    displayLabel = label || (normalized === 'healthy' ? 'HEALTHY' : normalized === 'online' ? 'ONLINE' : 'NORMAL');
  } else if (normalized === 'available' || normalized === 'on_demand') {
    bg = 'bg-sky-50/90 border-sky-200 text-sky-800 font-medium';
    dotColor = 'bg-sky-500';
    displayLabel = label || (normalized === 'on_demand' ? 'ON-DEMAND' : 'AVAILABLE');
  }

  const sizeClasses = {
    sm: 'text-3xs px-2.5 py-0.5 gap-1.5 rounded-full font-medium shadow-2xs',
    md: 'text-2xs px-3 py-0.5 gap-1.5 rounded-full font-medium shadow-2xs',
    lg: 'text-xs px-4 py-1 gap-2 rounded-full font-medium shadow-2xs',
  }[size];

  return (
    <span className={`inline-flex items-center whitespace-nowrap border font-sans select-none shrink-0 ${sizeClasses} ${bg} ${className}`}>
      {showDot && <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${dotColor}`} />}
      <span className="leading-none whitespace-nowrap">{displayLabel}</span>
    </span>
  );
};
