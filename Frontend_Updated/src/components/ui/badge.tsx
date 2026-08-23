import * as React from 'react';
import { cn } from '../../lib/utils';

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'destructive' | 'outline' | 'normal' | 'observe' | 'review' | 'escalated' | 'live' | 'tech' | 'glass';
  size?: 'sm' | 'default' | 'lg';
  dot?: boolean;
}

export function Badge({
  className,
  variant = 'default',
  size = 'default',
  dot = false,
  children,
  ...props
}: BadgeProps) {
  const baseStyles =
    'inline-flex items-center gap-1.5 font-mono font-bold rounded-full border transition-colors select-none tabular-nums';

  const variants = {
    default:
      'bg-neutral-900 text-white border-neutral-900',
    secondary:
      'bg-neutral-100 text-neutral-800 border-neutral-200',
    destructive:
      'bg-neutral-900 text-white border-neutral-900',
    outline:
      'border-neutral-300 text-neutral-900 bg-white',
    normal:
      'bg-neutral-100 text-neutral-800 border-neutral-200',
    observe:
      'bg-neutral-100 text-neutral-600 border-neutral-200',
    review:
      'bg-neutral-900 text-white border-neutral-900',
    escalated:
      'bg-black text-white border-black font-extrabold',
    live:
      'bg-neutral-900 text-white border-neutral-900',
    tech:
      'bg-neutral-100 text-neutral-800 border-neutral-200',
    glass:
      'bg-white text-neutral-900 border-neutral-300',
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-2xs',
    default: 'px-2.5 py-0.5 text-2xs',
    lg: 'px-3 py-1 text-xs',
  };

  return (
    <div className={cn(baseStyles, variants[variant], sizes[size], className)} {...props}>
      {dot && (
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full shrink-0',
            variant === 'escalated' || variant === 'review' || variant === 'default' || variant === 'live'
              ? 'bg-white'
              : 'bg-neutral-900'
          )}
        />
      )}
      {children}
    </div>
  );
}
