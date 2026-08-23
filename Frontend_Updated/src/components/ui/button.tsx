import * as React from 'react';
import { motion, HTMLMotionProps } from 'motion/react';
import { cn, playHapticSound } from '../../lib/utils';

export interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  variant?: 'default' | 'glass' | 'secondary' | 'destructive' | 'outline' | 'ghost' | 'glow' | 'pill';
  size?: 'default' | 'sm' | 'lg' | 'icon' | 'xs';
  children?: React.ReactNode;
  haptic?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', haptic = true, onClick, children, ...props }, ref) => {
    const baseStyles =
      'relative inline-flex items-center justify-center font-sans font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 disabled:pointer-events-none disabled:opacity-40 select-none cursor-pointer';

    const variants = {
      default:
        'bg-neutral-900 hover:bg-black text-white font-medium rounded-xl border border-neutral-900 shadow-sm',
      glass:
        'bg-white border border-neutral-200 text-neutral-900 hover:bg-neutral-50 rounded-xl shadow-xs',
      secondary:
        'bg-neutral-100 hover:bg-neutral-200 text-neutral-900 border border-neutral-200 rounded-xl',
      destructive:
        'bg-neutral-900 hover:bg-black text-white border border-neutral-900 rounded-xl',
      outline:
        'border border-neutral-300 bg-white hover:bg-neutral-50 text-neutral-900 rounded-xl',
      ghost:
        'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 rounded-xl',
      glow:
        'bg-neutral-900 hover:bg-black text-white rounded-xl font-medium border border-neutral-900 shadow-sm',
      pill:
        'bg-white border border-neutral-300 rounded-full text-neutral-900 hover:bg-neutral-50 px-4 shadow-xs',
    };

    const sizes = {
      default: 'h-10 px-4 py-2 text-sm',
      xs: 'h-7 px-3 text-xs rounded-lg',
      sm: 'h-8 px-3.5 text-xs rounded-lg',
      lg: 'h-12 px-6 text-base rounded-2xl',
      icon: 'h-9 w-9 p-0 rounded-xl',
    };

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (haptic) playHapticSound('click');
      onClick?.(e);
    };

    return (
      <motion.button
        ref={ref}
        whileTap={{ scale: 0.97 }}
        transition={{ type: 'spring', damping: 25, stiffness: 450 }}
        onClick={handleClick}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        {...props}
      >
        {children}
      </motion.button>
    );
  }
);

Button.displayName = 'Button';
