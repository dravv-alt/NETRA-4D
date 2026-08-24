import * as React from 'react';
import { motion } from 'motion/react';
import { cn, playHapticSound } from '../../lib/utils';

export interface SwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
  size?: 'sm' | 'default';
}

export const Switch: React.FC<SwitchProps> = ({
  checked,
  onCheckedChange,
  disabled = false,
  className,
  size = 'default',
}) => {
  const isSm = size === 'sm';

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (disabled) return;
    playHapticSound('snap');
    onCheckedChange(!checked);
  };

  return (
    <motion.button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={handleClick}
      whileTap={{ scale: 0.94 }}
      className={cn(
        'relative inline-flex shrink-0 cursor-pointer rounded-full transition-colors duration-150 ease-in-out focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-900 disabled:cursor-not-allowed disabled:opacity-40',
        isSm ? 'h-4 w-7' : 'h-6 w-11',
        checked ? 'bg-neutral-900' : 'bg-neutral-200 border border-neutral-300',
        className
      )}
    >
      <motion.span
        layout
        transition={{ type: 'spring', damping: 25, stiffness: 400 }}
        className={cn(
          'pointer-events-none block rounded-full bg-white shadow-xs ring-0 transition-transform',
          isSm ? 'h-3 w-3' : 'h-4.5 w-4.5',
          checked
            ? isSm
              ? 'translate-x-3.5 mt-0.5 ml-0.5'
              : 'translate-x-5.5 mt-0.75 ml-0.75'
            : isSm
            ? 'translate-x-0.5 mt-0.5'
            : 'translate-x-0.75 mt-0.75'
        )}
      />
    </motion.button>
  );
};
