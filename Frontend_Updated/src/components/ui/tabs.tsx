import * as React from 'react';
import { motion } from 'motion/react';
import { cn, playHapticSound } from '../../lib/utils';

export interface TabItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  badge?: string | number;
}

export interface TabsProps {
  items: TabItem[];
  activeId: string;
  onChange: (id: string) => void;
  layoutId?: string;
  size?: 'sm' | 'default' | 'lg';
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({
  items,
  activeId,
  onChange,
  layoutId = 'activeTabCapsule',
  size = 'default',
  className,
}) => {
  const sizes = {
    sm: 'p-1 text-xs gap-1',
    default: 'p-1 text-sm gap-1',
    lg: 'p-1.5 text-base gap-1.5',
  };

  const itemSizes = {
    sm: 'px-3 py-1.5',
    default: 'px-4 py-2',
    lg: 'px-5 py-2.5',
  };

  return (
    <div
      className={cn(
        'inline-flex items-center rounded-xl bg-neutral-100 border border-neutral-200 shadow-2xs font-sans select-none',
        sizes[size],
        className
      )}
    >
      {items.map((item) => {
        const isActive = activeId === item.id;
        return (
          <motion.button
            key={item.id}
            onClick={() => {
              playHapticSound('click');
              onChange(item.id);
            }}
            whileTap={{ scale: 0.97 }}
            className={cn(
              'relative flex items-center gap-2 rounded-lg font-medium transition-colors duration-150 cursor-pointer',
              itemSizes[size],
              isActive
                ? 'text-neutral-950 font-semibold'
                : 'text-neutral-500 hover:text-neutral-900'
            )}
          >
            {isActive && (
              <motion.div
                layoutId={layoutId}
                transition={{ type: 'spring', damping: 28, stiffness: 380 }}
                className="absolute inset-0 bg-white border border-neutral-200/80 rounded-lg shadow-2xs"
              />
            )}
            <span className="relative z-10 flex items-center gap-2">
              {item.icon && <span className="shrink-0">{item.icon}</span>}
              <span>{item.label}</span>
              {item.badge !== undefined && (
                <span
                  className={cn(
                    'px-2 py-0.5 text-xs font-bold rounded-full',
                    isActive
                      ? 'bg-neutral-900 text-white'
                      : 'bg-neutral-200 text-neutral-700'
                  )}
                >
                  {item.badge}
                </span>
              )}
            </span>
          </motion.button>
        );
      })}
    </div>
  );
};
