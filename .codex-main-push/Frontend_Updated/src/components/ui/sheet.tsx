import * as React from 'react';
import { motion, AnimatePresence, PanInfo } from 'motion/react';
import { X } from 'lucide-react';
import { cn, playHapticSound, projectMomentum } from '../../lib/utils';

export interface SheetProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  description?: string;
  children: React.ReactNode;
  side?: 'bottom' | 'right';
  className?: string;
}

export const Sheet: React.FC<SheetProps> = ({
  isOpen,
  onClose,
  title,
  description,
  children,
  side = 'bottom',
  className,
}) => {
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    const projectedOffset = info.offset.y + projectMomentum(info.velocity.y);
    if (projectedOffset > 120 || info.velocity.y > 400) {
      playHapticSound('warn');
      onClose();
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-end justify-center">
          {/* Dimmed Translucent Scrim */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-900/30 backdrop-blur-sm"
          />

          {/* Apple Modal Sheet (Light Mode) */}
          <motion.div
            initial={side === 'bottom' ? { y: '100%', opacity: 0.5 } : { x: '100%', opacity: 0.5 }}
            animate={side === 'bottom' ? { y: 0, opacity: 1 } : { x: 0, opacity: 1 }}
            exit={side === 'bottom' ? { y: '100%', opacity: 0.5 } : { x: '100%', opacity: 0.5 }}
            transition={{ type: 'spring', damping: 28, stiffness: 320 }}
            drag={side === 'bottom' ? 'y' : false}
            dragConstraints={{ top: 0 }}
            dragElastic={{ top: 0.05, bottom: 0.8 }}
            onDragEnd={handleDragEnd}
            className={cn(
              'relative z-50 w-full apple-glass-elevated border border-black/[0.08] overflow-hidden flex flex-col font-mono text-xs bg-white/95 text-slate-900',
              side === 'bottom'
                ? 'max-w-4xl max-h-[85vh] rounded-t-3xl shadow-2xl mb-0'
                : 'max-w-md h-full rounded-l-3xl shadow-2xl ml-auto',
              className
            )}
          >
            {/* Grab Handle for Bottom Sheet */}
            {side === 'bottom' && (
              <div className="flex justify-center pt-3 pb-1 cursor-grab active:cursor-grabbing">
                <div className="w-12 h-1.2 rounded-full bg-slate-300 hover:bg-slate-400 transition-colors" />
              </div>
            )}

            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-black/[0.06]">
              <div>
                {title && <h2 className="font-bold text-sm text-telemetry-bright tracking-tight">{title}</h2>}
                {description && <p className="text-2xs text-telemetry-dim mt-0.5">{description}</p>}
              </div>
              <button
                onClick={() => {
                  playHapticSound('click');
                  onClose();
                }}
                className="p-1.5 rounded-full hover:bg-black/5 text-telemetry-dim hover:text-telemetry-bright transition-colors cursor-pointer"
              >
                <X size={16} />
              </button>
            </div>

            {/* Content */}
            <div className="p-4 overflow-y-auto flex-1">{children}</div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
