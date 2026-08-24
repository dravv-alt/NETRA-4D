import * as React from 'react';
import { motion, useMotionValue, useSpring, useTransform, AnimatePresence, type MotionValue } from 'motion/react';
import { cn, playHapticSound } from '../../lib/utils';

export interface DockItemData {
  id: string;
  label: string;
  icon: React.ReactNode;
  onClick?: () => void;
  isActive?: boolean;
  badge?: string | number;
}

export interface MagneticDockProps {
  items: DockItemData[];
  iconSize?: number;
  maxScale?: number;
  magneticDistance?: number;
  className?: string;
}

interface DockItemProps {
  item: DockItemData;
  mouseX: MotionValue<number>;
  iconSize: number;
  maxScale: number;
  magneticDistance: number;
}

function DockItem({ item, mouseX, iconSize, maxScale, magneticDistance }: DockItemProps) {
  const ref = React.useRef<HTMLButtonElement>(null);
  const [isHovered, setIsHovered] = React.useState(false);

  const distance = useTransform(mouseX, (val: number) => {
    if (!ref.current) return magneticDistance + 1;
    const rect = ref.current.getBoundingClientRect();
    const center = rect.left + rect.width / 2;
    return val - center;
  });

  const scale = useTransform(distance, [-magneticDistance, 0, magneticDistance], [1, maxScale, 1]);
  const springConfig = { damping: 18, stiffness: 320, mass: 0.4 };
  const smoothScale = useSpring(scale, springConfig);

  const size = useTransform(smoothScale, (s) => s * iconSize);
  const y = useTransform(smoothScale, (s) => (s - 1) * -10);
  const smoothY = useSpring(y, springConfig);

  const handleClick = () => {
    playHapticSound('snap');
    item.onClick?.();
  };

  return (
    <div className="relative flex flex-col items-center">
      {/* Tooltip on hover */}
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.94 }}
            animate={{ opacity: 1, y: -6, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.94 }}
            transition={{ type: 'spring', damping: 25, stiffness: 400 }}
            className="absolute bottom-full mb-1 px-3 py-1 bg-neutral-900 rounded-lg text-xs font-sans font-medium text-white whitespace-nowrap shadow-md pointer-events-none z-50"
          >
            {item.label}
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        ref={ref}
        onClick={handleClick}
        onMouseEnter={() => {
          setIsHovered(true);
          playHapticSound('click');
        }}
        onMouseLeave={() => setIsHovered(false)}
        className={cn(
          'relative flex items-center justify-center rounded-2xl transition-colors cursor-pointer select-none',
          'focus:outline-none focus-visible:ring-1 focus-visible:ring-neutral-900',
          item.isActive
            ? 'text-white'
            : 'text-neutral-600 hover:text-neutral-950'
        )}
        style={{
          width: size,
          height: size,
          y: smoothY,
        }}
        whileTap={{ scale: 0.92 }}
      >
        {/* Minimalist Icon Container */}
        <div
          className={cn(
            'w-full h-full rounded-2xl flex items-center justify-center p-2.5 border transition-all duration-150',
            item.isActive
              ? 'bg-neutral-900 border-neutral-900 text-white shadow-sm'
              : 'bg-white border-neutral-200 text-neutral-700 hover:border-neutral-400 hover:bg-neutral-50 shadow-2xs'
          )}
        >
          {item.icon}
        </div>

        {/* Badge counter */}
        {item.badge !== undefined && (
          <span className="absolute -top-1 -right-1 px-1.5 py-0.2 min-w-4 text-center rounded-full bg-neutral-900 text-white font-sans font-bold text-3xs border border-white shadow-sm">
            {item.badge}
          </span>
        )}

        {/* Active Indicator Dot */}
        {item.isActive && (
          <motion.div
            layoutId="dockActiveDot"
            className="absolute -bottom-1.5 w-1.5 h-1.5 rounded-full bg-neutral-900"
          />
        )}
      </motion.button>
    </div>
  );
}

export function MagneticDock({
  items,
  iconSize = 40,
  maxScale = 1.35,
  magneticDistance = 140,
  className,
}: MagneticDockProps) {
  const mouseX = useMotionValue(Infinity);

  return (
    <motion.div
      onMouseMove={(e) => mouseX.set(e.pageX)}
      onMouseLeave={() => mouseX.set(Infinity)}
      className={cn(
        'inline-flex items-end gap-2.5 px-4 py-3 rounded-3xl bg-white/95 border border-neutral-200 shadow-xl backdrop-blur-md',
        className
      )}
    >
      {items.map((item) => (
        <DockItem
          key={item.id}
          item={item}
          mouseX={mouseX}
          iconSize={iconSize}
          maxScale={maxScale}
          magneticDistance={magneticDistance}
        />
      ))}
    </motion.div>
  );
}
