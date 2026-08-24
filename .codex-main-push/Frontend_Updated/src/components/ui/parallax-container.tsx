import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'motion/react';
import { cn } from '../../lib/utils';

interface ParallaxContainerProps {
  children: React.ReactNode;
  className?: string;
  depth?: number;
}

export const ParallaxContainer: React.FC<ParallaxContainerProps> = ({
  children,
  className,
  depth = 40,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end end'],
  });

  const yBackground = useTransform(scrollYProgress, [0, 1], [0, -depth * 1.5]);
  const yForeground = useTransform(scrollYProgress, [0, 1], [0, -depth * 0.3]);
  const opacityFade = useTransform(scrollYProgress, [0, 0.7, 1], [1, 0.95, 0.9]);

  return (
    <div ref={containerRef} className={cn('relative min-h-full w-full', className)}>
      {/* Background Ambient Mesh Parallax Layer */}
      <motion.div
        style={{ y: yBackground, opacity: opacityFade }}
        className="pointer-events-none absolute -inset-10 bg-mesh-radial z-0 opacity-60"
      />

      {/* Foreground Content Layer */}
      <motion.div style={{ y: yForeground }} className="relative z-10 w-full">
        {children}
      </motion.div>
    </div>
  );
};
