import React, { useRef } from 'react';
import { motion, useMotionValue, useSpring, useTransform, HTMLMotionProps } from 'motion/react';
import { cn } from '../../lib/utils';

export interface ParallaxCardProps extends Omit<HTMLMotionProps<'div'>, 'children'> {
  children: React.ReactNode;
  tiltAngle?: number;
  className?: string;
  glareEffect?: boolean;
}

export const ParallaxCard: React.FC<ParallaxCardProps> = ({
  children,
  tiltAngle = 10,
  className,
  glareEffect = true,
  ...props
}) => {
  const cardRef = useRef<HTMLDivElement>(null);

  // Normalized mouse coordinates from -0.5 to 0.5
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Smooth springs for rotation
  const springConfig = { damping: 22, stiffness: 280, mass: 0.6 };
  const rotateX = useSpring(useTransform(mouseY, [-0.5, 0.5], [tiltAngle, -tiltAngle]), springConfig);
  const rotateY = useSpring(useTransform(mouseX, [-0.5, 0.5], [-tiltAngle, tiltAngle]), springConfig);

  // Glare position
  const glareX = useTransform(mouseX, [-0.5, 0.5], ['0%', '100%']);
  const glareY = useTransform(mouseY, [-0.5, 0.5], ['0%', '100%']);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    mouseX.set(x);
    mouseY.set(y);
  };

  const handleMouseLeave = () => {
    mouseX.set(0);
    mouseY.set(0);
  };

  return (
    <div style={{ perspective: 1000 }} className="h-full w-full">
      <motion.div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          rotateX,
          rotateY,
          transformStyle: 'preserve-3d',
        }}
        whileHover={{ scale: 1.012 }}
        transition={{ type: 'spring', damping: 25, stiffness: 350 }}
        className={cn(
          'apple-card relative overflow-hidden h-full flex flex-col',
          className
        )}
        {...props}
      >
        {/* Dynamic Specular Glare Layer (Light Mode) */}
        {glareEffect && (
          <motion.div
            className="pointer-events-none absolute -inset-px rounded-2xl opacity-0 hover:opacity-100 transition-opacity duration-300 z-30"
            style={{
              background: useTransform(
                [glareX, glareY],
                ([x, y]) => `radial-gradient(400px circle at ${x} ${y}, rgba(255, 255, 255, 0.7), transparent 60%)`
              ),
            }}
          />
        )}
        {children}
      </motion.div>
    </div>
  );
};
