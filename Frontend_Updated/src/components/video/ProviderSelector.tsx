import React from 'react';
import { motion } from 'motion/react';
import { AIProvider } from '../../types';
import { Cpu, Sparkles, Box, Compass } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';

interface ProviderSelectorProps {
  activeProvider: AIProvider;
  onSelectProvider: (provider: AIProvider) => void;
}

export const ProviderSelector: React.FC<ProviderSelectorProps> = ({
  activeProvider,
  onSelectProvider,
}) => {
  const providers: { id: AIProvider; label: string; icon: React.ReactNode; desc: string }[] = [
    { id: 'fusion', label: 'Fusion Engine', icon: <Sparkles size={13} />, desc: 'Multi-modal weighted confidence across pose, object & context' },
    { id: 'yolo_pose', label: 'YOLO-Pose', icon: <Box size={13} />, desc: '18.3 FPS Real-time 2D Detection & ByteTrack association' },
    { id: 'rtmpose', label: 'RTMPose', icon: <Cpu size={13} />, desc: '8.6 FPS Sub-pixel keypoint accuracy & wrist proximity' },
    { id: 'multihmr_3d', label: 'Multi-HMR 3D', icon: <Compass size={13} />, desc: '2.8 FPS 3D torso orientation & gaze angle vector' },
  ];

  return (
    <div className="flex items-center gap-1.5 font-mono text-2xs apple-glass p-1 rounded-2xl border border-black/[0.06] shadow-2xs bg-slate-100/70">
      <span className="text-slate-400 uppercase px-2 font-bold text-3xs">Provider:</span>
      <div className="flex items-center gap-1">
        {providers.map((p) => {
          const isSelected = activeProvider === p.id;
          return (
            <motion.button
              key={p.id}
              onClick={() => {
                playHapticSound('snap');
                onSelectProvider(p.id);
              }}
              title={p.desc}
              whileTap={{ scale: 0.96 }}
              className={`relative px-3 py-1.5 rounded-xl flex items-center gap-1.5 transition-colors duration-150 cursor-pointer ${
                isSelected
                  ? 'text-sky-700 font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-black/5'
              }`}
            >
              {isSelected && (
                <motion.div
                  layoutId="providerPill"
                  transition={{ type: 'spring', damping: 28, stiffness: 380 }}
                  className="absolute inset-0 bg-white border border-black/[0.06] rounded-xl shadow-xs"
                />
              )}
              <span className={`relative z-10 ${isSelected ? 'text-sky-600' : 'text-slate-400'}`}>
                {p.icon}
              </span>
              <span className="relative z-10">{p.label}</span>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
};
