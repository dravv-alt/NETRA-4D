import * as React from 'react';
import { Layers } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Switch } from '../ui/switch';

export interface AILayersState {
  personBoxes: boolean;
  trackingIds: boolean;
  poseSkeleton: boolean;
  objects: boolean;
  seatMapping: boolean;
  attentionRays: boolean;
  behaviorEvents: boolean;
  threeDVector: boolean;
}

interface LayerControlWidgetProps {
  layers: AILayersState;
  onToggleLayer: (layerKey: keyof AILayersState) => void;
}

export const LayerControlWidget: React.FC<LayerControlWidgetProps> = ({
  layers,
  onToggleLayer,
}) => {
  const layerItems: { key: keyof AILayersState; label: string; desc: string }[] = [
    { key: 'personBoxes', label: 'Person Boxes', desc: 'YOLO Detection Bounding Boxes' },
    { key: 'trackingIds', label: 'Tracking IDs', desc: 'ByteTrack Temporal Identity' },
    { key: 'poseSkeleton', label: 'Pose Skeleton', desc: '17-Point 2D Keypoint Lattice' },
    { key: 'objects', label: 'Object Candidates', desc: 'Mobile Devices & Peripherals' },
    { key: 'seatMapping', label: 'Seat Mapping', desc: 'Hall Grid Coordinate Anchors' },
    { key: 'attentionRays', label: 'Attention Rays', desc: 'Gaze & Head Yaw Vectors' },
    { key: 'behaviorEvents', label: 'Behavior HUD', desc: 'Active Evidence Anomaly Labels' },
    { key: 'threeDVector', label: '3D Mesh Vector', desc: 'Multi-HMR Projected Joints' },
  ];

  return (
    <div className="bg-white p-4 font-mono text-xs rounded-2xl border border-neutral-200 space-y-3.5 shadow-2xs">
      <div className="flex items-center justify-between pb-2.5 border-b border-neutral-200">
        <div className="flex items-center gap-2 font-bold text-neutral-950 uppercase tracking-wider text-2xs">
          <Layers size={14} className="text-neutral-900" />
          <span>AI Visual Layers</span>
        </div>
        <span className="text-2xs text-neutral-500 font-bold px-2 py-0.5 rounded-full bg-neutral-100 border border-neutral-200">HUD</span>
      </div>

      <div className="space-y-1">
        {layerItems.map(({ key, label, desc }) => {
          const isChecked = layers[key];
          const isAvailable = key !== 'threeDVector';
          return (
            <div
              key={key}
              title={isAvailable ? desc : 'No validated 3D output is available to render'}
              onClick={() => {
                if (!isAvailable) return;
                playHapticSound('snap');
                onToggleLayer(key);
              }}
              className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg transition-colors select-none ${isAvailable ? 'hover:bg-neutral-100 cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
            >
              <span className={`text-2xs truncate ${isChecked ? 'text-neutral-950 font-semibold' : 'text-neutral-500'}`}>
                {label}
              </span>

              <span onClick={(event) => event.stopPropagation()}>
                <Switch
                  checked={isChecked}
                  onCheckedChange={() => {
                    onToggleLayer(key);
                  }}
                  disabled={!isAvailable}
                  size="sm"
                />
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
