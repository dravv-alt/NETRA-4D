import React from 'react';
import { motion } from 'motion/react';
import { PieChart, TrendingUp, ShieldCheck } from 'lucide-react';
import { ParallaxCard } from '../ui/parallax-card';

export const BehaviorDistribution: React.FC = () => {
  const behaviorCategories = [
    { label: 'Normal Laptop Focus', pct: 80, color: 'bg-emerald-400', count: '16 Terminals' },
    { label: 'Routine Attention / Glance', pct: 10, color: 'bg-violet-400', count: '2 Terminals' },
    { label: 'Needs Invigilator Review', pct: 5, color: 'bg-amber-400', count: '1 Terminal' },
    { label: 'Critical Anomaly', pct: 5, color: 'bg-rose-500', count: '1 Terminal' },
  ];

  const timeBuckets = [
    { time: '15:00', events: 1 },
    { time: '15:10', events: 2 },
    { time: '15:20', events: 0 },
    { time: '15:30', events: 3 },
    { time: '15:40', events: 2 },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 font-sans">
      {/* Posture Distribution */}
      <ParallaxCard tiltAngle={4} className="p-6 flex flex-col justify-between bg-white border border-neutral-200 shadow-xs rounded-2xl">
        <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
          <div className="flex items-center gap-2.5 text-base font-semibold text-neutral-950">
            <div className="p-1.5 rounded-lg bg-sky-50 text-sky-700 border border-sky-100">
              <PieChart size={16} />
            </div>
            <span>Behavioral Distribution</span>
          </div>
          <span className="text-xs text-neutral-500 font-medium">20 Lab Terminals</span>
        </div>

        {/* Stacked Bar in Soft Pastel */}
        <div className="my-4">
          <div className="h-3.5 w-full rounded-full overflow-hidden flex bg-neutral-100 border border-neutral-200 p-0.5">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: '80%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="bg-emerald-400 h-full rounded-l-full"
              title="Normal: 80%"
            />
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: '10%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200, delay: 0.1 }}
              className="bg-violet-400 h-full"
              title="Observed: 10%"
            />
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: '5%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200, delay: 0.2 }}
              className="bg-amber-400 h-full"
              title="Needs Review: 5%"
            />
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: '5%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200, delay: 0.3 }}
              className="bg-rose-500 h-full rounded-r-full"
              title="Escalated: 5%"
            />
          </div>
        </div>

        {/* Breakdown Items with Pastel Dots */}
        <div className="space-y-2.5 text-xs">
          {behaviorCategories.map((cat, idx) => (
            <div key={idx} className="flex items-center justify-between text-neutral-600">
              <div className="flex items-center gap-2.5">
                <span className={`w-2.5 h-2.5 rounded-full ${cat.color}`} />
                <span>{cat.label}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-neutral-400">{cat.count}</span>
                <strong className="text-neutral-900 tabular-nums w-10 text-right font-semibold">{cat.pct}%</strong>
              </div>
            </div>
          ))}
        </div>
      </ParallaxCard>

      {/* Activity Timeline Bar Chart */}
      <ParallaxCard tiltAngle={4} className="p-6 flex flex-col justify-between bg-white border border-neutral-200 shadow-xs rounded-2xl">
        <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
          <div className="flex items-center gap-2.5 text-base font-semibold text-neutral-950">
            <div className="p-1.5 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-100">
              <TrendingUp size={16} />
            </div>
            <span>Signal Frequency</span>
          </div>
          <span className="text-xs text-neutral-500 font-medium">Rolling 45 Mins</span>
        </div>

        {/* Histogram Representation in Soft Pastel */}
        <div className="h-32 flex items-end justify-between gap-4 px-4 pt-4 pb-2 border-b border-neutral-200">
          {timeBuckets.map((bucket, idx) => {
            const height = Math.max(15, bucket.events * 30);
            return (
              <div key={idx} className="flex-1 flex flex-col items-center gap-2 h-full justify-end group cursor-pointer">
                <span className="text-xs text-neutral-500 tabular-nums font-semibold group-hover:text-black transition-colors">
                  {bucket.events}
                </span>
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: `${height}%` }}
                  transition={{ type: 'spring', damping: 25, stiffness: 220, delay: idx * 0.08 }}
                  className={`w-full rounded-t-lg transition-all duration-150 ${
                    bucket.events > 2
                      ? 'bg-rose-400'
                      : bucket.events === 2
                      ? 'bg-amber-400'
                      : 'bg-emerald-300 group-hover:bg-emerald-400'
                  }`}
                />
                <span className="text-xs text-neutral-400 font-medium">{bucket.time}</span>
              </div>
            );
          })}
        </div>

        {/* Footnote */}
        <div className="text-xs text-neutral-500 mt-3 flex items-center justify-between">
          <span>Mean Rate: <strong className="text-neutral-900 font-semibold">0.11 signals/min</strong></span>
          <span className="text-emerald-700 font-medium flex items-center gap-1.5">
            <ShieldCheck size={14} className="text-emerald-600" />
            Integrity: Optimal
          </span>
        </div>
      </ParallaxCard>
    </div>
  );
};
