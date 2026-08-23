import React from 'react';
import { Users, AlertTriangle, Eye, ShieldAlert, Zap } from 'lucide-react';
import { MetricValue } from '../ui/MetricValue';
import { SystemHealthData } from '../../types';

interface MetricStripProps {
  candidatesCount?: number;
  activeSignalsCount?: number;
  needsReviewCount?: number;
  confirmedCasesCount?: number;
  health?: SystemHealthData;
}

export const MetricStrip: React.FC<MetricStripProps> = ({
  candidatesCount = 20,
  activeSignalsCount = 3,
  needsReviewCount = 2,
  confirmedCasesCount = 1,
  health,
}) => {
  const normalCount = candidatesCount - activeSignalsCount;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 font-sans">
      {/* Metric 1: Total Candidates */}
      <MetricValue
        label="Total Active"
        value={candidatesCount}
        sublabel={`${candidatesCount} Lab Terminals`}
        icon={
          <div className="p-1.5 rounded-lg bg-sky-50 text-sky-700 border border-sky-100">
            <Users size={16} />
          </div>
        }
      />

      {/* Metric 2: Nominal Writing */}
      <MetricValue
        label="Nominal Focus"
        value={normalCount}
        sublabel={`${((normalCount / candidatesCount) * 100).toFixed(0)}% of Cohort`}
        icon={
          <div className="p-1.5 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-100">
            <Zap size={16} />
          </div>
        }
        trend="Nominal"
      />

      {/* Metric 3: Under Observation */}
      <MetricValue
        label="Observation"
        value={activeSignalsCount}
        sublabel="Low threshold cues"
        icon={
          <div className="p-1.5 rounded-lg bg-violet-50 text-violet-700 border border-violet-100">
            <Eye size={16} />
          </div>
        }
      />

      {/* Metric 4: Needs Review */}
      <MetricValue
        label="Needs Review"
        value={needsReviewCount}
        sublabel="Pending triage"
        icon={
          <div className="p-1.5 rounded-lg bg-amber-50 text-amber-700 border border-amber-100">
            <AlertTriangle size={16} />
          </div>
        }
        trend="Elevated"
      />

      {/* Metric 5: Critical Anomaly */}
      <MetricValue
        label="Critical Anomaly"
        value={confirmedCasesCount}
        sublabel={health ? `${health.pipelineLatencyMs}ms DAG` : "Multi-signal"}
        icon={
          <div className="p-1.5 rounded-lg bg-rose-50 text-rose-700 border border-rose-100">
            <ShieldAlert size={16} />
          </div>
        }
        trend={confirmedCasesCount > 0 ? "Urgent" : "Clear"}
      />
    </div>
  );
};
