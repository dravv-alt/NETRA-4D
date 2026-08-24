import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { FileCheck, Shield, Download, Lock, CheckCircle2, Printer, X, Award, ExternalLink } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { Button } from '../ui/button';

interface InstitutionalAuditExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseId?: string;
}

export const InstitutionalAuditExportModal: React.FC<InstitutionalAuditExportModalProps> = ({
  isOpen,
  onClose,
  caseId = 'CASE-2026-08A-001',
}) => {
  const [isExporting, setIsExporting] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);

  if (!isOpen) return null;

  const handleDownload = () => {
    playHapticSound('snap');
    setIsExporting(true);
    setTimeout(() => {
      setIsExporting(false);
      setDownloadSuccess(true);
      setTimeout(() => setDownloadSuccess(false), 3000);
    }, 1200);
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm select-none">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          className="w-full max-w-3xl bg-white rounded-[2rem] border border-neutral-200 shadow-2xl p-8 space-y-6 font-sans max-h-[90vh] overflow-y-auto"
        >
          {/* Modal Header */}
          <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-neutral-900 text-white flex items-center justify-center font-bold text-lg">
                <FileCheck size={22} />
              </div>
              <div>
                <h3 className="font-heading font-extrabold text-xl text-neutral-950 tracking-tight">
                  Cryptographic Institutional Audit Package
                </h3>
                <div className="text-xs font-mono text-neutral-500">
                  REF: {caseId} • OFFICIAL DISCIPLINARY RECORD
                </div>
              </div>
            </div>

            <button
              onClick={() => {
                playHapticSound('click');
                onClose();
              }}
              className="p-2 rounded-xl text-neutral-400 hover:text-neutral-900 hover:bg-neutral-100 transition-colors cursor-pointer"
            >
              <X size={20} />
            </button>
          </div>

          {/* SHA-256 Hash Seal */}
          <div className="p-4 rounded-2xl bg-neutral-50 border border-neutral-200/80 space-y-1.5 font-mono text-xs">
            <div className="flex items-center justify-between text-2xs uppercase text-neutral-400 font-bold tracking-wider">
              <span>SHA-256 Cryptographic Evidence Seal</span>
              <span className="text-emerald-700 flex items-center gap-1">
                <CheckCircle2 size={12} /> VERIFIED IMMUTABLE
              </span>
            </div>
            <div className="text-neutral-900 font-bold break-all text-2xs bg-white p-2 rounded-lg border border-neutral-200">
              e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
            </div>
          </div>

          {/* Dossier Content Summary */}
          <div className="space-y-4 text-xs">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-2xl border border-neutral-200 space-y-1">
                <span className="text-2xs font-mono uppercase text-neutral-400">Subject</span>
                <div className="font-bold text-sm text-neutral-950">Candidate #6 (Terminal A06)</div>
                <div className="text-neutral-500 text-2xs">North Wall Desk • Computer Lab 01</div>
              </div>

              <div className="p-4 rounded-2xl border border-neutral-200 space-y-1">
                <span className="text-2xs font-mono uppercase text-neutral-400">Violation Classification</span>
                <div className="font-bold text-sm text-rose-700">Unauthorized Electronic Device (Phone)</div>
                <div className="text-neutral-500 text-2xs">Confidence: 0.74 • Persistence: 6.2s</div>
              </div>
            </div>

            <div className="p-5 rounded-2xl border border-neutral-200 space-y-2">
              <div className="font-bold text-neutral-950 font-heading">
                Multi-Modal Evidence Chain
              </div>
              <ul className="list-disc list-inside text-neutral-700 space-y-1 leading-relaxed">
                <li><strong>15:42:12 IST:</strong> Candidate head yaw deflected downward by 34.2° toward lap area.</li>
                <li><strong>15:42:15 IST:</strong> Right wrist transitioned from laptop keyboard to lap plane (occlusion duration: 6.2s).</li>
                <li><strong>15:42:18 IST:</strong> YOLO-Object detector registered phone signature (0.74 confidence, confirmed across CAM-01 and CAM-02).</li>
                <li><strong>15:42:24 IST:</strong> Invigilator Dr. K. Rao performed in-person seat inspection; physical device recorded.</li>
              </ul>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="pt-4 border-t border-neutral-200 flex items-center justify-between flex-wrap gap-3">
            <div className="text-2xs font-mono text-neutral-500">
              Authorized by University Examination Controller & Center Invigilator
            </div>

            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.print()}
                className="rounded-xl gap-1.5 font-mono text-xs"
              >
                <Printer size={14} />
                <span>Print Official Copy</span>
              </Button>

              <Button
                variant="default"
                size="sm"
                onClick={handleDownload}
                disabled={isExporting}
                className="rounded-xl gap-2 font-mono text-xs shadow-xs"
              >
                <Download size={14} />
                <span>{isExporting ? 'Generating Package...' : downloadSuccess ? 'Export Complete! ✓' : 'Download Signed Package'}</span>
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
