'use client';

import { useState } from 'react';
import type { AnalysisResponse } from '@/lib/api';
import { Eye, Activity, Layers, Sparkles } from 'lucide-react';

interface HeatmapTabProps {
  preview: string;
  result: AnalysisResponse;
}

type ViewMode = 'gradcam' | 'ela' | 'face';

export const HeatmapTab = ({ preview, result }: HeatmapTabProps) => {
  const [viewMode, setViewMode] = useState<ViewMode>('gradcam');

  const activeImage =
    viewMode === 'gradcam' && result.gradcam_base64
      ? `data:image/png;base64,${result.gradcam_base64}`
      : viewMode === 'ela' && result.noise_residual_base64
      ? `data:image/png;base64,${result.noise_residual_base64}`
      : viewMode === 'face' && result.face_crop_base64
      ? `data:image/png;base64,${result.face_crop_base64}`
      : preview;

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-border bg-card p-6 shadow-soft">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <div>
            <h2 className="font-display text-lg font-semibold">
              Explainable AI (XAI) Visual Evidence
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              Inspect the real Grad‑CAM activation heatmap, Error Level Analysis (ELA) noise residuals, and extracted face region.
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-muted/50 p-1 rounded-xl border border-border">
            <button
              onClick={() => setViewMode('gradcam')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                viewMode === 'gradcam' ? 'bg-background shadow-soft text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Sparkles className="h-3.5 w-3.5 text-brand-orange" /> Grad‑CAM
            </button>
            <button
              onClick={() => setViewMode('ela')}
              disabled={!result.noise_residual_base64}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 disabled:opacity-40 ${
                viewMode === 'ela' ? 'bg-background shadow-soft text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Layers className="h-3.5 w-3.5 text-brand-purple" /> ELA Noise Map
            </button>
            <button
              onClick={() => setViewMode('face')}
              disabled={!result.face_crop_base64}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 disabled:opacity-40 ${
                viewMode === 'face' ? 'bg-background shadow-soft text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Eye className="h-3.5 w-3.5 text-emerald-500" /> Face Crop
            </button>
          </div>
        </div>

        <div className="relative rounded-2xl overflow-hidden border border-border bg-black aspect-video flex items-center justify-center">
          <img
            src={activeImage}
            alt="Forensic visual output"
            className="max-h-full max-w-full object-contain"
          />
        </div>

        <div className="mt-4 flex items-center justify-between flex-wrap gap-4 text-xs text-muted-foreground">
          <span>
            {viewMode === 'gradcam' && 'Grad-CAM highlights spatial regions (red/yellow) that most influenced the neural network verdict.'}
            {viewMode === 'ela' && 'Error Level Analysis (ELA) amplifies JPEG compression residuals to reveal synthetic blending seams.'}
            {viewMode === 'face' && 'MTCNN-isolated face region used as input tensor (384×384) for spatial-temporal inference.'}
          </span>
        </div>
      </div>

      {/* Temporal & Geometric Jitter Forensics (when available from video or multi-frame analysis) */}
      {(result.temporal_rollout || result.geometric_jitter) && (
        <div className="rounded-3xl border border-border bg-card p-6 shadow-soft">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-5 w-5 text-brand-orange" />
            <h3 className="font-display font-semibold">Temporal &amp; Geometric Forensics</h3>
          </div>
          <div className="grid sm:grid-cols-3 gap-4">
            {result.temporal_rollout && (
              <>
                <div className="rounded-2xl border border-border bg-muted/30 p-4">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Shatter Frame Index</p>
                  <p className="mt-1 font-display text-2xl font-bold text-brand-orange">
                    Frame #{result.temporal_rollout.shatter_index}
                  </p>
                </div>
                <div className="rounded-2xl border border-border bg-muted/30 p-4">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Max Frame Delta</p>
                  <p className="mt-1 font-display text-2xl font-bold text-brand-purple">
                    {result.temporal_rollout.shatter_delta}%
                  </p>
                </div>
              </>
            )}
            {result.geometric_jitter && (
              <div className="rounded-2xl border border-border bg-muted/30 p-4">
                <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Landmark Jitter Score</p>
                <p className="mt-1 font-display text-2xl font-bold text-foreground">
                  {result.geometric_jitter.jitter_score}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
