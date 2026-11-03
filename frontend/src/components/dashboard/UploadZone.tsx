'use client';

import { useCallback, useRef, useState } from 'react';
import { Upload, ImageIcon, X, Loader2, ShieldCheck, Cloud } from 'lucide-react';

interface UploadZoneProps {
  preview: string | null;
  fileName: string | null;
  fileSize: number | null;
  isAnalyzing: boolean;
  allowCloud: boolean;
  onToggleCloud: (val: boolean) => void;
  onFile: (file: File) => void;
  onClear: () => void;
  onAnalyze: () => void;
}

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

export const UploadZone = ({
  preview,
  fileName,
  fileSize,
  isAnalyzing,
  allowCloud,
  onToggleCloud,
  onFile,
  onClear,
  onAnalyze,
}: UploadZoneProps) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) return;
      if (file.size > 20 * 1024 * 1024) return;
      onFile(file);
    },
    [onFile]
  );

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-soft">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-lg font-semibold">Source media</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            JPG, PNG, WEBP or MP4 · max 20MB
          </p>
        </div>
        {preview && (
          <button
            onClick={onClear}
            disabled={isAnalyzing}
            className="inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded-xl border border-border bg-muted/40 hover:bg-muted/80 text-muted-foreground transition-colors disabled:opacity-50"
          >
            <X className="h-4 w-4" /> Clear
          </button>
        )}
      </div>

      {/* Privacy Guarantee Badge */}
      <div className="mb-4 rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-3 flex items-start gap-2.5">
        <ShieldCheck className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" />
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          <span className="font-semibold text-foreground">GPS &amp; EXIF Auto-Stripped:</span> Location coordinates and device serials are automatically removed before processing.
        </p>
      </div>

      {!preview ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={cn(
            'relative cursor-pointer rounded-2xl border-2 border-dashed transition-all',
            'flex flex-col items-center justify-center text-center p-10 min-h-[260px]',
            dragOver
              ? 'border-brand-orange bg-brand-orange/5'
              : 'border-border hover:border-brand-purple/60 hover:bg-muted/40'
          )}
        >
          <div className="h-14 w-14 rounded-2xl bg-gradient-brand flex items-center justify-center shadow-glow">
            <Upload className="h-6 w-6 text-white" />
          </div>
          <p className="mt-5 font-display font-semibold">
            Drop an image or video to analyze
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            or <span className="text-foreground underline">browse files</span>
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*,video/mp4,video/quicktime"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden border border-border bg-muted aspect-video">
            <img
              src={preview}
              alt="Uploaded preview"
              className="w-full h-full object-contain"
            />
            {isAnalyzing && (
              <div className="absolute inset-0 bg-background/40 backdrop-blur-sm flex items-center justify-center">
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-brand-orange to-transparent animate-scan" />
                <div className="rounded-full bg-card border border-border px-4 py-2 flex items-center gap-2 shadow-elevated">
                  <Loader2 className="h-4 w-4 animate-spin text-brand-orange" />
                  <span className="text-sm font-medium">Analyzing…</span>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
            <span className="flex items-center gap-1.5 truncate max-w-[240px]">
              <ImageIcon className="h-3.5 w-3.5 shrink-0" />
              {fileName}
            </span>
            <span>
              {fileSize ? (fileSize / 1024 / 1024).toFixed(2) : '0'} MB
            </span>
          </div>

          {/* Explicit Cloud Opt-In Toggle */}
          <label className="flex items-start gap-3 rounded-2xl border border-border bg-muted/30 p-3.5 cursor-pointer hover:border-brand-orange/40 transition-colors">
            <input
              type="checkbox"
              checked={allowCloud}
              onChange={(e) => onToggleCloud(e.target.checked)}
              className="mt-0.5 h-4 w-4 rounded accent-brand-orange"
            />
            <div className="text-xs leading-relaxed">
              <span className="font-semibold text-foreground flex items-center gap-1.5">
                <Cloud className="h-3.5 w-3.5 text-brand-orange" />
                Opt-in to Cloud Verification (Reality Defender)
              </span>
              <span className="text-muted-foreground block mt-0.5">
                {allowCloud
                  ? 'A metadata-stripped copy will be sent to Reality Defender Cloud.'
                  : 'Off (Default): Media stays 100% on this local machine.'}
              </span>
            </div>
          </label>

          <button
            onClick={onAnalyze}
            disabled={isAnalyzing}
            className="w-full h-12 rounded-2xl bg-foreground text-background hover:bg-foreground/90 text-base font-medium flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running detection
              </>
            ) : (
              allowCloud ? 'Analyze (Local + Cloud)' : 'Analyze Locally (Private)'
            )}
          </button>
        </div>
      )}
    </div>
  );
};
