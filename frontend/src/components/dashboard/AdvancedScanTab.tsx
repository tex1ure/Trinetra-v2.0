import React from 'react';
import { motion } from 'framer-motion';
import {
  Sparkles, Loader2, AlertCircle, ShieldAlert, ShieldCheck, ShieldQuestion,
  Target, Scale, Search, Lightbulb, TriangleAlert, RefreshCw,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { advanceScan } from '@/lib/api';

interface AdvancedScanTabProps {
  file: File;
  analysis: string | null;
  setAnalysis: (v: string | null) => void;
  isLoading: boolean;
  setIsLoading: (v: boolean) => void;
  error: string | null;
  setError: (v: string | null) => void;
}

// â”€â”€ Section config â”€â”€
const SECTION_CONFIG: Record<string, { icon: React.ElementType; color: string; bg: string; border: string }> = {
  verdict:      { icon: ShieldAlert,    color: 'text-brand-orange',    bg: 'bg-brand-orange/10', border: 'border-brand-orange/20' },
  confidence:   { icon: Target,         color: 'text-blue-500',        bg: 'bg-blue-500/10',     border: 'border-blue-500/20' },
  strongest:    { icon: Search,         color: 'text-red-500',         bg: 'bg-red-500/10',      border: 'border-red-500/20' },
  against:      { icon: Scale,          color: 'text-emerald-500',     bg: 'bg-emerald-500/10',  border: 'border-emerald-500/20' },
  suspicious:   { icon: Target,         color: 'text-amber-500',       bg: 'bg-amber-500/10',    border: 'border-amber-500/20' },
  alternative:  { icon: Lightbulb,      color: 'text-violet-500',      bg: 'bg-violet-500/10',   border: 'border-violet-500/20' },
  failure:      { icon: TriangleAlert,  color: 'text-rose-500',        bg: 'bg-rose-500/10',     border: 'border-rose-500/20' },
  finalrule:    { icon: ShieldCheck,    color: 'text-indigo-500',      bg: 'bg-indigo-500/10',   border: 'border-indigo-500/20' },
};

// â”€â”€ Robust parser: matches **Header:** patterns anywhere in a line â”€â”€
interface ParsedSection {
  key: string;
  title: string;
  content: string;
}

const SECTION_MATCHERS: { key: string; regex: RegExp; title: string }[] = [
  { key: 'verdict',     regex: /^\s*\*{0,2}\s*Verdict\s*:?\s*\*{0,2}\s*/i,                                   title: 'Verdict' },
  { key: 'confidence',  regex: /^\s*\*{0,2}\s*Confidence\s*:?\s*\*{0,2}\s*/i,                                title: 'Confidence Score' },
  { key: 'strongest',   regex: /^\s*\*{0,2}\s*(?:Why this (?:verdict )?is strongest|Strongest Evidence)\s*:?\s*\*{0,2}\s*/i,        title: 'Key Evidence' },
  { key: 'against',     regex: /^\s*\*{0,2}\s*(?:What argues against it|Contradicting Evidence)\s*:?\s*\*{0,2}\s*/i,                    title: 'Counter-Evidence' },
  { key: 'suspicious',  regex: /^\s*\*{0,2}\s*Most suspicious regions?\s*:?\s*\*{0,2}\s*/i,                  title: 'Suspicious Regions' },
  { key: 'alternative', regex: /^\s*\*{0,2}\s*Alternative explanations?\s*:?\s*\*{0,2}\s*/i,                 title: 'Alternative Explanations' },
  { key: 'failure',     regex: /^\s*\*{0,2}\s*Failure risk\s*:?\s*\*{0,2}\s*/i,                              title: 'Failure Risk' },
  { key: 'finalrule',   regex: /^\s*\*{0,2}\s*Final rule\s*:?\s*\*{0,2}\s*/i,                                title: 'Final Rule' },
];

function parseAnalysis(raw: string): ParsedSection[] {
  const sections: ParsedSection[] = [];
  let currentSection: ParsedSection | null = null;

  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (currentSection) currentSection.content += '\n';
      continue;
    }

    let matched = false;
    for (const matcher of SECTION_MATCHERS) {
      if (matcher.regex.test(trimmed)) {
        // Save previous section
        if (currentSection) {
          currentSection.content = currentSection.content.trim();
          sections.push(currentSection);
        }
        // Extract any inline content after the header
        const afterHeader = trimmed.replace(matcher.regex, '').replace(/^\s*:?\s*/, '').trim();
        currentSection = { key: matcher.key, title: matcher.title, content: afterHeader };
        matched = true;
        break;
      }
    }

    if (!matched && currentSection) {
      currentSection.content += '\n' + line;
    }
  }

  if (currentSection) {
    currentSection.content = currentSection.content.trim();
    sections.push(currentSection);
  }

  return sections;
}

// â”€â”€ Verdict badge â”€â”€
function VerdictBadge({ verdict }: { verdict: string }) {
  const lower = verdict.toLowerCase();
  const isBad = lower.includes('manipulated') || lower.includes('fake') || lower.includes('ai-generated') || lower.includes('ai generated') || lower.includes('deepfake');
  const isReal = lower.includes('real') || lower.includes('authentic');

  let gradient = 'from-amber-500 to-orange-500';
  let Icon = ShieldQuestion;
  let labelColor = 'text-amber-100';
  if (isBad) {
    gradient = 'from-red-500 to-rose-600';
    Icon = ShieldAlert;
    labelColor = 'text-red-100';
  } else if (isReal) {
    gradient = 'from-emerald-500 to-green-600';
    Icon = ShieldCheck;
    labelColor = 'text-emerald-100';
  }

  return (
    <div className={`relative overflow-hidden rounded-2xl bg-gradient-to-br ${gradient} p-6 shadow-lg`}>
      {/* Background decoration */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
      <div className="absolute bottom-0 left-0 w-20 h-20 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />

      <div className="relative z-10 flex items-center gap-4">
        <div className="h-14 w-14 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
          <Icon className="h-7 w-7 text-white" />
        </div>
        <div>
          <p className={`text-xs font-medium uppercase tracking-widest ${labelColor} mb-1`}>Verdict</p>
          <p className="text-white font-display font-bold text-2xl sm:text-3xl tracking-tight">{verdict}</p>
        </div>
      </div>
    </div>
  );
}

// â”€â”€ Confidence bar â”€â”€
function ConfidenceBar({ raw }: { raw: string }) {
  const num = parseInt(raw.replace(/[^0-9]/g, ''), 10);
  if (isNaN(num)) return <span className="text-foreground font-bold text-lg">{raw}</span>;

  const pct = Math.min(100, Math.max(0, num));
  let barColor = 'bg-emerald-500';
  if (pct >= 80) barColor = 'bg-red-500';
  else if (pct >= 50) barColor = 'bg-amber-500';

  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-2">
        <span className="font-display font-bold text-3xl text-foreground">{pct}</span>
        <span className="text-sm text-muted-foreground font-medium">/ 100</span>
      </div>
      <div className="h-2.5 w-full bg-muted rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`h-full rounded-full ${barColor}`}
        />
      </div>
    </div>
  );
}

// â”€â”€ Section card â”€â”€
function SectionCard({ section, index }: { section: ParsedSection; index: number }) {
  const config = SECTION_CONFIG[section.key] || SECTION_CONFIG['strongest'];
  const Icon = config.icon;

  // Verdict â€” big hero card
  if (section.key === 'verdict') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.06 }}
      >
        <VerdictBadge verdict={section.content} />
      </motion.div>
    );
  }

  // Confidence â€” with animated bar
  if (section.key === 'confidence') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.06 }}
        className={`rounded-2xl border ${config.border} ${config.bg} p-5`}
      >
        <div className="flex items-center gap-2 mb-3">
          <Icon className={`h-5 w-5 ${config.color}`} />
          <h4 className="font-display font-bold text-base text-foreground">{section.title}</h4>
        </div>
        <ConfidenceBar raw={section.content} />
      </motion.div>
    );
  }

  // Default card
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="rounded-2xl border border-border bg-card p-5"
    >
      <div className="flex items-center gap-2 mb-3">
        <div className={`h-8 w-8 rounded-lg ${config.bg} flex items-center justify-center shrink-0`}>
          <Icon className={`h-4 w-4 ${config.color}`} />
        </div>
        <h4 className="font-display font-bold text-base text-foreground">{section.title}</h4>
      </div>
      <div className="prose prose-sm dark:prose-invert max-w-none text-muted-foreground prose-strong:text-foreground prose-li:marker:text-muted-foreground/50 leading-relaxed">
        <ReactMarkdown>{section.content}</ReactMarkdown>
      </div>
    </motion.div>
  );
}

// â”€â”€ Main component â”€â”€
export function AdvancedScanTab({ file, analysis, setAnalysis, isLoading, setIsLoading, error, setError }: AdvancedScanTabProps) {
  const [cloudConsent, setCloudConsent] = React.useState(false);

  const handleScan = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await advanceScan(file, cloudConsent);
      setAnalysis(result.analysis);
    } catch (err: any) {
      setError(err.message || 'Advanced scan failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const sections = analysis ? parseAnalysis(analysis) : [];

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-border bg-card p-6 shadow-soft">
        <div className="flex items-center gap-3 mb-4">
          <div className="h-10 w-10 rounded-xl bg-gradient-brand flex items-center justify-center text-white">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-display text-lg font-bold">Deep Forensic Analysis</h3>
            <p className="text-sm text-muted-foreground">Powered by Gemini Gemma-4-31B-IT</p>
          </div>
        </div>

        {!analysis && !isLoading && !error && (
          <div className="mt-6 flex flex-col items-center justify-center py-10 text-center border border-dashed border-border rounded-xl bg-muted/30">
            <p className="text-muted-foreground mb-4 max-w-sm text-sm">
              Run an advanced multimodal scan to inspect every visible region for signs of AI generation, face swapping, body editing, and synthetic reconstruction.
            </p>
            <label className="flex items-start gap-2.5 max-w-md text-left mb-4 p-3 rounded-xl border border-amber-500/30 bg-amber-500/5 cursor-pointer">
              <input
                type="checkbox"
                checked={cloudConsent}
                onChange={(e) => setCloudConsent(e.target.checked)}
                className="mt-0.5 h-4 w-4 accent-brand-orange shrink-0"
              />
              <span className="text-xs text-muted-foreground leading-relaxed">
                <strong className="text-foreground">External Cloud Consent:</strong> I understand this sends a GPS/EXIF-stripped copy of my image to Google&apos;s Gemini API for multimodal inspection. Do not use with intimate or confidential imagery on free-tier API keys.
              </span>
            </label>
            <button
              onClick={handleScan}
              disabled={!cloudConsent}
              className="px-6 py-2.5 rounded-xl bg-foreground text-background font-medium hover:bg-foreground/90 transition-colors flex items-center gap-2 shadow-glow disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Sparkles className="h-4 w-4" /> Start Advanced Scan
            </button>
          </div>
        )}

        {isLoading && (
          <div className="mt-6 flex flex-col items-center justify-center py-12 text-center">
            <Loader2 className="h-8 w-8 animate-spin text-brand-orange mb-4" />
            <p className="text-sm font-medium animate-pulse text-foreground">
              Running deep multimodal forensic analysis...
            </p>
            <p className="text-xs text-muted-foreground mt-2">
              This may take up to 20 seconds depending on image complexity.
            </p>
          </div>
        )}

        {error && (
          <div className="mt-6 p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive flex gap-3">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <div className="text-sm">{error}</div>
          </div>
        )}

        {analysis && !isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 space-y-4"
          >
            {/* Hero: Verdict full-width */}
            {sections.filter(s => s.key === 'verdict').map((s, i) => (
              <SectionCard key={s.key} section={s} index={i} />
            ))}

            {/* Confidence below verdict */}
            {sections.filter(s => s.key === 'confidence').map((s, i) => (
              <SectionCard key={s.key} section={s} index={i + 1} />
            ))}

            {/* Remaining sections */}
            {sections.filter(s => s.key !== 'verdict' && s.key !== 'confidence').map((s, i) => (
              <SectionCard key={s.key} section={s} index={i + 2} />
            ))}

            {/* Fallback if parsing fails */}
            {sections.length === 0 && (
              <div className="prose prose-sm dark:prose-invert max-w-none bg-muted/30 p-5 rounded-xl border border-border">
                <ReactMarkdown>{analysis}</ReactMarkdown>
              </div>
            )}
            
            <div className="flex justify-end pt-2">
              <button
                onClick={handleScan}
                className="px-4 py-2 rounded-xl bg-muted hover:bg-muted/80 text-foreground font-medium text-sm transition-colors flex items-center gap-2"
              >
                <RefreshCw className="h-3.5 w-3.5" /> Rescan Image
              </button>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

