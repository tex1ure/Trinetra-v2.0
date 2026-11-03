'use client';

import React, { useState, useCallback, useEffect } from 'react';
import {
  FlameKindling, FileText, Sparkles, Lock, LogOut,
  Settings, Sun, Moon, Loader2, Languages, ChevronDown,
  MapPin, Gavel, Globe, Shield,
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { UploadZone } from '@/components/dashboard/UploadZone';
import { ResultTab } from '@/components/dashboard/ResultTab';
import { HeatmapTab } from '@/components/dashboard/HeatmapTab';
import { SummaryTab } from '@/components/dashboard/SummaryTab';
import { AdvancedScanTab } from '@/components/dashboard/AdvancedScanTab';
import { AlibiTab } from '@/components/dashboard/AlibiTab';
import { TakedownTab } from '@/components/dashboard/TakedownTab';
import { BlastRadiusTab } from '@/components/dashboard/BlastRadiusTab';
import { CrisisTab } from '@/components/dashboard/CrisisTab';
import {
  SettingsPanel,
  LANGUAGE_OPTIONS,
  TRANSLATIONS,
  type LangCode,
  type HistoryItem,
} from '@/components/dashboard/SettingsPanel';
import { analyzeMedia, checkHealth, type AnalysisResponse } from '@/lib/api';

// â”€â”€â”€ Tab config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
type TabId = 'result' | 'heatmap' | 'summary' | 'advanced';
const TABS: { id: TabId; label: string; Icon: React.ElementType }[] = [
  { id: 'result',  label: 'Result',  Icon: Sparkles },
  { id: 'heatmap', label: 'Heatmap', Icon: FlameKindling },
  { id: 'summary', label: 'Summary', Icon: FileText },
  { id: 'advanced', label: 'Advanced Scan', Icon: Sparkles },
];

// â”€â”€â”€ Victim Response tabs (FAKE only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
type VictimTabId = 'alibi' | 'takedown' | 'blast-radius' | 'crisis';
const VICTIM_TABS: { id: VictimTabId; label: string; Icon: React.ElementType; color: string }[] = [
  { id: 'alibi',        label: 'Digital Alibi',  Icon: MapPin, color: 'from-blue-500 to-indigo-600' },
  { id: 'takedown',     label: 'Takedown',       Icon: Gavel,  color: 'from-red-500 to-rose-600' },
  { id: 'blast-radius', label: 'Blast Radius',   Icon: Globe,  color: 'from-amber-500 to-red-600' },
  { id: 'crisis',       label: 'Crisis Support', Icon: Shield, color: 'from-pink-500 to-purple-600' },
];

// â”€â”€â”€ Login Screen â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function LoginScreen({
  onLogin,
  lang,
}: {
  onLogin: () => void;
  lang: LangCode;
}) {
  const t = TRANSLATIONS[lang];
  return (
    <div className="min-h-screen bg-background bg-aurora flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md rounded-3xl border border-border bg-card p-10 shadow-elevated"
      >
        <div className="flex items-center gap-3 mb-8">
          <img
            src="/Trinetra_V3.png"
            alt="Logo"
            className="h-9 w-9 object-contain"
            onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
          />
          <span className="font-display font-bold text-xl">Trinetra</span>
        </div>

        <h1 className="font-display text-3xl font-bold tracking-tighter mb-2">{t.welcome}</h1>
        <p className="text-muted-foreground text-sm mb-8">{t.welcomeSub}</p>

        <div className="space-y-4">
          <input
            type="email"
            placeholder="Email address"
            className="w-full px-4 py-3 rounded-2xl bg-muted/40 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-brand-orange/40 text-sm"
          />
          <input
            type="password"
            placeholder="Password"
            className="w-full px-4 py-3 rounded-2xl bg-muted/40 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-brand-orange/40 text-sm"
          />
          <button
            onClick={onLogin}
            className="w-full py-3 rounded-2xl bg-foreground text-background font-semibold text-sm hover:bg-foreground/90 transition-colors flex items-center justify-center gap-2"
          >
            <Lock className="h-4 w-4" /> {t.signIn}
          </button>
        </div>

        <div className="relative my-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-border" />
          </div>
          <div className="relative flex justify-center">
            <span className="bg-card px-3 text-xs text-muted-foreground">OR</span>
          </div>
        </div>

        <button
          onClick={onLogin}
          className="w-full py-3 rounded-2xl border-2 border-border text-foreground font-semibold text-sm hover:border-brand-orange/60 hover:text-brand-orange transition-colors"
        >
          {t.guest}
        </button>
      </motion.div>
    </div>
  );
}

// â”€â”€â”€ Empty / Waiting state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function EmptyState({ hasFile, isAnalyzing }: { hasFile: boolean; isAnalyzing: boolean }) {
  return (
    <div className="rounded-3xl border border-dashed border-border bg-card/40 min-h-[500px] flex flex-col items-center justify-center text-center p-10">
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-brand blur-3xl opacity-20 rounded-full" />
        <img
          src="/Trinetra_V3.png"
          alt=""
          aria-hidden
          className="relative h-20 w-20 animate-float"
          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
        />
      </div>
      <h3 className="mt-8 font-display text-2xl font-bold tracking-tight">
        {isAnalyzing ? 'Running detectionâ€¦' : hasFile ? 'Ready to analyze' : 'Upload an image to begin'}
      </h3>
      <p className="text-muted-foreground mt-2 max-w-sm text-sm">
        {isAnalyzing
          ? 'Trinetra is examining facial landmarks, frequency artifacts, and GAN fingerprints.'
          : hasFile
          ? 'Click "Analyze with Trinetra" to run the deepfake detection pipeline.'
          : 'Drag a JPG or PNG into the panel on the left, then run analysis to see the verdict, heatmap, and summary tabs populate.'}
      </p>
    </div>
  );
}

// â”€â”€â”€ Main Page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function DashboardPage() {
  const [isLoggedIn, setIsLoggedIn]         = useState(true);
  const [darkMode, setDarkMode]             = useState(false);
  const [lang, setLang]                     = useState<LangCode>('en');
  const [showLangMenu, setShowLangMenu]     = useState(false);
  const [showSettings, setShowSettings]     = useState(false);
  const [file, setFile]                     = useState<File | null>(null);
  const [preview, setPreview]               = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing]       = useState(false);
  const [result, setResult]                 = useState<AnalysisResponse | null>(null);
  const [error, setError]                   = useState<string | null>(null);
  const [tab, setTab]                       = useState<TabId>('result');
  const [backendOnline, setBackendOnline]   = useState<boolean | null>(null);
  const [history, setHistory]               = useState<HistoryItem[]>([]);
  const [victimTab, setVictimTab]           = useState<VictimTabId>('alibi');
  const [panicMode, setPanicMode]           = useState(false);
  const [allowCloud, setAllowCloud]         = useState(false);

  // Advanced scan state â€” lifted here so it survives tab switches
  const [advancedAnalysis, setAdvancedAnalysis] = useState<string | null>(null);
  const [advancedLoading, setAdvancedLoading]   = useState(false);
  const [advancedError, setAdvancedError]       = useState<string | null>(null);

  const t = TRANSLATIONS[lang] || TRANSLATIONS['en'];
  const currentLang = LANGUAGE_OPTIONS.find((o) => o.code === lang);

  // Dark mode
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  // Health check
  useEffect(() => {
    if (!isLoggedIn) return;
    checkHealth()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, [isLoggedIn]);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
    // Reset advanced scan when a new file is uploaded
    setAdvancedAnalysis(null);
    setAdvancedLoading(false);
    setAdvancedError(null);
  }, []);

  const handleClear = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }, [preview]);

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setResult(null);
    setError(null);
    try {
      const r = await analyzeMedia(file, allowCloud);
      setResult(r);
      setTab('result');
      // Record in history
      setHistory((prev) => [
        {
          id: Date.now().toString(),
          name: file.name,
          date: new Date().toLocaleString(),
          verdict: r.primary_verdict,
          confidence: r.confidence_score,
        },
        ...prev.slice(0, 19), // keep last 20
      ]);
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Is the backend running?');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    handleClear();
  };

  if (!isLoggedIn) {
    return <LoginScreen onLogin={() => setIsLoggedIn(true)} lang={lang} />;
  }

  return (
    <div className={`min-h-screen bg-background ${panicMode ? 'panic-blur' : ''}`}>
      {/* â”€â”€ Header â”€â”€ */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-background/80 border-b border-border">
        <div className="container flex items-center justify-between h-16">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <img
              src="/Trinetra_V3.png"
              alt="Logo"
              className="h-7 w-7 object-contain"
              onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
            />
            <span className="font-display font-bold">Trinetra</span>
            <span className="text-xs text-muted-foreground hidden sm:inline">/ Dashboard</span>
          </div>

          {/* Right Controls */}
          <div className="flex items-center gap-2">

            {/* Backend pill */}
            <span className="hidden md:inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/40 px-3 py-1 text-xs font-mono text-muted-foreground">
              {backendOnline === null && <Loader2 className="h-3 w-3 animate-spin" />}
              {backendOnline === true  && <><span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />API connected Â· v1.0</>}
              {backendOnline === false && <><span className="h-1.5 w-1.5 rounded-full bg-red-500" />API offline</>}
            </span>

            {/* Language switcher */}
            <div className="relative">
              <button
                onClick={() => setShowLangMenu((v) => !v)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-border bg-muted/40 hover:bg-muted/70 text-sm font-medium transition-colors"
              >
                <Languages className="h-4 w-4 text-brand-orange" />
                <span>{currentLang?.native}</span>
                <ChevronDown className="h-3 w-3 text-muted-foreground" />
              </button>

              <AnimatePresence>
                {showLangMenu && (
                  <motion.div
                    initial={{ opacity: 0, y: -8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -8, scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-full mt-2 w-48 rounded-2xl border border-border bg-card shadow-elevated overflow-hidden z-50"
                  >
                    {LANGUAGE_OPTIONS.map((opt) => (
                      <button
                        key={opt.code}
                        onClick={() => { setLang(opt.code); setShowLangMenu(false); }}
                        className={`w-full text-left px-4 py-2.5 text-sm transition-colors ${
                          lang === opt.code
                            ? 'font-bold text-brand-orange bg-brand-orange/10'
                            : 'text-foreground hover:bg-muted/60 hover:text-brand-orange'
                        }`}
                      >
                        <span className="font-bold mr-2">{opt.native}</span>
                        <span className="text-muted-foreground text-xs">{opt.label}</span>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Dark mode */}
            <button
              onClick={() => setDarkMode((d) => !d)}
              className="p-2 rounded-xl hover:bg-muted/60 text-muted-foreground transition-colors"
              title="Toggle theme"
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>

            {/* Settings */}
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 rounded-xl hover:bg-muted/60 text-muted-foreground hover:text-brand-orange transition-colors"
              title="Settings"
            >
              <Settings className="h-4 w-4" />
            </button>

            {/* Logout */}
            <button
              onClick={handleLogout}
              className="p-2 rounded-xl hover:bg-muted/60 text-muted-foreground transition-colors"
              title="Sign out"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* â”€â”€ Main â”€â”€ */}
      <main className="container py-8 md:py-12">
        {/* Page header */}
        <div className="mb-8">
          <h1 className="font-display text-3xl md:text-5xl font-bold tracking-tighter">
            {t.workspace} <span className="text-gradient-brand">{t.workspaceSub}</span>
          </h1>
          <p className="text-muted-foreground mt-2 max-w-xl text-sm md:text-base">{t.subtitle}</p>
        </div>

        {/* Two-column layout */}
        <div className="grid lg:grid-cols-[400px_1fr] gap-6">

          {/* Left: upload (sticky) */}
          <div className="lg:self-start">
            <UploadZone
              preview={preview}
              fileName={file?.name ?? null}
              fileSize={file?.size ?? null}
              isAnalyzing={isAnalyzing}
              allowCloud={allowCloud}
              onToggleCloud={setAllowCloud}
              onFile={handleFile}
              onClear={handleClear}
              onAnalyze={handleAnalyze}
            />
            {error && (
              <div className="mt-4 p-4 rounded-2xl bg-destructive/10 border border-destructive/30 text-destructive text-sm font-medium">
                {error}
              </div>
            )}
          </div>

          {/* Right: results */}
          <div>
            {!result ? (
              <EmptyState hasFile={!!file} isAnalyzing={isAnalyzing} />
            ) : (
              <div>
                {/* Tab bar */}
                <div className="w-full grid grid-cols-4 h-12 rounded-2xl bg-muted/60 p-1 mb-6">
                  {TABS.map(({ id, label, Icon }) => (
                    <button
                      key={id}
                      onClick={() => setTab(id)}
                      className={`rounded-xl flex items-center justify-center gap-2 text-sm font-medium transition-all ${
                        tab === id
                          ? 'bg-background shadow-soft text-foreground'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                      <span className="hidden sm:inline">{label}</span>
                    </button>
                  ))}
                </div>

                {/* Tab content */}
                <AnimatePresence mode="wait">
                  <motion.div
                    key={tab}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.2 }}
                  >
                    {tab === 'result'  && <ResultTab result={result} />}
                    {tab === 'heatmap' && <HeatmapTab preview={preview!} result={result} />}
                    {tab === 'summary' && <SummaryTab result={result} fileName={file?.name ?? 'image'} />}
                    {tab === 'advanced' && <AdvancedScanTab file={file!} analysis={advancedAnalysis} setAnalysis={setAdvancedAnalysis} isLoading={advancedLoading} setIsLoading={setAdvancedLoading} error={advancedError} setError={setAdvancedError} />}
                  </motion.div>
                </AnimatePresence>
              </div>
            )}
          </div>

          {/* â”€â”€ Victim Response Toolkit (FAKE verdict only) â”€â”€ */}
          {result && result.primary_verdict === 'FAKE' && (
            <div className="lg:col-span-2 mt-4">
              {/* Section header */}
              <div className="flex items-center gap-3 mb-6">
                <div className="h-10 w-10 rounded-2xl bg-gradient-to-br from-red-500 to-brand-orange flex items-center justify-center shadow-glow">
                  <Shield className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="font-display text-2xl md:text-3xl font-bold tracking-tighter">
                    Victim Response <span className="text-gradient-brand">Toolkit</span>
                  </h2>
                  <p className="text-xs text-muted-foreground mt-0.5">Fight back â€” prove your truth, take it down, track the spread, get support</p>
                </div>
              </div>

              {/* Victim tab bar */}
              <div className="w-full grid grid-cols-4 h-12 rounded-2xl bg-muted/60 p-1 mb-6">
                {VICTIM_TABS.map(({ id, label, Icon, color }) => (
                  <button
                    key={id}
                    onClick={() => setVictimTab(id)}
                    className={`rounded-xl flex items-center justify-center gap-2 text-sm font-medium transition-all ${
                      victimTab === id
                        ? 'bg-background shadow-soft text-foreground'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{label}</span>
                  </button>
                ))}
              </div>

              {/* Victim tab content */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={victimTab}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                >
                  {victimTab === 'alibi'        && <AlibiTab />}
                  {victimTab === 'takedown'     && <TakedownTab result={result} />}
                  {victimTab === 'blast-radius' && <BlastRadiusTab />}
                  {victimTab === 'crisis'       && <CrisisTab onPanic={setPanicMode} />}
                </motion.div>
              </AnimatePresence>
            </div>
          )}
        </div>
      </main>

      {/* â”€â”€ Settings Drawer â”€â”€ */}
      <SettingsPanel
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        lang={lang}
        setLang={setLang}
        history={history}
        userEmail=""
        onLogout={handleLogout}
      />

      {/* Close lang menu on outside click */}
      {showLangMenu && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowLangMenu(false)}
        />
      )}
    </div>
  );
}

