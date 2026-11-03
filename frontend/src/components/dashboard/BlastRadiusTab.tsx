'use client';

import React, { useState } from 'react';
import {
  Globe, Search, CheckCircle2, ExternalLink, Copy, Shield, Plus, Trash2, Link2
} from 'lucide-react';
import { motion } from 'framer-motion';

interface SpreadEntry {
  id: string;
  url: string;
  platform: string;
  dateFound: string;
  risk: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'taken-down' | 'pending';
}

const RISK_STYLES: Record<string, string> = {
  critical: 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20',
  high: 'bg-brand-orange/10 text-brand-orange border-brand-orange/20',
  medium: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20',
  low: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
};

const STATUS_STYLES: Record<string, string> = {
  active: 'bg-red-500/10 text-red-600 dark:text-red-400',
  'taken-down': 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
  pending: 'bg-amber-500/10 text-amber-600 dark:text-amber-400',
};

const SEARCH_TOOLS = [
  {
    name: 'StopNCII.org (Hash Block)',
    desc: 'Generate a device-side hash to block intimate imagery across Meta, TikTok, Reddit & Bumble without uploading the image.',
    url: 'https://stopncii.org/',
    badge: 'Recommended First Step',
  },
  {
    name: 'Google Lens / Images',
    desc: 'Check for visual matches and indexed copies across Google Search.',
    url: 'https://images.google.com/',
    badge: 'Visual Search',
  },
  {
    name: 'TinEye Reverse Search',
    desc: 'Search for exact and cropped image matches across the web.',
    url: 'https://tineye.com/',
    badge: 'Exact Match',
  },
  {
    name: 'Bing Visual Search',
    desc: 'Locate reposted copies indexed by Microsoft Bing.',
    url: 'https://www.bing.com/visualsearch',
    badge: 'Visual Search',
  },
];

export const BlastRadiusTab = () => {
  const [results, setResults] = useState<SpreadEntry[]>([]);
  const [newUrl, setNewUrl] = useState('');
  const [newPlatform, setNewPlatform] = useState('X (Twitter)');
  const [newRisk, setNewRisk] = useState<'critical' | 'high' | 'medium' | 'low'>('high');
  const [copied, setCopied] = useState<string | null>(null);

  const addDiscoveredUrl = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newUrl.trim()) return;
    setResults((prev) => [
      {
        id: `url-${Date.now()}`,
        url: newUrl.trim(),
        platform: newPlatform,
        dateFound: new Date().toLocaleString(),
        risk: newRisk,
        status: 'active',
      },
      ...prev,
    ]);
    setNewUrl('');
  };

  const toggleStatus = (id: string) => {
    setResults((prev) =>
      prev.map((r) =>
        r.id === id
          ? { ...r, status: r.status === 'active' ? 'taken-down' : 'active' }
          : r
      )
    );
  };

  const removeEntry = (id: string) => {
    setResults((prev) => prev.filter((r) => r.id !== id));
  };

  const copyUrl = (url: string, id: string) => {
    navigator.clipboard.writeText(url);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const activeCount = results.filter((r) => r.status === 'active').length;
  const takenDownCount = results.filter((r) => r.status === 'taken-down').length;
  const containment = results.length > 0 ? Math.round((takenDownCount / results.length) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Reverse Search & Hash Protection Hub */}
      <div className="rounded-3xl border border-border bg-card p-8 shadow-soft">
        <div className="flex items-center gap-3 mb-6">
          <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-amber-500 to-red-600 flex items-center justify-center shadow-lg">
            <Globe className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="font-display text-lg font-semibold">Blast Radius Containment & Discovery</h2>
            <p className="text-xs text-muted-foreground">
              Use trusted reverse-image engines and hash-blocking networks, then log discovered URLs below to track removal.
            </p>
          </div>
        </div>

        <div className="grid sm:grid-cols-2 gap-3">
          {SEARCH_TOOLS.map((tool) => (
            <a
              key={tool.name}
              href={tool.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col justify-between rounded-2xl border border-border bg-muted/20 p-4 hover:border-brand-orange/40 hover:bg-muted/40 transition-all group"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <span className="font-display font-semibold text-sm group-hover:text-brand-orange transition-colors">
                    {tool.name}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-brand-orange/10 text-brand-orange">
                    {tool.badge}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{tool.desc}</p>
              </div>
              <div className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-foreground group-hover:text-brand-orange">
                Open Tool <ExternalLink className="h-3 w-3" />
              </div>
            </a>
          ))}
        </div>
      </div>

      {/* Log Discovered URLs */}
      <div className="rounded-3xl border border-border bg-card p-8 shadow-soft">
        <h3 className="font-display text-lg font-semibold mb-1">Containment URL Log</h3>
        <p className="text-xs text-muted-foreground mb-5">
          Record any URLs where the manipulated media appears so you can track takedown requests systematically.
        </p>

        <form onSubmit={addDiscoveredUrl} className="grid sm:grid-cols-[1fr_160px_130px_auto] gap-3 mb-6">
          <input
            type="url"
            required
            value={newUrl}
            onChange={(e) => setNewUrl(e.target.value)}
            placeholder="https://platform.com/post/..."
            className="px-4 py-2.5 rounded-xl bg-muted/40 border border-border text-sm focus:outline-none focus:ring-2 focus:ring-brand-orange/40"
          />
          <select
            value={newPlatform}
            onChange={(e) => setNewPlatform(e.target.value)}
            className="px-3 py-2.5 rounded-xl bg-muted/40 border border-border text-sm focus:outline-none"
          >
            <option>X (Twitter)</option>
            <option>Instagram / Facebook</option>
            <option>TikTok</option>
            <option>YouTube</option>
            <option>Reddit</option>
            <option>Telegram</option>
            <option>Other Website</option>
          </select>
          <select
            value={newRisk}
            onChange={(e) => setNewRisk(e.target.value as any)}
            className="px-3 py-2.5 rounded-xl bg-muted/40 border border-border text-sm focus:outline-none"
          >
            <option value="critical">Critical Risk</option>
            <option value="high">High Risk</option>
            <option value="medium">Medium Risk</option>
            <option value="low">Low Risk</option>
          </select>
          <button
            type="submit"
            className="inline-flex items-center justify-center gap-1.5 px-5 py-2.5 rounded-xl bg-foreground text-background text-sm font-semibold hover:bg-foreground/90 transition-colors"
          >
            <Plus className="h-4 w-4" /> Log URL
          </button>
        </form>

        {results.length > 0 && (
          <div className="mb-6 grid grid-cols-4 gap-3">
            {[
              { val: results.length, label: 'Logged', cls: 'text-brand-orange' },
              { val: activeCount, label: 'Active', cls: 'text-red-500' },
              { val: takenDownCount, label: 'Removed', cls: 'text-emerald-500' },
              { val: `${containment}%`, label: 'Contained', cls: 'text-brand-purple' },
            ].map((s) => (
              <div key={s.label} className="rounded-2xl border border-border bg-muted/30 p-3 text-center">
                <p className={`text-2xl font-display font-bold ${s.cls}`}>{s.val}</p>
                <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">{s.label}</p>
              </div>
            ))}
          </div>
        )}

        {results.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-border rounded-2xl bg-muted/10">
            <Link2 className="h-8 w-8 mx-auto text-muted-foreground/40 mb-2" />
            <p className="text-sm text-muted-foreground">No URLs logged yet. Add any links you need to track for takedown above.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {results.map((r, i) => (
              <motion.div
                key={r.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
                className={`rounded-2xl border p-4 transition-all ${
                  r.status === 'taken-down'
                    ? 'border-emerald-500/20 bg-emerald-500/5 opacity-75'
                    : 'border-border bg-muted/20'
                }`}
              >
                <div className="flex items-start justify-between gap-4 flex-wrap">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <span className="text-sm font-medium">{r.platform}</span>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold border ${RISK_STYLES[r.risk]}`}>
                        {r.risk.toUpperCase()}
                      </span>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold ${STATUS_STYLES[r.status]}`}>
                        {r.status === 'taken-down' ? '✓ Removed' : '● Active'}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground font-mono truncate">{r.url}</p>
                    <p className="text-[11px] text-muted-foreground mt-1">Logged: {r.dateFound}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={() => copyUrl(r.url, r.id)}
                      className="inline-flex items-center gap-1 px-3 py-1.5 rounded-xl border border-border text-xs font-medium hover:bg-muted transition-colors"
                    >
                      {copied === r.id ? <><CheckCircle2 className="h-3 w-3 text-emerald-500" /> Copied</> : <><Copy className="h-3 w-3" /> Copy</>}
                    </button>
                    <button
                      onClick={() => toggleStatus(r.id)}
                      className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-xl text-xs font-medium transition-colors ${
                        r.status === 'taken-down'
                          ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
                          : 'bg-brand-orange text-white hover:bg-brand-orange/90'
                      }`}
                    >
                      {r.status === 'taken-down' ? <><CheckCircle2 className="h-3 w-3" /> Done</> : <><Shield className="h-3 w-3" /> Mark Removed</>}
                    </button>
                    <button
                      onClick={() => removeEntry(r.id)}
                      className="p-1.5 rounded-xl text-muted-foreground hover:text-destructive transition-colors"
                      title="Delete entry"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
