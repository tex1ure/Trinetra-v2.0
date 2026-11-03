'use client';

import React, { useState } from 'react';
import {
  MapPin, Clock, Plus, Trash2, Download, CheckCircle2, X, FileCheck2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface AlibiEvent {
  id: string;
  time: string;
  title: string;
  location: string;
  source: string;
}

export const AlibiTab = () => {
  const [incidentDate, setIncidentDate] = useState('');
  const [incidentTime, setIncidentTime] = useState('');
  const [events, setEvents] = useState<AlibiEvent[]>([]);
  const [showAddForm, setShowAddForm] = useState(true);
  const [newEvent, setNewEvent] = useState({ time: '', title: '', location: '', source: 'Location / Receipt / Log' });

  const addManualEvent = () => {
    if (!newEvent.time || !newEvent.title) return;
    setEvents((prev) =>
      [
        ...prev,
        {
          id: `entry-${Date.now()}`,
          time: newEvent.time,
          title: newEvent.title,
          location: newEvent.location || 'Not specified',
          source: newEvent.source || 'User Record',
        },
      ].sort((a, b) => a.time.localeCompare(b.time))
    );
    setNewEvent({ time: '', title: '', location: '', source: 'Location / Receipt / Log' });
  };

  const exportAlibi = () => {
    const report = {
      alibi_report: {
        generated_by: 'Trinetra Deepfake Detection Platform',
        generated_at: new Date().toISOString(),
        incident_date: incidentDate,
        incident_time: incidentTime,
        total_events: events.length,
        timeline: events,
      },
      disclaimer:
        'This timeline was compiled by the user to document corroborating evidence (receipts, location history, calendar logs, witnesses) for the specified date and time.',
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trinetra-alibi-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-border bg-card p-8 shadow-soft">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
              <MapPin className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="font-display text-lg font-semibold">Digital Alibi & Evidence Log</h2>
              <p className="text-xs text-muted-foreground">
                Document your real-world timeline (receipts, transit logs, Wi-Fi connections, witnesses) to refute fabricated media.
              </p>
            </div>
          </div>
          {events.length > 0 && (
            <button
              onClick={exportAlibi}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-foreground text-background hover:bg-foreground/90 text-sm font-medium transition-colors"
            >
              <Download className="h-4 w-4" /> Export Alibi JSON
            </button>
          )}
        </div>

        <div className="mt-6 grid sm:grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-muted-foreground font-semibold uppercase tracking-widest block mb-2">
              Alleged Incident Date
            </label>
            <input
              type="date"
              value={incidentDate}
              onChange={(e) => setIncidentDate(e.target.value)}
              className="w-full px-4 py-3 rounded-2xl bg-muted/40 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500/40 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground font-semibold uppercase tracking-widest block mb-2">
              Alleged Incident Time
            </label>
            <input
              type="time"
              value={incidentTime}
              onChange={(e) => setIncidentTime(e.target.value)}
              className="w-full px-4 py-3 rounded-2xl bg-muted/40 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500/40 text-sm"
            />
          </div>
        </div>
      </div>

      <div className="rounded-3xl border border-border bg-card p-8 shadow-soft">
        <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
          <div>
            <h3 className="font-display text-lg font-semibold">Corroborating Timeline</h3>
            <p className="text-xs text-muted-foreground">
              Add verifiable events for {incidentDate || 'the incident date'}
            </p>
          </div>
          <button
            onClick={() => setShowAddForm((v) => !v)}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full border border-border bg-muted/40 hover:bg-muted text-sm font-medium transition-colors"
          >
            <Plus className="h-4 w-4" /> {showAddForm ? 'Hide Form' : 'Add Entry'}
          </button>
        </div>

        <AnimatePresence>
          {showAddForm && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="rounded-2xl border border-blue-500/30 bg-blue-500/5 p-5 mb-6">
                <div className="grid sm:grid-cols-4 gap-3">
                  <input
                    type="time"
                    value={newEvent.time}
                    onChange={(e) => setNewEvent((p) => ({ ...p, time: e.target.value }))}
                    className="px-3 py-2.5 rounded-xl bg-background border border-border text-sm focus:outline-none"
                  />
                  <input
                    type="text"
                    value={newEvent.title}
                    onChange={(e) => setNewEvent((p) => ({ ...p, title: e.target.value }))}
                    placeholder="Activity (e.g. In office meeting)"
                    className="px-3 py-2.5 rounded-xl bg-background border border-border text-sm focus:outline-none"
                  />
                  <input
                    type="text"
                    value={newEvent.location}
                    onChange={(e) => setNewEvent((p) => ({ ...p, location: e.target.value }))}
                    placeholder="Location"
                    className="px-3 py-2.5 rounded-xl bg-background border border-border text-sm focus:outline-none"
                  />
                  <input
                    type="text"
                    value={newEvent.source}
                    onChange={(e) => setNewEvent((p) => ({ ...p, source: e.target.value }))}
                    placeholder="Proof (e.g. UPI Receipt, Badge Log)"
                    className="px-3 py-2.5 rounded-xl bg-background border border-border text-sm focus:outline-none"
                  />
                </div>
                <button
                  onClick={addManualEvent}
                  className="mt-3 px-4 py-2 rounded-xl bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 transition-colors"
                >
                  Add to Timeline
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {events.length === 0 ? (
          <div className="text-center py-12">
            <Clock className="h-10 w-10 mx-auto text-muted-foreground/40 mb-3" />
            <p className="text-sm text-muted-foreground">
              No timeline entries added yet. Record your real-world activities and supporting proof above.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {events.map((evt) => (
              <div
                key={evt.id}
                className="flex items-start justify-between gap-4 rounded-2xl border border-border bg-muted/20 p-4"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-blue-600 dark:text-blue-400 font-bold">{evt.time}</span>
                    <span className="inline-flex items-center gap-1 rounded-full bg-blue-500/10 px-2 py-0.5 text-[10px] font-semibold text-blue-600 dark:text-blue-400">
                      <FileCheck2 className="h-3 w-3" /> {evt.source}
                    </span>
                  </div>
                  <p className="text-sm font-medium mt-1">{evt.title}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{evt.location}</p>
                </div>
                <button
                  onClick={() => setEvents((p) => p.filter((e) => e.id !== evt.id))}
                  className="text-muted-foreground hover:text-destructive transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
