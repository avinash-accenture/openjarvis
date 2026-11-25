import { useEffect, useState } from 'react';
import { Newspaper, Play, Square, ChevronDown, ChevronUp } from 'lucide-react';
import { fetchBriefingAudio, fetchTodayBriefing, type TodayBriefing } from '../../lib/api';
import { useTtsStore } from '../../lib/tts';

const BRIEFING_ID = 'morning-briefing';

/** Today's morning briefing: play it in the configured voice or read it. */
export function BriefingCard() {
  const [briefing, setBriefing] = useState<TodayBriefing | null>(null);
  const [open, setOpen] = useState(false);
  const state = useTtsStore((s) => s.state);
  const speakingId = useTtsStore((s) => s.speakingId);
  const error = useTtsStore((s) => (s.errorId === BRIEFING_ID ? s.error : null));
  const playAudio = useTtsStore((s) => s.playAudio);
  const stop = useTtsStore((s) => s.stop);

  useEffect(() => {
    let active = true;
    fetchTodayBriefing()
      .then((b) => { if (active) setBriefing(b); })
      .catch(() => { if (active) setBriefing(null); });
    return () => { active = false; };
  }, []);

  if (!briefing) return null;

  const busy = speakingId === BRIEFING_ID && state !== 'idle';
  const time = new Date(briefing.generated_at).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });

  return (
    <div
      className="w-full max-w-md mb-6 rounded-xl px-4 py-3 text-left"
      style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
      data-testid="briefing-card"
    >
      <div className="flex items-center gap-3">
        <Newspaper size={18} style={{ color: 'var(--color-accent)', flexShrink: 0 }} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium" style={{ color: 'var(--color-text)' }}>
            Today&apos;s briefing
          </div>
          <div className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
            Prepared at {time}
          </div>
        </div>
        {briefing.audio_available && (
          <button
            onClick={() => (busy ? stop() : playAudio(BRIEFING_ID, fetchBriefingAudio))}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer"
            style={{ background: 'var(--color-accent)', color: 'var(--color-on-accent)', border: 'none' }}
            aria-label={busy ? 'Stop the briefing' : 'Play the briefing'}
          >
            {busy ? <Square size={12} /> : <Play size={12} />}
            {busy ? (state === 'loading' ? 'Loading' : 'Stop') : 'Play'}
          </button>
        )}
        <button
          onClick={() => setOpen(!open)}
          className="p-1.5 rounded cursor-pointer"
          style={{ color: 'var(--color-text-tertiary)', background: 'transparent', border: 'none' }}
          aria-label={open ? 'Hide the briefing text' : 'Read the briefing'}
        >
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>
      {error && (
        <div role="alert" className="text-xs mt-2" style={{ color: 'var(--color-error)' }}>
          {error}
        </div>
      )}
      {open && (
        <p className="text-sm mt-3 whitespace-pre-line" style={{ color: 'var(--color-text-secondary)' }}>
          {briefing.text}
        </p>
      )}
    </div>
  );
}
