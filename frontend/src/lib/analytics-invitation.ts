import { fetchAnalyticsIdentity } from './analytics';

const ANALYTICS_SEEN_KEY = 'openjarvis-analytics-choice-seen';

interface AnalyticsInvitation {
  ready: boolean;
  sharingModalSeen: boolean;
  open: () => void;
  markSharingModalSeen: () => void;
}

/** Offer the sharing dialog only after the app can actually display it. */
export function offerAnalyticsChoice({
  ready, sharingModalSeen, open, markSharingModalSeen,
}: AnalyticsInvitation): (() => void) | undefined {
  if (!ready) return;
  if (sharingModalSeen && localStorage.getItem(ANALYTICS_SEEN_KEY)) return;

  // Ask the server first: a local-only server cannot enable external
  // analytics, so the sharing dialog is never offered there.
  let active = true;
  fetchAnalyticsIdentity().then((identity) => {
    if (!active) return;
    const offer = identity.available !== false
      && (!sharingModalSeen || !identity.choice_recorded);
    if (offer) open();
    if (!sharingModalSeen) markSharingModalSeen();
    localStorage.setItem(ANALYTICS_SEEN_KEY, 'true');
  }).catch(() => { /* Retry next launch if the server was unavailable. */ });
  return () => { active = false; };
}
