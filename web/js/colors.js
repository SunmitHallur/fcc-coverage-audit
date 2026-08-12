/** Shared color scales for national choropleth, detail maps, and legend. */

/** Unified tower site_class colors (must match map_render.py). */
export const TOWER_COLORS = {
  new_site: '#16a34a',
  expanded_site: '#c2410c',
  prior_site: '#64748b',
  default: '#64748b',
};

/** True-ish RSRP stops: weak red → strong green (dBm, [r,g,b]). */
export const SIGNAL_STOPS = [
  [-120, [139, 0, 0]],
  [-110, [220, 38, 38]],
  [-100, [249, 115, 22]],
  [-90, [234, 179, 8]],
  [-80, [132, 204, 22]],
  [-70, [34, 197, 94]],
  [-60, [22, 163, 74]],
];

/** Flat binary coverage class (no real signal). */
export const FLAT_COVERAGE_COLOR = '#94a3b8';

export function signalColor(dbm) {
  const stops = SIGNAL_STOPS;
  if (dbm <= stops[0][0]) return `rgb(${stops[0][1].join(',')})`;
  if (dbm >= stops[stops.length - 1][0]) {
    return `rgb(${stops[stops.length - 1][1].join(',')})`;
  }
  for (let i = 0; i < stops.length - 1; i++) {
    const [d0, c0] = stops[i];
    const [d1, c1] = stops[i + 1];
    if (dbm >= d0 && dbm <= d1) {
      const t = (dbm - d0) / (d1 - d0);
      const rgb = c0.map((v, j) => Math.round(v + t * (c1[j] - v)));
      return `rgb(${rgb.join(',')})`;
    }
  }
  return FLAT_COVERAGE_COLOR;
}

/** National map fill: flagged = red intensity; unflagged = slate→blue by priority. */
export function colorForRecord(rec) {
  if (!rec) return '#334155';
  const p = Math.max(0, Math.min(1, Number(rec.priority) || 0));
  if (rec.flag) {
    const t = 0.35 + 0.65 * p;
    const r = 239;
    const g = Math.round(68 + (1 - t) * 100);
    const b = Math.round(68 + (1 - t) * 100);
    return `rgb(${r},${g},${b})`;
  }
  const t = 0.25 + 0.55 * p;
  const r = Math.round(51 + (1 - t) * 40);
  const g = Math.round(65 + (1 - t) * 50);
  const b = Math.round(85 + t * 80);
  return `rgb(${r},${g},${b})`;
}

export function severityPillClass(label) {
  const s = String(label || '').toLowerCase();
  if (s.includes('top 1')) return 'sev-critical';
  if (s.includes('top 5')) return 'sev-high';
  if (s.includes('top 10') || s.includes('flag')) return 'sev-moderate';
  return 'sev-low';
}
