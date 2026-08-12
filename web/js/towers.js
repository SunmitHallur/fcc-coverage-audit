/** Tower panel helpers — keep ASR ground truth distinct from inferred sites. */

import { TOWER_COLORS } from './colors.js';

export { TOWER_COLORS };

export function towerColor(siteClass) {
  return TOWER_COLORS[siteClass] || TOWER_COLORS.default;
}

export function formatTowerPanel(rec) {
  const m = rec?.metrics || {};
  const prior = m.prior_towers ?? null;
  const current = m.current_towers ?? null;
  const asrCount = m.asr_new_structure_count;
  const asrHas = m.asr_has_new_structure;
  return {
    prior,
    current,
    newTowers: m.new_towers ?? 0,
    inferredNote:
      'Inferred from coverage shape — approximate. Prefer ASR registered structures when available.',
    asrLine:
      asrHas == null
        ? null
        : asrHas
          ? `ASR: ${asrCount || 0} registered structure(s) in this county during the window`
          : 'ASR: no new registered structure in this county during the window',
  };
}
