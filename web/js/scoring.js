/** Flag-math panel: show score contributions when present. */

export function renderFlagMathInto(panelEls, fm) {
  const {
    panel, scoreFill, threshLine, scoreLabel, badge, list,
  } = panelEls;
  if (!fm) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = '';
  const score = fm.priority_score ?? 0;
  const thresh = fm.flag_threshold ?? 0;
  scoreFill.style.width = `${Math.min(100, score * 100)}%`;
  if (thresh > 0.001) {
    threshLine.style.display = '';
    threshLine.style.left = `${Math.min(99, thresh * 100)}%`;
  } else {
    threshLine.style.display = 'none';
  }
  scoreLabel.textContent = `${score.toFixed(3)} vs ${thresh.toFixed(3)} cutoff`;
  badge.textContent = fm.flag ? 'Flagged' : 'Not flagged';
  badge.className = `fm-status ${fm.flag ? 'flagged' : 'not-flagged'}`;

  list.innerHTML = '';
  const feats = fm.features || [];
  const useContrib = feats.some(f => f.contribution != null && Number(f.contribution) !== 0);
  const maxContrib = useContrib
    ? Math.max(...feats.map(f => Math.abs(Number(f.contribution) || 0)), 1e-9)
    : 1;
  feats.forEach(f => {
    const contrib = f.contribution != null ? Number(f.contribution) : null;
    if (useContrib && (contrib == null || contrib === 0) && !f.value) return;
    if (!useContrib && (f.value == null || f.value === 0)) return;
    const row = document.createElement('div');
    row.className = 'fm-feat-row';
    const isBool = typeof f.value === 'boolean';
    const rawVal = isBool ? (f.value ? 'yes' : 'no')
      : (f.value == null ? '—' : Number(f.value).toFixed(2));
    const shown = useContrib && contrib != null
      ? `${contrib >= 0 ? '+' : ''}${contrib.toFixed(3)}`
      : rawVal;
    const level = useContrib && contrib != null
      ? Math.abs(contrib) / maxContrib
      : (isBool ? (f.value ? 1 : 0) : Math.max(0, Math.min(1, Number(f.value) || 0)));
    const barPct = Math.round(level * 100);
    const hi = level >= 0.5 ? ' hi' : '';
    row.innerHTML = `
      <span class="fm-feat-name">${f.label}</span>
      <span class="fm-feat-val" title="raw=${rawVal}">${shown}</span>
      <div class="fm-feat-bar"><div class="fm-feat-fill${hi}" style="width:${barPct}%"></div></div>
    `;
    list.appendChild(row);
  });
}
