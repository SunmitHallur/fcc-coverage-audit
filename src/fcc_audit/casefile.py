"""Auto-generate per-county case files in the consultant-memo format.

Structure: Claim → Evidence → Contradiction → Recommendation

Each case file is written as Markdown (and optionally PDF via weasyprint).
Inputs: the scored + explained DataFrame produced by the pipeline, plus
optional detail JSON (as produced by webbundle.py) for hex-level context.
"""
from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

def _pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"

def _km2(v: Any) -> str:
    try:
        f = float(v)
        return f"{f:.1f} km²" if f >= 10 else f"{f:.2f} km²"
    except (TypeError, ValueError):
        return "N/A"

def _fmt_score(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "N/A"

def _yesno(v: Any) -> str:
    if v is None:
        return "unknown"
    try:
        return "yes" if bool(v) else "no"
    except Exception:
        return "unknown"

def _severity_color(sev: str) -> str:
    return {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}.get(sev, "⚪")


# ── core case-file renderer ───────────────────────────────────────────────────

def render_case_file(
    row: pd.Series | dict[str, Any],
    detail: dict[str, Any] | None = None,
    provider_name: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Return full Markdown text for one county case file.

    Parameters
    ----------
    row:
        A single row from the scored + explained DataFrame (or plain dict).
    detail:
        Optional detail JSON produced by webbundle.py for hex-level context.
    provider_name:
        Human-readable carrier name if available (defaults to provider_id).
    generated_at:
        Timestamp to stamp into the header (defaults to now).
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()

    now = (generated_at or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")

    # ── unpack fields ─────────────────────────────────────────────────────────
    geoid = row.get("geoid", "unknown")
    county_name = row.get("name", geoid)
    pid = row.get("provider_id", "")
    carrier = provider_name or pid
    service = row.get("service", "unknown")
    flag = bool(row.get("flag", False))
    severity = row.get("severity", "Low")
    priority = row.get("priority_score", row.get("priority", 0.0))

    added_km2 = row.get("added_km2", None)
    added_frac = row.get("added_frac_of_county", None)
    same_site = row.get("same_site_growth_share", None)
    blanket = row.get("blanket_fillin", None)
    unattr = row.get("unattributed_share", None)
    boundary = row.get("boundary_snap_share", None)

    asr_no_struct = row.get("asr_no_new_structure", None)
    meas_gap = row.get("measurement_gap", None)

    prior_towers = row.get("prior_towers", detail.get("towers_prior") if detail else None)
    current_towers = row.get("current_towers", detail.get("towers_current") if detail else None)

    prior_vintage = row.get("prior_vintage", detail.get("prior_vintage") if detail else None)
    current_vintage = row.get("current_vintage", detail.get("current_vintage") if detail else None)

    reason = row.get("flag_reason", "ranked by composite anomaly score")

    explanation: dict[str, Any] = {}
    try:
        explanation = json.loads(row["explanation_json"]) if "explanation_json" in row else {}
    except Exception:
        pass
    headline = explanation.get("headline", "")
    bullets = explanation.get("bullets", [])
    recommendation = explanation.get("recommendation", "")

    fm = row.get("flag_math") or {}

    # ── document ─────────────────────────────────────────────────────────────
    sev_icon = _severity_color(severity)
    flag_line = f"**FLAGGED FOR REVIEW** {sev_icon} {severity}" if flag else f"Not flagged {sev_icon} {severity}"

    lines: list[str] = []

    # Title block
    lines += [
        f"# Case File: {county_name} — {carrier} — {service}",
        "",
        f"> Generated: {now}  ",
        f"> Status: {flag_line}  ",
        f"> Priority score: `{_fmt_score(priority)}`  ",
        f"> Geoid: `{geoid}` | Carrier ID: `{pid}` | Service: `{service}`",
        "",
        "---",
        "",
    ]

    # ── 1. CLAIM ─────────────────────────────────────────────────────────────
    lines += [
        "## 1. Claim",
        "",
        f"**{carrier}** filed an FCC Broadband Data Collection ({service}) coverage change "
        f"showing coverage area growth in **{county_name}** "
        + (f"(vintage window: {prior_vintage} → {current_vintage})." if prior_vintage and current_vintage else "."),
        "",
    ]

    claim_rows = [
        ("Added coverage area", _km2(added_km2)),
        ("Share of county area added", _pct(added_frac)),
        ("Prior coverage towers (estimated)", str(int(prior_towers)) if prior_towers is not None else "N/A"),
        ("Current coverage towers (estimated)", str(int(current_towers)) if current_towers is not None else "N/A"),
    ]
    lines += _table(["Metric", "Value"], claim_rows)
    lines.append("")

    # ── 2. EVIDENCE ──────────────────────────────────────────────────────────
    lines += [
        "## 2. Evidence",
        "",
        "### 2a. Attribution of growth",
        "",
        "The pipeline attributes each gained coverage hexagon to either an "
        "existing site or a newly inferred site.",
        "",
    ]

    evid_rows = [
        ("Same-site growth share", _pct(same_site), "Growth attributed to existing towers"),
        ("Blanket fill-in score", _pct(blanket), "Uniform coverage appearing simultaneously"),
        ("Unattributed growth share", _pct(unattr), "Hexes not attributable to any inferred site"),
        ("Boundary snap share", _pct(boundary), "Hexes near county edge (border effects)"),
    ]
    lines += _table(["Signal", "Value", "Description"], evid_rows)
    lines.append("")

    if bullets:
        lines += ["### 2b. Analytical bullets", ""]
        for b in bullets:
            lines.append(f"- {b}")
        lines.append("")

    # ── 3. CONTRADICTION (ground truth) ──────────────────────────────────────
    lines += [
        "## 3. Contradiction / Ground Truth",
        "",
        "Independent, non-carrier data sources that can corroborate or contradict the claim:",
        "",
    ]

    gt_rows = []
    if asr_no_struct is not None:
        gt_label = "No new FCC-registered tower structure found in county during the vintage window"
        gt_val = "YES ⚠️" if float(asr_no_struct) >= 0.5 else "No (structure found — consistent with growth)"
        gt_rows.append(("FCC ASR: no new structure", gt_val, gt_label))
    else:
        gt_rows.append(("FCC ASR: no new structure", "data not loaded", "Run groundtruth_asr enrichment"))

    if meas_gap is not None:
        gap_pct = _pct(meas_gap)
        gap_label = "Fraction of claimed area not corroborated by field measurements"
        gap_icon = "⚠️" if float(meas_gap) >= 0.2 else "✓"
        gt_rows.append(("Measured coverage gap", f"{gap_pct} {gap_icon}", gap_label))
    else:
        gt_rows.append(("Measured coverage gap", "data not loaded", "Run groundtruth_measured enrichment"))

    lines += _table(["Source", "Finding", "Notes"], gt_rows)
    lines.append("")

    if asr_no_struct is not None and float(asr_no_struct) >= 0.5:
        lines += [
            "> ⚠️ **Note:** The FCC Antenna Structure Registration (ASR) database shows no new antenna structures "
            "were registered in this county during the relevant window. This is inconsistent with a claim that "
            "new physical towers drove the coverage expansion. The carrier may be claiming gains from power "
            "increases or software changes to existing towers — or the filing may be inaccurate.",
            "",
        ]

    if meas_gap is not None and float(meas_gap) >= 0.2:
        lines += [
            f"> ⚠️ **Note:** Independent speed-test/field-measurement data (Ookla Open Data) shows "
            f"approximately {_pct(meas_gap)} of the claimed coverage area had no real-world measurements "
            "recorded in the period. Coverage claimed on paper but not observed in field data is a "
            "characteristic pattern of inflated filings.",
            "",
        ]

    # ── 4. RECOMMENDATION ────────────────────────────────────────────────────
    lines += [
        "## 4. Recommendation",
        "",
    ]

    if flag:
        lines += [
            f"**{sev_icon} {severity} priority — recommend drive-test verification.**",
            "",
            recommendation or (
                f"Schedule a drive test in {county_name} to verify {carrier}'s {service} coverage claims. "
                "Focus on hexes attributed to same-site growth and the unattributed fringe area."
            ),
            "",
            f"Primary flag reason: *{reason}*",
            "",
        ]
    else:
        lines += [
            "**No drive-test action required at this time.**",
            "",
            recommendation or (
                f"The coverage change in {county_name} ({carrier} / {service}) appears consistent with "
                "normal organic growth. No anomalous patterns were detected above the flag threshold."
            ),
            "",
        ]

    # ── 5. SCORING DETAIL ────────────────────────────────────────────────────
    if fm and fm.get("features"):
        lines += [
            "## 5. Scoring Detail",
            "",
            f"Priority score: **{_fmt_score(fm.get('priority_score'))}** "
            f"(flag threshold: {_fmt_score(fm.get('flag_threshold'))})",
            "",
        ]
        feat_rows = []
        for f in fm["features"]:
            val = f.get("value")
            if val is None:
                continue
            val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
            feat_rows.append((
                f.get("label", f.get("feature", "?")),
                val_str,
                str(f.get("weight", "")),
                f"{f.get('contribution', 0):.4f}",
            ))
        if feat_rows:
            lines += _table(["Feature", "Value", "Weight", "Contribution"], feat_rows)
            lines.append("")

    lines += [
        "---",
        f"*Auto-generated by fcc-coverage-audit pipeline. Do not edit manually. "
        f"Re-generate with: `python -m fcc_audit case-files --geoid {geoid}`*",
    ]

    return "\n".join(lines)


def _table(headers: list[str], rows: list[tuple[str, ...]]) -> list[str]:
    """Return GFM table lines."""
    widths = [len(h) for h in headers]
    str_rows = [tuple(str(c) for c in r) for r in rows]
    for r in str_rows:
        for i, c in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))

    def fmt_row(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [fmt_row(tuple(headers))]
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    for r in str_rows:
        out.append(fmt_row(r))
    return out


# ── batch generation ─────────────────────────────────────────────────────────

def generate_case_files(
    scored: pd.DataFrame,
    out_dir: Path,
    *,
    flagged_only: bool = True,
    provider_names: dict[int, str] | None = None,
    detail_loader: Any | None = None,
    to_pdf: bool = False,
    llm_backend: Any | None = None,
) -> list[Path]:
    """Write one Markdown case file per county row in *scored*.

    Parameters
    ----------
    scored:
        The full scored + explained DataFrame (must have ``explanation_json``).
    out_dir:
        Directory to write case files into.
    flagged_only:
        If True (default), only write files for rows where ``flag`` is True.
    provider_names:
        Optional mapping from provider_id (int) -> human-readable name.
    detail_loader:
        Optional callable ``(provider_id, service, geoid) -> dict | None``
        that loads the detail JSON for a county (for hex-level context).
    to_pdf:
        If True, convert each Markdown to PDF via weasyprint (requires it
        installed; silently skipped if unavailable).

    Returns
    -------
    List of Paths to written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provider_names = provider_names or {}

    rows = scored[scored["flag"].astype(bool)] if flagged_only else scored
    written: list[Path] = []

    for _, row in rows.iterrows():
        geoid = row.get("geoid", "unknown")
        pid = row.get("provider_id", "")
        svc = str(row.get("service", "unknown")).replace(" ", "_").lower()

        detail = None
        if detail_loader is not None:
            try:
                detail = detail_loader(pid, svc, geoid)
            except Exception:
                pass

        pname = provider_names.get(int(pid), None) if pid else None
        md = render_case_file(row, detail=detail, provider_name=pname)

        # Optional LLM pass to improve recommendation prose
        if llm_backend is not None:
            try:
                md = llm_backend.draft(md, dict(row))
            except Exception:
                pass

        slug = f"{geoid}_{pid}_{svc}"
        md_path = out_dir / f"{slug}.md"
        md_path.write_text(md, encoding="utf-8")
        written.append(md_path)

        if to_pdf:
            _try_write_pdf(md, md_path.with_suffix(".pdf"))

    return written


def _try_write_pdf(md_text: str, out_path: Path) -> None:
    """Convert markdown to PDF via weasyprint if available; silently skip."""
    try:
        import weasyprint  # type: ignore
        # Convert markdown to HTML first using mistune if available
        try:
            import mistune  # type: ignore
            body = mistune.create_markdown()(md_text)
        except ImportError:
            # Fallback: wrap pre-formatted text in minimal HTML
            escaped = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            body = f"<pre style='white-space:pre-wrap;font-family:sans-serif'>{escaped}</pre>"

        html = textwrap.dedent(f"""
            <!DOCTYPE html><html><head>
            <meta charset='utf-8'>
            <style>
              body {{ font-family: sans-serif; font-size: 11pt; max-width: 800px; margin: auto; padding: 2em; }}
              h1 {{ border-bottom: 2px solid #333; padding-bottom: 4px; }}
              h2 {{ border-bottom: 1px solid #aaa; margin-top: 1.5em; }}
              table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 9pt; }}
              th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
              th {{ background: #f0f0f0; }}
              blockquote {{ border-left: 4px solid #f59e0b; padding-left: 1em; color: #555; }}
              code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
              pre code {{ background: none; }}
            </style>
            </head><body>{body}</body></html>
        """)
        weasyprint.HTML(string=html).write_pdf(str(out_path))
    except Exception:
        pass


# ── CLI integration ───────────────────────────────────────────────────────────

def cmd_case_files(cfg: Any, args: Any) -> int:  # noqa: ANN401
    """CLI handler for the ``case-files`` subcommand."""
    import sys
    from pathlib import Path as P

    from fcc_audit import persist, explain
    from fcc_audit.llm_narrative import build_backend

    out_dir = P(getattr(args, "out_dir", None) or cfg.paths.get("outputs", "data/outputs")) / "case_files"
    flagged_only = not getattr(args, "all", False)
    to_pdf = getattr(args, "pdf", False)
    geoid_filter: str | None = getattr(args, "geoid", None)
    llm_name: str = getattr(args, "llm", "none") or "none"

    llm_backend = build_backend(
        llm_name,
        local_url=getattr(args, "llm_url", "http://localhost:11434"),
        local_model=getattr(args, "llm_model", "llama3"),
        gemini_api_key=getattr(args, "gemini_api_key", None),
    )

    scored = persist.load_accumulated_scored(cfg)
    if scored.empty:
        print("[case-files] No scored data found. Run the pipeline first.", file=sys.stderr)
        return 1

    if "explanation_json" not in scored.columns:
        scored = explain.add_explanations(scored)

    if geoid_filter:
        scored = scored[scored["geoid"].astype(str) == geoid_filter]
        if scored.empty:
            print(f"[case-files] No rows found for geoid={geoid_filter}", file=sys.stderr)
            return 1

    written = generate_case_files(
        scored,
        out_dir=out_dir,
        flagged_only=flagged_only,
        to_pdf=to_pdf,
        llm_backend=llm_backend if llm_name != "none" else None,
    )
    print(f"[case-files] Wrote {len(written)} case file(s) to {out_dir}")
    return 0
