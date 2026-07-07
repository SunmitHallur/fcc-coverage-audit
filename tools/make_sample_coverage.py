"""Build a single self-contained HTML with per-county coverage hexes.

Reads the saved coverage snapshot parquet (data/processed/coverage/*.parquet)
memory-safely -- filtered to ONE provider + service and a handful of counties --
and emits a standalone `sample_coverage.html` with the hex data embedded inline.
Engineers can just double-click the file (it pulls MapLibre + h3-js from CDN, so
it needs internet, but no local server and no Python). This is the same coverage
overlay the main web app draws on county click, packaged for sharing without the
full 650M-row bundle that OOMs `build-web`.

Examples (run from the repo root, venv active):

    # Auto-pick the top flagged AT&T 5G 7/1 counties (uses the scored parquet)
    python tools/make_sample_coverage.py

    # A specific state, more counties
    python tools/make_sample_coverage.py --state 48 --max-counties 10

    # Exact counties, a different provider/service
    python tools/make_sample_coverage.py --provider 131425 --service "5G-NR 7/1" \
        --counties 48201,48113,48085
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("make_sample_coverage")


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "config" / "pipeline.yaml").exists():
            return parent
    return here.parents[1]


def _pick_counties(
    scored_dir: Path,
    provider_id: int,
    service: str,
    state: str | None,
    explicit: list[str] | None,
    max_counties: int,
) -> tuple[list[str], dict[str, dict]]:
    """Return (geoids, per-geoid metadata) chosen for the sample.

    Prefers explicit --counties; otherwise ranks the scored parquet by
    flagged-then-priority so the sample shows the most interesting counties.
    """
    import pandas as pd

    meta: dict[str, dict] = {}
    labels = {"prior": "Prior", "current": "Current"}
    parts = sorted(glob.glob(str(scored_dir / "scored_*.parquet")))
    scored = pd.DataFrame()
    if parts:
        scored = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        scored = scored[scored["county_geoid"].astype(str).str.len() == 5]
        scored["county_geoid"] = scored["county_geoid"].astype(str)
        s = scored[
            (scored["provider_id"] == provider_id)
            & (scored["technology"].astype(str) == service)
        ].copy()
        if state:
            s = s[s["county_geoid"].str.startswith(state)]
        if not s.empty:
            if "batch_prior" in s.columns and s["batch_prior"].notna().any():
                labels["prior"] = str(s["batch_prior"].dropna().iloc[-1])
            if "batch_current" in s.columns and s["batch_current"].notna().any():
                labels["current"] = str(s["batch_current"].dropna().iloc[-1])
        for _, r in s.iterrows():
            meta[r["county_geoid"]] = {
                "name": str(r.get("county_name", r["county_geoid"])),
                "flagged": bool(r.get("flag_for_review", False)),
                "priority": float(r.get("priority_score", 0.0) or 0.0),
                "added_km2": float(r.get("added_km2", 0.0) or 0.0),
                "pct_increase": float(r.get("pct_increase", 0.0) or 0.0),
                "prior_towers": int(r.get("prior_towers", 0) or 0),
                "current_towers": int(r.get("current_towers", 0) or 0),
                "new_towers": int(r.get("new_towers", 0) or 0),
                "explanation": str(r.get("plain_explanation", "") or ""),
            }

    if explicit:
        geoids = [g.strip() for g in explicit if g.strip()]
        return geoids, meta, labels

    if not meta:
        raise SystemExit(
            "No scored data found to auto-pick counties. Pass --counties "
            "48201,48113 explicitly, or point --scored-dir at your scored parquet."
        )

    ranked = sorted(
        meta.items(),
        key=lambda kv: (kv[1]["flagged"], kv[1]["priority"]),
        reverse=True,
    )
    return [g for g, _ in ranked[:max_counties]], meta, labels


def _load_hexes(
    coverage_path: Path,
    provider_id: int,
    service: str,
    geoids: list[str],
) -> dict[str, dict[str, list]]:
    """Stream the coverage parquet, keeping only rows for the chosen keys.

    Uses a pushed-down pyarrow filter so we never materialize the full table
    (which can be hundreds of millions of rows nationwide)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(coverage_path), format="parquet")
    flt = (
        (pc.field("provider_id") == provider_id)
        & (pc.field("technology") == service)
        & (pc.field("county_geoid").isin(geoids))
    )
    table = dataset.to_table(
        columns=["h3", "signal_dbm", "county_geoid", "vintage"], filter=flt
    )
    log.info("read %s matching coverage rows", f"{table.num_rows:,}")

    # The pipeline tags coverage rows with vintage = 'prior' / 'current'
    # (see cli._analyze_unit), so bucket deterministically on that.
    out: dict[str, dict[str, list]] = {g: {"prior": [], "current": []} for g in geoids}
    geoid_col = table.column("county_geoid").to_pylist()
    h3_col = table.column("h3").to_pylist()
    sig_col = table.column("signal_dbm").to_pylist()
    vint_col = table.column("vintage").to_pylist()
    for g, h, s, v in zip(geoid_col, h3_col, sig_col, vint_col):
        bucket = "current" if str(v) == "current" else "prior"
        # Encode signal to compact int8 the same way the web bundle does:
        # encoded = round(dBm) + 100 (JS decodeSignal reverses it).
        try:
            enc = int(round(float(s))) + 100
        except (TypeError, ValueError):
            enc = 100
        out.setdefault(g, {"prior": [], "current": []})[bucket].append([h, enc])
    return out


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("__DATA__", data_json)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = _find_project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", default=None,
                    help="coverage parquet (default: newest data/processed/coverage/coverage_*.parquet)")
    ap.add_argument("--scored-dir", default=str(root / "data" / "processed" / "scored"))
    ap.add_argument("--provider", type=int, default=130077, help="provider_id (default AT&T 130077)")
    ap.add_argument("--service", default="5G-NR 7/1", help="service label (default '5G-NR 7/1')")
    ap.add_argument("--state", default=None, help="2-digit state FIPS to restrict county picks (e.g. 48)")
    ap.add_argument("--counties", default=None, help="explicit comma-separated county GEOIDs")
    ap.add_argument("--max-counties", type=int, default=6)
    ap.add_argument("--out", default=str(root / "sample_coverage.html"))
    args = ap.parse_args(argv)

    coverage_path = Path(args.coverage) if args.coverage else None
    if coverage_path is None:
        cands = sorted(
            glob.glob(str(root / "data" / "processed" / "coverage" / "coverage_*.parquet")),
            key=os.path.getmtime,
        )
        if not cands:
            raise SystemExit("No coverage parquet found under data/processed/coverage/.")
        coverage_path = Path(cands[-1])
    log.info("coverage source: %s", coverage_path)

    explicit = args.counties.split(",") if args.counties else None
    geoids, meta, labels = _pick_counties(
        Path(args.scored_dir), args.provider, args.service, args.state,
        explicit, args.max_counties,
    )
    if not geoids:
        raise SystemExit("No counties selected. Try --counties or a different --state/--provider.")
    log.info("counties: %s", ", ".join(geoids))

    hexes = _load_hexes(coverage_path, args.provider, args.service, geoids)

    counties = []
    for g in geoids:
        h = hexes.get(g, {"prior": [], "current": []})
        m = meta.get(g, {})
        counties.append({
            "geoid": g,
            "name": m.get("name", g),
            "flagged": m.get("flagged", False),
            "priority": round(m.get("priority", 0.0), 3),
            "added_km2": round(m.get("added_km2", 0.0), 1),
            "pct_increase": round(m.get("pct_increase", 0.0), 1),
            "prior_towers": m.get("prior_towers", 0),
            "current_towers": m.get("current_towers", 0),
            "new_towers": m.get("new_towers", 0),
            "explanation": m.get("explanation", ""),
            "prior": h["prior"],
            "current": h["current"],
        })
    counties = [c for c in counties if c["prior"] or c["current"]]
    if not counties:
        raise SystemExit(
            "Selected counties had no coverage rows in the parquet. "
            "Check --provider/--service match the run that produced the data."
        )

    payload = {
        "provider_id": args.provider,
        "service": args.service,
        "prior_label": labels.get("prior", "Prior"),
        "current_label": labels.get("current", "Current"),
        "counties": counties,
    }
    out_path = Path(args.out)
    out_path.write_text(build_html(payload), encoding="utf-8")
    total_hex = sum(len(c["prior"]) + len(c["current"]) for c in counties)
    log.info("wrote %s (%d counties, %s hexes, %.1f MB)",
             out_path, len(counties), f"{total_hex:,}", out_path.stat().st_size / 1e6)
    print(f"\nDone -> {out_path}\nOpen it directly in a browser, or email it to the engineers.")
    return 0


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FCC Coverage Sample — prior vs current hexes</title>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/h3-js@4.1.0/dist/h3-js.umd.js"></script>
<style>
  :root { --bg:#0f172a; --panel:#111c33; --ink:#e2e8f0; --muted:#94a3b8; --line:#1e293b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:14px 18px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:16px; }
  header .sub { color:var(--muted); font-size:13px; margin-top:3px; }
  .controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap; padding:12px 18px; border-bottom:1px solid var(--line); }
  select, label { font-size:13px; }
  select { background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:7px 10px; }
  .badge { font-size:11px; padding:2px 7px; border-radius:999px; background:#334155; color:#e2e8f0; }
  .badge.flag { background:#ef4444; }
  .stats { color:var(--muted); font-size:12.5px; }
  .stats b { color:var(--ink); }
  .maps { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:12px 18px; }
  .mapcard { background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
  .mapcard .lbl { padding:8px 12px; font-size:13px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; }
  .mapcard .m { height:64vh; }
  .legend { display:flex; gap:14px; align-items:center; padding:0 18px 14px; color:var(--muted); font-size:12px; flex-wrap:wrap; }
  .ramp { width:180px; height:12px; border-radius:6px;
    background:linear-gradient(90deg, rgb(200,55,45), rgb(245,185,52), rgb(140,200,72), rgb(60,185,95), rgb(28,178,178)); }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:middle; margin-right:5px; }
  .explain { padding:0 18px 16px; color:var(--muted); font-size:13px; max-width:1100px; }
  code { background:#1e293b; padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<header>
  <h1>FCC Mobile Coverage — prior vs current (sample)</h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="controls">
  <label>County
    <select id="county"></select>
  </label>
  <span id="flagbadge"></span>
  <label><input type="checkbox" id="highlightNew" checked /> highlight newly-covered hexes</label>
  <span class="stats" id="stats"></span>
</div>
<div class="maps">
  <div class="mapcard">
    <div class="lbl"><span id="lbl-prior">Prior</span><span class="stats" id="cnt-prior"></span></div>
    <div class="m" id="map-prior"></div>
  </div>
  <div class="mapcard">
    <div class="lbl"><span id="lbl-current">Current</span><span class="stats" id="cnt-current"></span></div>
    <div class="m" id="map-current"></div>
  </div>
</div>
<div class="legend">
  <span>Signal</span><span class="ramp"></span><span>weak → strong</span>
  <span><span class="swatch" style="background:#22c55e"></span>newly covered (current only)</span>
</div>
<div class="explain" id="explain"></div>

<script>
const DATA = __DATA__;
const h3lib = window.h3;

const _STOPS = [[-125,[200,55,45]],[-118,[235,120,40]],[-110,[245,185,52]],[-102,[220,215,58]],
  [-94,[140,200,72]],[-85,[60,185,95]],[-72,[30,180,155]],[-55,[28,178,178]]];
function signalColor(dbm){
  const s=_STOPS;
  if(dbm<=s[0][0])return `rgb(${s[0][1].join(',')})`;
  if(dbm>=s[s.length-1][0])return `rgb(${s[s.length-1][1].join(',')})`;
  for(let i=0;i<s.length-1;i++){const[d0,c0]=s[i],[d1,c1]=s[i+1];
    if(dbm>=d0&&dbm<=d1){const t=(dbm-d0)/(d1-d0);
      return `rgb(${Math.round(c0[0]+(c1[0]-c0[0])*t)},${Math.round(c0[1]+(c1[1]-c0[1])*t)},${Math.round(c0[2]+(c1[2]-c0[2])*t)})`;}}
  return `rgb(${s[0][1].join(',')})`;
}
// dBm = raw - 100 when raw looks encoded (>-40).
function decodeSignal(raw){ return (typeof raw==='number'&&raw>-40)?raw-100:raw; }

function ring(cell){ if(!h3lib?.cellToBoundary)return null; const r=h3lib.cellToBoundary(cell,true); return r?.length?[...r,r[0]]:null; }

function hexesToGeoJSON(hexes, newSet){
  const feats=[];
  (hexes||[]).forEach(([cell,raw])=>{
    const rg=ring(cell); if(!rg)return;
    const dbm=decodeSignal(raw);
    const isNew=newSet&&newSet.has(cell);
    feats.push({type:'Feature',
      properties:{color:isNew?'#22c55e':signalColor(dbm), isNew:isNew?1:0},
      geometry:{type:'Polygon',coordinates:[rg]}});
  });
  return {type:'FeatureCollection',features:feats};
}
function boundsOf(geo){
  let a=Infinity,b=Infinity,c=-Infinity,d=-Infinity;
  const v=co=>{ if(typeof co[0]==='number'){a=Math.min(a,co[0]);c=Math.max(c,co[0]);b=Math.min(b,co[1]);d=Math.max(d,co[1]);}else co.forEach(v); };
  (geo.features||[]).forEach(f=>v(f.geometry.coordinates));
  return Number.isFinite(a)?[[a,b],[c,d]]:null;
}
function baseStyle(){
  return {version:8, sources:{carto:{type:'raster',
    tiles:['https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
           'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'],tileSize:256,attribution:'© CARTO'}},
    layers:[{id:'bg',type:'raster',source:'carto'}]};
}
function makeMap(id){ return new maplibregl.Map({container:id,style:baseStyle(),center:[-98,39],zoom:4,attributionControl:false}); }

const mapPrior=makeMap('map-prior'), mapCurrent=makeMap('map-current');
let ready=0; [mapPrior,mapCurrent].forEach(m=>m.on('load',()=>{ if(++ready===2) render(); }));

function drawLayer(map, geo, bounds){
  const sid='hex', lid='hex-fill', lline='hex-line';
  if(map.getLayer(lline))map.removeLayer(lline);
  if(map.getLayer(lid))map.removeLayer(lid);
  if(map.getSource(sid))map.removeSource(sid);
  map.addSource(sid,{type:'geojson',data:geo});
  map.addLayer({id:lid,type:'fill',source:sid,paint:{'fill-color':['get','color'],'fill-opacity':0.72}});
  map.addLayer({id:lline,type:'line',source:sid,paint:{'line-color':['get','color'],'line-width':0.3,'line-opacity':0.4}});
  if(bounds)map.fitBounds(bounds,{padding:24,duration:0});
}

function populate(){
  const sel=document.getElementById('county');
  DATA.counties.forEach((c,i)=>{
    const o=document.createElement('option');
    o.value=i; o.textContent=`${c.name} (${c.geoid})${c.flagged?'  ⚑':''}`;
    sel.appendChild(o);
  });
  sel.addEventListener('change',render);
  document.getElementById('highlightNew').addEventListener('change',render);
  const pl=DATA.prior_label||'Prior', cl=DATA.current_label||'Current';
  document.getElementById('subtitle').textContent =
    `Provider ${DATA.provider_id} · ${DATA.service} · prior = ${pl}, current = ${cl}. Green hexes are covered in current but not prior.`;
  document.getElementById('lbl-prior').textContent = `Prior — ${pl}`;
  document.getElementById('lbl-current').textContent = `Current — ${cl}`;
}

function render(){
  const c=DATA.counties[+document.getElementById('county').value];
  if(!c)return;
  const priorSet=new Set(c.prior.map(x=>x[0]));
  const newSet=new Set(c.current.map(x=>x[0]).filter(h=>!priorSet.has(h)));
  const hi=document.getElementById('highlightNew').checked;
  const priorGeo=hexesToGeoJSON(c.prior,null);
  const curGeo=hexesToGeoJSON(c.current, hi?newSet:null);
  const b=boundsOf({type:'FeatureCollection',features:[...priorGeo.features,...curGeo.features]});
  drawLayer(mapPrior,priorGeo,b);
  drawLayer(mapCurrent,curGeo,b);
  document.getElementById('cnt-prior').textContent = `${c.prior.length.toLocaleString()} hexes`;
  document.getElementById('cnt-current').textContent =
    `${c.current.length.toLocaleString()} hexes · +${newSet.size.toLocaleString()} new`;
  document.getElementById('flagbadge').innerHTML =
    c.flagged?'<span class="badge flag">FLAGGED for review</span>':'<span class="badge">not flagged</span>';
  document.getElementById('stats').innerHTML =
    `priority <b>${c.priority}</b> · added <b>${c.added_km2}</b> km² · <b>${c.pct_increase}%</b> increase · towers <b>${c.prior_towers}→${c.current_towers}</b> (<b>${c.new_towers}</b> new)`;
  document.getElementById('explain').textContent = c.explanation || '';
}

populate();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
