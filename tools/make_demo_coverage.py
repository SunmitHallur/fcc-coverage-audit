"""Generate a standalone DEMO coverage HTML with synthetic, illustrative data.

Builds hand-crafted H3 coverage for six fictional counties designed to showcase
each feature clearly, and renders them with the SAME look as the deployed web
cockpit (web/index.html): light CARTO basemap, smooth heatmap (no hex grid
lines), 2-ring signal smoothing, circle tower markers, and a dark county
boundary outline.

Counties:
  * "Blanketed County"    - FLAGGED: huge jump, 0 new towers (same-site gaming).
  * "Boundary-Snap County"- FLAGGED: new coverage hugs the county line.
  * "New-Tower County"    - not flagged: growth radiates from a genuine new tower.
  * "Expanded-Tower County"- borderline: one existing tower's lobe grows.
  * "Steady County"       - not flagged: little change vintage to vintage.
  * "Signal Gradient"     - shows the weak->strong signal color ramp.

Output: a single self-contained sample_coverage_demo.html (data embedded inline;
MapLibre + h3-js from CDN) with a COLLAPSIBLE bottom-right legend. Emailable.
Run:  python tools/make_demo_coverage.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h3

RES = 9


def _disk(lat: float, lng: float, k: int) -> list[str]:
    return list(h3.grid_disk(h3.latlng_to_cell(lat, lng, RES), k))


def _ring(lat: float, lng: float, k: int) -> list[str]:
    """Hexes exactly on the k-th ring (a hollow outline)."""
    return list(h3.grid_ring(h3.latlng_to_cell(lat, lng, RES), k))


def _signal_by_distance(center: str, cell: str, near_dbm: float, far_dbm: float, kmax: int) -> float:
    d = h3.grid_distance(center, cell)
    t = min(d / max(kmax, 1), 1.0)
    return round(near_dbm + (far_dbm - near_dbm) * t, 1)


def _enc(dbm: float) -> int:
    return int(round(dbm)) + 100


def _hexes(cells, sig_fn) -> list[list]:
    return [[c, _enc(sig_fn(c))] for c in cells]


def _boundary_rect(counties_cells: list[str], pad: float = 0.06) -> dict:
    """A rectangular 'county' boundary around a set of hexes (illustrative)."""
    lats, lngs = [], []
    for c in counties_cells:
        la, ln = h3.cell_to_latlng(c)
        lats.append(la)
        lngs.append(ln)
    if not lats:
        return {}
    mnla, mxla = min(lats) - pad, max(lats) + pad
    mnln, mxln = min(lngs) - pad, max(lngs) + pad
    ring = [[mnln, mnla], [mxln, mnla], [mxln, mxla], [mnln, mxla], [mnln, mnla]]
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def build_counties() -> list[dict]:
    counties: list[dict] = []

    # 1) Blanketed County (FLAGGED) — small prior cluster, current blankets a
    # huge area with uniform strong signal, but tower count is unchanged.
    c1 = (39.10, -101.20)
    center1 = h3.latlng_to_cell(*c1, RES)
    prior1 = _disk(*c1, 3)
    current1 = _disk(*c1, 12)
    counties.append({
        "geoid": "20999", "name": "Blanketed County", "flagged": True,
        "priority": 0.94, "added_km2": 812.0, "pct_increase": 640.0,
        "prior_towers": 2, "current_towers": 2, "new_towers": 0,
        "explanation": "100% of new coverage claimed from existing sites; coverage up 640% with no new towers — classic same-site blanket fill. Recommend physical drive test.",
        "prior": _hexes(prior1, lambda c: _signal_by_distance(center1, c, -70, -95, 3)),
        "current": _hexes(current1, lambda c: -74.0),
        "sites_prior": [{"lat": c1[0], "lng": c1[1], "site_class": "prior_site"},
                        {"lat": c1[0] + 0.05, "lng": c1[1] + 0.06, "site_class": "prior_site"}],
        "sites_current": [{"lat": c1[0], "lng": c1[1], "site_class": "stable_site"},
                          {"lat": c1[0] + 0.05, "lng": c1[1] + 0.06, "site_class": "stable_site"}],
        "boundary": _boundary_rect(current1),
    })

    # 2) Boundary-Snap County (FLAGGED) — new coverage appears as a band hugging
    # the county outline rather than radiating from towers.
    c2 = (41.50, -100.00)
    center2 = h3.latlng_to_cell(*c2, RES)
    prior2 = _disk(*c2, 3)
    full2 = _disk(*c2, 10)
    edge2 = sorted(set(_ring(*c2, 9)) | set(_ring(*c2, 10)) | set(prior2))
    counties.append({
        "geoid": "20998", "name": "Boundary-Snap County", "flagged": True,
        "priority": 0.88, "added_km2": 420.0, "pct_increase": 380.0,
        "prior_towers": 1, "current_towers": 1, "new_towers": 0,
        "explanation": "New coverage aligns to the county boundary rather than radiating from a tower — a boundary-snapping tell. High same-site share, no new build. Recommend review.",
        "prior": _hexes(prior2, lambda c: _signal_by_distance(center2, c, -72, -96, 3)),
        "current": _hexes(edge2, lambda c: -80.0),
        "sites_prior": [{"lat": c2[0], "lng": c2[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c2[0], "lng": c2[1], "site_class": "stable_site"}],
        "boundary": _boundary_rect(full2, pad=0.02),
    })

    # 3) New-Tower County (NOT flagged) — current adds a second lobe around a
    # genuinely new tower ~8 km away.
    c3a = (38.20, -98.10)
    c3b = (38.28, -97.98)
    center3a = h3.latlng_to_cell(*c3a, RES)
    center3b = h3.latlng_to_cell(*c3b, RES)
    prior3 = _disk(*c3a, 4)
    current3 = sorted(set(_disk(*c3a, 4)) | set(_disk(*c3b, 4)))
    counties.append({
        "geoid": "20997", "name": "New-Tower County", "flagged": False,
        "priority": 0.41, "added_km2": 143.0, "pct_increase": 92.0,
        "prior_towers": 1, "current_towers": 2, "new_towers": 1,
        "explanation": "New coverage radiates from a newly built site (1 new tower, green). Growth is consistent with a real build-out — not selected.",
        "prior": _hexes(prior3, lambda c: _signal_by_distance(center3a, c, -68, -100, 4)),
        "current": _hexes(current3, lambda c: min(
            _signal_by_distance(center3a, c, -68, -100, 4),
            _signal_by_distance(center3b, c, -68, -100, 4))),
        "sites_prior": [{"lat": c3a[0], "lng": c3a[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c3a[0], "lng": c3a[1], "site_class": "stable_site"},
                          {"lat": c3b[0], "lng": c3b[1], "site_class": "new_site"}],
        "boundary": _boundary_rect(current3),
    })

    # 4) Expanded-Tower County (borderline) — one existing tower's lobe grows.
    c4 = (36.90, -99.30)
    center4 = h3.latlng_to_cell(*c4, RES)
    prior4 = _disk(*c4, 3)
    current4 = _disk(*c4, 6)
    counties.append({
        "geoid": "20996", "name": "Expanded-Tower County", "flagged": False,
        "priority": 0.52, "added_km2": 96.0, "pct_increase": 210.0,
        "prior_towers": 2, "current_towers": 2, "new_towers": 0,
        "explanation": "Existing tower's coverage lobe expanded (purple = expanded site). Moderate same-site growth; borderline — monitor but not a clear anomaly.",
        "prior": _hexes(prior4, lambda c: _signal_by_distance(center4, c, -70, -98, 3)),
        "current": _hexes(current4, lambda c: _signal_by_distance(center4, c, -70, -104, 6)),
        "sites_prior": [{"lat": c4[0], "lng": c4[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c4[0], "lng": c4[1], "site_class": "expanded_site"}],
        "boundary": _boundary_rect(current4),
    })

    # 5) Steady County (NOT flagged) — nearly identical vintages.
    c5 = (37.30, -100.60)
    center5 = h3.latlng_to_cell(*c5, RES)
    prior5 = _disk(*c5, 5)
    current5 = sorted(set(_disk(*c5, 5)) | set(h3.grid_disk(h3.latlng_to_cell(c5[0] + 0.02, c5[1] + 0.02, RES), 1)))
    counties.append({
        "geoid": "20995", "name": "Steady County", "flagged": False,
        "priority": 0.12, "added_km2": 9.0, "pct_increase": 4.0,
        "prior_towers": 3, "current_towers": 3, "new_towers": 0,
        "explanation": "Coverage essentially unchanged between vintages — no material growth to review.",
        "prior": _hexes(prior5, lambda c: _signal_by_distance(center5, c, -66, -98, 5)),
        "current": _hexes(current5, lambda c: _signal_by_distance(center5, c, -66, -98, 5)),
        "sites_prior": [{"lat": c5[0], "lng": c5[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c5[0], "lng": c5[1], "site_class": "stable_site"}],
        "boundary": _boundary_rect(current5),
    })

    # 6) Signal Gradient (NOT flagged) — showcases the weak->strong color ramp.
    c6 = (40.00, -99.50)
    center6 = h3.latlng_to_cell(*c6, RES)
    cells6 = _disk(*c6, 9)
    counties.append({
        "geoid": "20993", "name": "Signal Gradient (demo)", "flagged": False,
        "priority": 0.20, "added_km2": 0.0, "pct_increase": 0.0,
        "prior_towers": 1, "current_towers": 1, "new_towers": 0,
        "explanation": "Illustrative single tower: strong (teal/green) at the core fading to weak (orange/red) at the fringe — the signal color ramp used across the app.",
        "prior": _hexes(cells6, lambda c: _signal_by_distance(center6, c, -58, -118, 9)),
        "current": _hexes(cells6, lambda c: _signal_by_distance(center6, c, -58, -118, 9)),
        "sites_prior": [{"lat": c6[0], "lng": c6[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c6[0], "lng": c6[1], "site_class": "stable_site"}],
        "boundary": _boundary_rect(cells6),
    })
    return counties


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    payload = {
        "provider_id": "DEMO",
        "service": "5G-NR 7/1 (illustrative sample data)",
        "prior_label": "June 30, 2025",
        "current_label": "December 31, 2025",
        "counties": build_counties(),
    }
    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    out = root / "sample_coverage_demo.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} KB). Open it or email it directly.")
    return 0


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FCC Coverage — sample viewer</title>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/h3-js@4.1.0/dist/h3-js.umd.js"></script>
<style>
  :root { --bg:#f1f5f9; --panel:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:14px 18px; border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { margin:0; font-size:16px; }
  header .sub { color:var(--muted); font-size:13px; margin-top:3px; }
  .controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap; padding:12px 18px; border-bottom:1px solid var(--line); background:var(--panel); }
  select { background:#fff; color:var(--ink); border:1px solid #cbd5e1; border-radius:8px; padding:7px 10px; font-size:13px; }
  label { font-size:13px; }
  .badge { font-size:11px; padding:2px 7px; border-radius:999px; background:#e2e8f0; color:#0f172a; }
  .badge.flag { background:#ef4444; color:#fff; }
  .stats { color:var(--muted); font-size:12.5px; }
  .stats b { color:var(--ink); }
  .maps { position:relative; display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:12px 18px; }
  .mapcard { background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:0 1px 3px rgba(15,23,42,.06); }
  .mapcard .lbl { padding:8px 12px; font-size:13px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; font-weight:600; }
  .mapcard .m { height:64vh; }
  .explain { padding:0 18px 16px; color:#334155; font-size:13px; max-width:1100px; }

  /* Collapsible legend, bottom-right, floating over the maps */
  .legend-box { position:fixed; right:22px; bottom:22px; z-index:20; width:240px;
    background:rgba(255,255,255,0.97); border:1px solid var(--line); border-radius:10px;
    box-shadow:0 6px 24px rgba(15,23,42,0.18); font-size:12px; }
  .legend-hd { display:flex; justify-content:space-between; align-items:center; cursor:pointer;
    padding:9px 12px; user-select:none; }
  .legend-hd b { font-size:12.5px; }
  .legend-hd .chev { color:var(--muted); transition:transform .15s; }
  .legend-box.collapsed .legend-bd { display:none; }
  .legend-box.collapsed .chev { transform:rotate(-90deg); }
  .legend-bd { padding:2px 12px 12px; color:var(--muted); }
  .ramp { width:100%; height:12px; border-radius:6px; margin:6px 0 3px;
    background:linear-gradient(90deg, rgb(200,55,45), rgb(245,185,52), rgb(140,200,72), rgb(60,185,95), rgb(28,178,178)); }
  .ramp-lbl { display:flex; justify-content:space-between; }
  .row { display:flex; align-items:center; gap:7px; margin-top:8px; }
  .swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
  .dot { width:11px; height:11px; border-radius:50%; border:1.5px solid #0f172a; display:inline-block; }
</style>
</head>
<body>
<header>
  <h1>FCC Mobile Coverage — prior vs current</h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="controls">
  <label>County <select id="county"></select></label>
  <span id="flagbadge"></span>
  <label><input type="checkbox" id="highlightNew" checked /> highlight newly-covered hexes</label>
  <label><input type="checkbox" id="showTowers" checked /> show towers</label>
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
<div class="explain" id="explain"></div>

<div class="legend-box" id="legend">
  <div class="legend-hd" id="legend-hd"><b>Legend / shading key</b><span class="chev">▾</span></div>
  <div class="legend-bd">
    <div>Modeled signal</div>
    <div class="ramp"></div>
    <div class="ramp-lbl"><span>weak</span><span>strong</span></div>
    <div class="row"><span class="swatch" style="background:#22c55e"></span> newly covered (current only)</div>
    <div class="row"><span class="dot" style="background:#38bdf8"></span> existing tower</div>
    <div class="row"><span class="dot" style="background:#22c55e"></span> new tower</div>
    <div class="row"><span class="dot" style="background:#a855f7"></span> expanded tower</div>
  </div>
</div>

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
function decodeSignal(raw){ return (typeof raw==='number'&&raw>-40)?raw-100:raw; }
function ringLngLat(cell){ const r=h3lib.cellToBoundary(cell,true); return r?.length?[...r,r[0]]:null; }

// 2-ring distance-weighted smoothing so coarse bands read as a smooth heatmap
// (matches the deployed cockpit).
function smoothHexSignal(cellSig){
  if(!h3lib?.gridDiskDistances) return cellSig;
  const out=new Map();
  for(const [cell,dbm] of cellSig){
    let num=0,den=0,rings;
    try{ rings=h3lib.gridDiskDistances(cell,2); }catch{ out.set(cell,dbm); continue; }
    rings.forEach((ring,dist)=>{ const w=1/(1+dist); ring.forEach(n=>{ const v=cellSig.get(n); if(v!==undefined){num+=v*w;den+=w;} }); });
    out.set(cell, den?num/den:dbm);
  }
  return out;
}
function hexesToGeoJSON(hexes,newSet){
  const sorted=[...(hexes||[])].sort((a,b)=>(a[1]||0)-(b[1]||0));
  const cellSig=new Map();
  sorted.forEach(([cell,raw])=>cellSig.set(cell,decodeSignal(raw)));
  const smoothed=smoothHexSignal(cellSig);
  const feats=[];
  sorted.forEach(([cell])=>{
    const rg=ringLngLat(cell); if(!rg)return;
    const isNew=newSet&&newSet.has(cell);
    feats.push({type:'Feature',
      properties:{color:isNew?'#22c55e':signalColor(smoothed.get(cell))},
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
  return {version:8,sources:{carto:{type:'raster',
    tiles:['https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
           'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
           'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png'],
    tileSize:256,maxzoom:19,attribution:'© CARTO © OpenStreetMap'}},
    layers:[{id:'carto',type:'raster',source:'carto',paint:{'raster-opacity':1.0}}]};
}
function makeMap(id){ return new maplibregl.Map({container:id,style:baseStyle(),center:[-99,39],zoom:6,maxZoom:14,pixelRatio:2,attributionControl:false,preserveDrawingBuffer:true}); }
const mapPrior=makeMap('map-prior'), mapCurrent=makeMap('map-current');
let ready=0; [mapPrior,mapCurrent].forEach(m=>m.on('load',()=>{ if(++ready===2) render(); }));

let markers=[];
function clearMarkers(){ markers.forEach(m=>m.remove()); markers=[]; }
function addTowers(map, sites){
  if(!document.getElementById('showTowers').checked) return;
  const colors={new_site:'#22c55e',expanded_site:'#a855f7',prior_site:'#38bdf8',stable_site:'#38bdf8'};
  (sites||[]).forEach(s=>{
    const el=document.createElement('div');
    el.style.cssText=`width:13px;height:13px;border-radius:50%;background:${colors[s.site_class]||'#38bdf8'};border:2px solid #0f172a;box-shadow:0 0 0 1px rgba(255,255,255,.5)`;
    markers.push(new maplibregl.Marker({element:el}).setLngLat([s.lng,s.lat]).addTo(map));
  });
}
function drawLayer(map, geo, boundary, bounds){
  ['hex-fill','county-line','county-fill-tmp'].forEach(l=>{ if(map.getLayer(l))map.removeLayer(l); });
  ['hex','county'].forEach(s=>{ if(map.getSource(s))map.removeSource(s); });
  if(boundary && boundary.geometry){
    map.addSource('county',{type:'geojson',data:boundary});
    map.addLayer({id:'county-fill-tmp',type:'fill',source:'county',paint:{'fill-color':'#e8f0fe','fill-opacity':0.18}});
  }
  map.addSource('hex',{type:'geojson',data:geo});
  // fill-outline-color = fill color => adjacent same-signal hexes blend, no grid.
  map.addLayer({id:'hex-fill',type:'fill',source:'hex',
    paint:{'fill-color':['get','color'],'fill-opacity':0.85,'fill-outline-color':['get','color']}});
  if(boundary && boundary.geometry){
    map.addLayer({id:'county-line',type:'line',source:'county',
      paint:{'line-color':'#111827','line-width':3,'line-opacity':1}});
  }
  if(bounds)map.fitBounds(bounds,{padding:24,maxZoom:14,duration:0});
  requestAnimationFrame(()=>map.resize());
}

function populate(){
  const sel=document.getElementById('county');
  DATA.counties.forEach((c,i)=>{ const o=document.createElement('option'); o.value=i; o.textContent=`${c.name}${c.flagged?'  ⚑':''}`; sel.appendChild(o); });
  sel.addEventListener('change',render);
  document.getElementById('highlightNew').addEventListener('change',render);
  document.getElementById('showTowers').addEventListener('change',render);
  document.getElementById('subtitle').textContent =
    `${DATA.service} · prior = ${DATA.prior_label}, current = ${DATA.current_label}. Green = covered in current but not prior.`;
  document.getElementById('lbl-prior').textContent = `Prior — ${DATA.prior_label}`;
  document.getElementById('lbl-current').textContent = `Current — ${DATA.current_label}`;
  const lg=document.getElementById('legend');
  document.getElementById('legend-hd').addEventListener('click',()=>lg.classList.toggle('collapsed'));
}
function render(){
  const c=DATA.counties[+document.getElementById('county').value]; if(!c)return;
  const priorSet=new Set(c.prior.map(x=>x[0]));
  const newSet=new Set(c.current.map(x=>x[0]).filter(h=>!priorSet.has(h)));
  const hi=document.getElementById('highlightNew').checked;
  const priorGeo=hexesToGeoJSON(c.prior,null);
  const curGeo=hexesToGeoJSON(c.current, hi?newSet:null);
  let b=boundsOf({type:'FeatureCollection',features:[...priorGeo.features,...curGeo.features]});
  if(c.boundary?.geometry){ const bb=boundsOf({type:'FeatureCollection',features:[c.boundary]}); if(bb&&b){ b=[[Math.min(b[0][0],bb[0][0]),Math.min(b[0][1],bb[0][1])],[Math.max(b[1][0],bb[1][0]),Math.max(b[1][1],bb[1][1])]]; } }
  clearMarkers();
  drawLayer(mapPrior,priorGeo,c.boundary,b); drawLayer(mapCurrent,curGeo,c.boundary,b);
  addTowers(mapPrior,c.sites_prior); addTowers(mapCurrent,c.sites_current);
  document.getElementById('cnt-prior').textContent=`${c.prior.length.toLocaleString()} hexes`;
  document.getElementById('cnt-current').textContent=`${c.current.length.toLocaleString()} hexes · +${newSet.size.toLocaleString()} new`;
  document.getElementById('flagbadge').innerHTML=c.flagged?'<span class="badge flag">FLAGGED for review</span>':'<span class="badge">not flagged</span>';
  document.getElementById('stats').innerHTML=
    `priority <b>${c.priority}</b> · added <b>${c.added_km2}</b> km² · <b>${c.pct_increase}%</b> increase · towers <b>${c.prior_towers}→${c.current_towers}</b> (<b>${c.new_towers}</b> new)`;
  document.getElementById('explain').textContent=c.explanation||'';
}
populate();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
