"""Generate a standalone DEMO coverage HTML with synthetic, illustrative data.

Unlike make_sample_coverage.py (which reads real pipeline output), this builds
hand-crafted H3 coverage for a few fictional counties designed to showcase each
feature clearly:

  * "Blanketed County"  - FLAGGED: huge coverage jump, 0 new towers (gaming).
  * "New-Tower County"  - not flagged: growth radiates from a genuine new tower.
  * "Steady County"     - not flagged: little change vintage to vintage.
  * "Signal Gradient"   - shows the weak->strong signal color ramp.

Output is a single self-contained sample_coverage_demo.html (data embedded
inline; MapLibre + h3-js from CDN) with a COLLAPSIBLE bottom-right legend.
Emailable as-is. Run:  python tools/make_demo_coverage.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h3

RES = 9


def _disk(lat: float, lng: float, k: int) -> list[str]:
    return list(h3.grid_disk(h3.latlng_to_cell(lat, lng, RES), k))


def _signal_by_distance(center: str, cell: str, near_dbm: float, far_dbm: float, kmax: int) -> float:
    d = h3.grid_distance(center, cell)
    t = min(d / max(kmax, 1), 1.0)
    return round(near_dbm + (far_dbm - near_dbm) * t, 1)


def _enc(dbm: float) -> int:
    return int(round(dbm)) + 100


def _hexes(cells, sig_fn) -> list[list]:
    return [[c, _enc(sig_fn(c))] for c in cells]


def build_counties() -> list[dict]:
    counties: list[dict] = []

    # 1) Blanketed County (FLAGGED) — small prior cluster, current blankets a huge
    # area with uniformly strong signal, but tower count is unchanged (gaming).
    c1 = (39.10, -101.20)
    center1 = h3.latlng_to_cell(*c1, RES)
    prior1 = _disk(*c1, 3)
    current1 = _disk(*c1, 11)
    counties.append({
        "geoid": "20999", "name": "Blanketed County", "flagged": True,
        "priority": 0.94, "added_km2": 812.0, "pct_increase": 640.0,
        "prior_towers": 2, "current_towers": 2, "new_towers": 0,
        "explanation": "100% of new coverage claimed from existing sites; coverage up 640% with no new towers — classic same-site blanket fill. Recommend physical drive test.",
        "prior": _hexes(prior1, lambda c: _signal_by_distance(center1, c, -70, -95, 3)),
        # current: uniformly strong everywhere (implausible blanket)
        "current": _hexes(current1, lambda c: -72.0),
        "sites_prior": [{"lat": c1[0], "lng": c1[1], "site_class": "prior_site"},
                        {"lat": c1[0] + 0.05, "lng": c1[1] + 0.06, "site_class": "prior_site"}],
        "sites_current": [{"lat": c1[0], "lng": c1[1], "site_class": "stable_site"},
                          {"lat": c1[0] + 0.05, "lng": c1[1] + 0.06, "site_class": "stable_site"}],
    })

    # 2) New-Tower County (NOT flagged) — current adds a second lobe around a
    # genuinely new tower ~8 km away. Growth is tower-backed.
    c2a = (38.20, -98.10)
    c2b = (38.28, -97.98)   # new tower location
    center2a = h3.latlng_to_cell(*c2a, RES)
    center2b = h3.latlng_to_cell(*c2b, RES)
    prior2 = _disk(*c2a, 4)
    current2 = sorted(set(_disk(*c2a, 4)) | set(_disk(*c2b, 4)))
    def sig2(c):
        return min(_signal_by_distance(center2a, c, -68, -100, 4),
                   _signal_by_distance(center2b, c, -68, -100, 4))
    counties.append({
        "geoid": "20997", "name": "New-Tower County", "flagged": False,
        "priority": 0.41, "added_km2": 143.0, "pct_increase": 92.0,
        "prior_towers": 1, "current_towers": 2, "new_towers": 1,
        "explanation": "New coverage radiates from a newly built site (1 new tower). Growth is consistent with a real build-out — not selected.",
        "prior": _hexes(prior2, lambda c: _signal_by_distance(center2a, c, -68, -100, 4)),
        "current": _hexes(current2, sig2),
        "sites_prior": [{"lat": c2a[0], "lng": c2a[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c2a[0], "lng": c2a[1], "site_class": "stable_site"},
                          {"lat": c2b[0], "lng": c2b[1], "site_class": "new_site"}],
    })

    # 3) Steady County (NOT flagged) — nearly identical vintages.
    c3 = (37.30, -100.00)
    center3 = h3.latlng_to_cell(*c3, RES)
    prior3 = _disk(*c3, 5)
    current3 = _disk(*c3, 5) + h3.grid_disk(h3.latlng_to_cell(c3[0] + 0.02, c3[1] + 0.02, RES), 1)
    current3 = sorted(set(current3))
    counties.append({
        "geoid": "20995", "name": "Steady County", "flagged": False,
        "priority": 0.12, "added_km2": 9.0, "pct_increase": 4.0,
        "prior_towers": 3, "current_towers": 3, "new_towers": 0,
        "explanation": "Coverage essentially unchanged between vintages — no material growth to review.",
        "prior": _hexes(prior3, lambda c: _signal_by_distance(center3, c, -66, -98, 5)),
        "current": _hexes(current3, lambda c: _signal_by_distance(center3, c, -66, -98, 5)),
        "sites_prior": [{"lat": c3[0], "lng": c3[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c3[0], "lng": c3[1], "site_class": "stable_site"}],
    })

    # 4) Signal Gradient (NOT flagged) — showcases the weak->strong color ramp.
    c4 = (40.00, -99.50)
    center4 = h3.latlng_to_cell(*c4, RES)
    cells4 = _disk(*c4, 8)
    counties.append({
        "geoid": "20993", "name": "Signal Gradient (demo)", "flagged": False,
        "priority": 0.20, "added_km2": 0.0, "pct_increase": 0.0,
        "prior_towers": 1, "current_towers": 1, "new_towers": 0,
        "explanation": "Illustrative single tower: strong (teal/green) at the core fading to weak (orange/red) at the fringe — the signal color ramp used across the app.",
        "prior": _hexes(cells4, lambda c: _signal_by_distance(center4, c, -58, -118, 8)),
        "current": _hexes(cells4, lambda c: _signal_by_distance(center4, c, -58, -118, 8)),
        "sites_prior": [{"lat": c4[0], "lng": c4[1], "site_class": "prior_site"}],
        "sites_current": [{"lat": c4[0], "lng": c4[1], "site_class": "stable_site"}],
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
  :root { --bg:#0f172a; --panel:#111c33; --ink:#e2e8f0; --muted:#94a3b8; --line:#1e293b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:14px 18px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:16px; }
  header .sub { color:var(--muted); font-size:13px; margin-top:3px; }
  .controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap; padding:12px 18px; border-bottom:1px solid var(--line); }
  select { background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:7px 10px; font-size:13px; }
  label { font-size:13px; }
  .badge { font-size:11px; padding:2px 7px; border-radius:999px; background:#334155; }
  .badge.flag { background:#ef4444; }
  .stats { color:var(--muted); font-size:12.5px; }
  .stats b { color:var(--ink); }
  .maps { position:relative; display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:12px 18px; }
  .mapcard { background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
  .mapcard .lbl { padding:8px 12px; font-size:13px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; }
  .mapcard .m { height:64vh; }
  .explain { padding:0 18px 16px; color:var(--muted); font-size:13px; max-width:1100px; }

  /* Collapsible legend, bottom-right, floating over the maps */
  .legend-box { position:fixed; right:22px; bottom:22px; z-index:20; width:240px;
    background:rgba(17,28,51,0.96); border:1px solid var(--line); border-radius:10px;
    box-shadow:0 6px 24px rgba(0,0,0,0.45); font-size:12px; }
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
function ring(cell){ const r=h3lib.cellToBoundary(cell,true); return r?.length?[...r,r[0]]:null; }

function hexesToGeoJSON(hexes,newSet){
  const feats=[];
  (hexes||[]).forEach(([cell,raw])=>{
    const rg=ring(cell); if(!rg)return;
    const isNew=newSet&&newSet.has(cell);
    feats.push({type:'Feature',
      properties:{color:isNew?'#22c55e':signalColor(decodeSignal(raw))},
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
  return {version:8,sources:{c:{type:'raster',
    tiles:['https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png','https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'],
    tileSize:256,attribution:'© CARTO'}},layers:[{id:'bg',type:'raster',source:'c'}]};
}
function makeMap(id){ return new maplibregl.Map({container:id,style:baseStyle(),center:[-99,39],zoom:5,attributionControl:false}); }
const mapPrior=makeMap('map-prior'), mapCurrent=makeMap('map-current');
let ready=0; [mapPrior,mapCurrent].forEach(m=>m.on('load',()=>{ if(++ready===2) render(); }));

let markers=[];
function clearMarkers(){ markers.forEach(m=>m.remove()); markers=[]; }
function addTowers(map, sites){
  if(!document.getElementById('showTowers').checked) return;
  const colors={new_site:'#22c55e',expanded_site:'#a855f7',prior_site:'#38bdf8',stable_site:'#38bdf8'};
  (sites||[]).forEach(s=>{
    const el=document.createElement('div');
    el.style.cssText=`width:12px;height:12px;border-radius:50%;background:${colors[s.site_class]||'#38bdf8'};border:1.6px solid #0f172a;box-shadow:0 0 0 1px rgba(255,255,255,.25)`;
    markers.push(new maplibregl.Marker({element:el}).setLngLat([s.lng,s.lat]).addTo(map));
  });
}
function drawLayer(map,geo,bounds){
  ['hex-line','hex-fill'].forEach(l=>{ if(map.getLayer(l))map.removeLayer(l); });
  if(map.getSource('hex'))map.removeSource('hex');
  map.addSource('hex',{type:'geojson',data:geo});
  map.addLayer({id:'hex-fill',type:'fill',source:'hex',paint:{'fill-color':['get','color'],'fill-opacity':0.72}});
  map.addLayer({id:'hex-line',type:'line',source:'hex',paint:{'line-color':['get','color'],'line-width':0.3,'line-opacity':0.4}});
  if(bounds)map.fitBounds(bounds,{padding:26,duration:0});
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
  const b=boundsOf({type:'FeatureCollection',features:[...priorGeo.features,...curGeo.features]});
  clearMarkers();
  drawLayer(mapPrior,priorGeo,b); drawLayer(mapCurrent,curGeo,b);
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
