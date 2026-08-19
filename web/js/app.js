/** Coverage Change Audit — main viewer (ES module). */
import {
  TOWER_COLORS, SIGNAL_STOPS, FLAT_COVERAGE_COLOR,
  signalColor as signalColorImported,
  colorForRecord as colorForRecordImported,
  severityPillClass as severityPillClassImported,
} from './colors.js';
import { renderFlagMathInto } from './scoring.js';
import { formatTowerPanel, towerColor } from './towers.js';

const maplibregl = window.maplibregl;
const h3lib = window.h3;


const DATA_BASE = 'public/data';

    const severityPillClass = severityPillClassImported;

    let meta = {}, records = {}, countiesGeo = null;
    let currentProvider = null, currentService = null;
    let towerMarkers = [];
    let towersCache = {};
    let compareMaps = { prior: null, current: null };
    let detailCache = {};
    let detailRequestId = 0;
    let listShowAll = false;
    let compareViewMode = 'side'; // side | swipe
    let lastRasterUrls = { prior: '', current: '' };

    
    function formatVintage(v) {
      if (!v) return '?';
      const m = String(v).match(/^(\d{4})-(\d{2})-(\d{2})/);
      if (!m) return v;
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      return `${months[+m[2] - 1]} ${+m[3]}, ${m[1]}`;
    }

    function formatScope(scope) {
      const text = String(scope || 'all').trim();
      if (!text || text.toLowerCase() === 'all') return 'National scope';
      const codes = [...new Set(text.split(',').map(s => s.trim()).filter(Boolean))];
      return codes.length === 1 ? `State FIPS ${codes[0]}` : `${codes.length} states/DC`;
    }

    function safeServiceKey(service) {
      return String(service).replace(/\//g, '-').replace(/\s/g, '');
    }

    const _SIGNAL_STOPS = SIGNAL_STOPS;
    function signalColor(dbm) {
      const stops = _SIGNAL_STOPS;
      if (dbm <= stops[0][0]) return `rgb(${stops[0][1].join(',')})`;
      if (dbm >= stops[stops.length - 1][0]) return `rgb(${stops[stops.length - 1][1].join(',')})`;
      for (let i = 0; i < stops.length - 1; i++) {
        const [d0, c0] = stops[i], [d1, c1] = stops[i + 1];
        if (dbm >= d0 && dbm <= d1) {
          const t = (dbm - d0) / (d1 - d0);
          const r = Math.round(c0[0] + (c1[0] - c0[0]) * t);
          const g = Math.round(c0[1] + (c1[1] - c0[1]) * t);
          const b = Math.round(c0[2] + (c1[2] - c0[2]) * t);
          return `rgb(${r},${g},${b})`;
        }
      }
      return `rgb(${stops[0][1].join(',')})`;
    }

    function getCountyFeature(geoid) {
      if (!countiesGeo?.features) return null;
      return countiesGeo.features.find(f => String(f.properties.geoid) === String(geoid)) || null;
    }

    function countyBounds(countyFeature) {
      if (!countyFeature) return null;
      return boundsFromGeoJSON({ type: 'FeatureCollection', features: [countyFeature] });
    }

    function h3RingLngLat(cell) {
      if (!h3lib?.cellToBoundary) return null;
      const ring = h3lib.cellToBoundary(cell, true);
      if (!ring?.length) return null;
      return [...ring, ring[0]];
    }

    function resolveCountyFeature(geoid, detail) {
      return getCountyFeature(geoid)
        || (detail?.county_boundary?.type === 'Feature' ? detail.county_boundary : null);
    }

    // Decode compact int8-encoded signal: dBm = raw - 100
    // Encoding: encoded = round(dBm) + 100, so typical FCC range (-40 to -140 dBm)
    // maps to encoded values (+60 to -40). Real raw dBm floats are always ≤ -40,
    // so anything > -40 is unambiguously an encoded int8 value.
    function decodeSignal(raw) {
      if (typeof raw === 'number' && raw > -40) return raw - 100;
      return raw;
    }

    // Distance-weighted smoothing of per-hex signal over the 2-ring neighborhood.
    // The FCC viewer renders a continuous interpolated RSRP raster; our res-9 hex
    // signal is discretized into coarse bands (e.g. -100/-90/-80), which reads as
    // hard color steps. Averaging each hex with its neighbors turns those steps
    // into a smooth gradient that more closely resembles the FCC render, without
    // changing the underlying data (raw values are still used for analysis).
    function smoothHexSignal(cellSig) {
      if (!h3lib?.gridDiskDistances) return cellSig;
      const out = new Map();
      for (const [cell, dbm] of cellSig) {
        let num = 0, den = 0;
        let rings;
        try { rings = h3lib.gridDiskDistances(cell, 2); } catch { out.set(cell, dbm); continue; }
        rings.forEach((ring, dist) => {
          const w = 1 / (1 + dist);
          ring.forEach(n => {
            const v = cellSig.get(n);
            if (v !== undefined) { num += v * w; den += w; }
          });
        });
        out.set(cell, den ? num / den : dbm);
      }
      return out;
    }

    function hexesToGeoJSON(hexes, sites, opts = {}) {
      const flat = !!opts.flat;
      const sorted = [...(hexes || [])].sort((a, b) => (a[1] || 0) - (b[1] || 0));
      const cellSig = new Map();
      sorted.forEach(([cell, rawSig]) => cellSig.set(cell, decodeSignal(rawSig)));
      const smoothed = flat ? cellSig : smoothHexSignal(cellSig);
      const flatColor = '#94a3b8';
      const features = [];
      sorted.forEach(([cell]) => {
        const dbm = smoothed.get(cell);
        const ring = h3RingLngLat(cell);
        if (!ring) return;
        features.push({
          type: 'Feature',
          properties: { dbm, color: flat ? flatColor : signalColor(dbm) },
          geometry: { type: 'Polygon', coordinates: [ring] }
        });
      });
      (sites || []).forEach(s => {
        features.push({
          type: 'Feature',
          properties: {
            kind: 'tower',
            site_class: s.site_class || 'site',
            in_county: s.in_county !== false,
          },
          geometry: { type: 'Point', coordinates: [s.lng, s.lat] }
        });
      });
      return { type: 'FeatureCollection', features };
    }

    function boundsFromGeoJSON(geo) {
      let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
      const visit = coords => {
        if (typeof coords[0] === 'number') {
          minLng = Math.min(minLng, coords[0]); maxLng = Math.max(maxLng, coords[0]);
          minLat = Math.min(minLat, coords[1]); maxLat = Math.max(maxLat, coords[1]);
        } else coords.forEach(visit);
      };
      (geo.features || []).forEach(f => visit(f.geometry.coordinates));
      if (!Number.isFinite(minLng)) return null;
      return [[minLng, minLat], [maxLng, maxLat]];
    }

    function unionBounds(...boundsList) {
      let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
      boundsList.filter(Boolean).forEach(b => {
        minLng = Math.min(minLng, b[0][0]); minLat = Math.min(minLat, b[0][1]);
        maxLng = Math.max(maxLng, b[1][0]); maxLat = Math.max(maxLat, b[1][1]);
      });
      if (!Number.isFinite(minLng)) return null;
      return [[minLng, minLat], [maxLng, maxLat]];
    }

    function destroyCompareMaps() {
      ['prior', 'current'].forEach(k => {
        if (compareMaps[k]) { compareMaps[k].remove(); compareMaps[k] = null; }
      });
    }

    function setCompareMode(mode) {
      const useRaster = mode === 'raster';
      ['prior', 'current'].forEach(side => {
        document.getElementById(`map-${side}-raster`).style.display = useRaster ? 'flex' : 'none';
        document.getElementById(`map-${side}`).style.display = useRaster ? 'none' : 'block';
      });
      document.getElementById('signal-gradient-legend').style.display = 'grid';
      if (useRaster) destroyCompareMaps();
    }

    function setCompareViewLayout(view) {
      compareViewMode = view;
      const isSwipe = view === 'swipe' && window.matchMedia('(min-width: 901px)').matches;
      document.getElementById('compare-maps').style.display = isSwipe ? 'none' : 'grid';
      document.getElementById('swipe-compare').style.display = isSwipe ? 'block' : 'none';
      document.getElementById('btn-side-by-side').classList.toggle('active', !isSwipe);
      document.getElementById('btn-swipe').classList.toggle('active', isSwipe);
    }

    function updateSwipePosition(pct) {
      const clip = `inset(0 ${100 - pct}% 0 0)`;
      document.getElementById('swipe-overlay').style.clipPath = clip;
      document.getElementById('swipe-handle').style.left = `${pct}%`;
    }

    function detailAssetBase(rec) {
      const pid = rec.provider_id || currentProvider;
      const svc = rec.service || currentService;
      const geoid = rec.geoid;
      return `${DATA_BASE}/details/${pid}/${safeServiceKey(svc)}/${geoid}`;
    }

    function showRasterCompare(detail, rec) {
      if (!detail.prior_map && !detail.current_map) {
        showPredictedRasterCompare(rec);
        return;
      }
      const base = detailAssetBase(rec);
      const priorUrl = `${base}/${detail.prior_map || 'prior.png'}`;
      const currentUrl = `${base}/${detail.current_map || 'current.png'}`;
      lastRasterUrls = { prior: priorUrl, current: currentUrl };
      document.getElementById('img-prior').src = priorUrl;
      document.getElementById('img-current').src = currentUrl;
      document.getElementById('swipe-prior').src = priorUrl;
      document.getElementById('swipe-current').src = currentUrl;
      document.getElementById('compare-toolbar').style.display = 'flex';
      setCompareMode('raster');
      setCompareViewLayout(compareViewMode);
      updateSwipePosition(Number(document.getElementById('swipe-slider').value));
    }

    function showPredictedRasterCompare(rec) {
      // Do not invent PNG URLs — default builds omit rasters and broken <img>
      // icons confuse reviewers. Hex maps are the primary path.
      lastRasterUrls = { prior: null, current: null };
      document.getElementById('img-prior').removeAttribute('src');
      document.getElementById('img-current').removeAttribute('src');
      document.getElementById('swipe-prior').removeAttribute('src');
      document.getElementById('swipe-current').removeAttribute('src');
      document.getElementById('compare-toolbar').style.display = 'none';
      setCompareMode('hex');
    }

    async function downloadComparison() {
      const { prior, current } = lastRasterUrls;
      if (!prior || !current) return;
      const loadImg = src => new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
      });
      try {
        const [pImg, cImg] = await Promise.all([loadImg(prior), loadImg(current)]);
        const h = Math.max(pImg.naturalHeight, cImg.naturalHeight);
        const w = pImg.naturalWidth + cImg.naturalWidth + 24;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h + 40;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ebe6dc';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#111827';
        ctx.font = '14px Inter, sans-serif';
        ctx.fillText(document.getElementById('map-prior-label').textContent, 8, 18);
        ctx.fillText(document.getElementById('map-current-label').textContent, pImg.naturalWidth + 16, 18);
        ctx.drawImage(pImg, 0, 28, pImg.naturalWidth, pImg.naturalHeight);
        ctx.drawImage(cImg, pImg.naturalWidth + 24, 28, cImg.naturalWidth, cImg.naturalHeight);
        const a = document.createElement('a');
        a.download = `${recSlug()}-coverage-compare.png`;
        a.href = canvas.toDataURL('image/png');
        a.click();
      } catch {
        window.open(prior, '_blank');
        window.open(current, '_blank');
      }
    }

    function recSlug() {
      const title = document.getElementById('detail-title').textContent || 'county';
      return title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    }

    function formatStatKm2(v) {
      if (v == null || !Number.isFinite(v)) return '—';
      return v >= 10 ? v.toFixed(1) : v.toFixed(2);
    }

    function formatStatPct(v) {
      if (v == null || !Number.isFinite(v)) return '—';
      return `${(v * 100).toFixed(1)}%`;
    }

    function fillDetailHeader(rec, detail) {
      document.getElementById('detail-title').textContent = rec.name;
      document.getElementById('detail-meta').textContent =
        `${rec.provider_name} · ${rec.service} · Priority ${(rec.priority || 0).toFixed(2)}`;
      const sevLabel = rec.explanation?.severity || 'Below threshold';
      const pill = document.getElementById('detail-severity-pill');
      pill.style.display = 'inline-block';
      pill.className = `pill ${severityPillClass(sevLabel)}`;
      pill.textContent = sevLabel;
      if (rec.flag) {
        const flag = document.createElement('span');
        flag.className = 'pill flag';
        flag.textContent = 'FLAG';
        if (!pill.dataset.hasFlag) {
          pill.after(flag);
          pill.dataset.hasFlag = '1';
        }
      } else if (pill.nextElementSibling?.classList?.contains('flag')) {
        pill.nextElementSibling.remove();
        delete pill.dataset.hasFlag;
      }
      const m = rec.metrics || {};
      const d = detail || {};
      document.getElementById('stat-added-km2').textContent = formatStatKm2(m.added_km2);
      document.getElementById('stat-pct').textContent = formatStatPct(m.pct_increase);
      const priorT = d.towers_prior ?? m.prior_towers;
      const currentT = d.towers_current ?? m.current_towers;
      document.getElementById('stat-towers').textContent =
        priorT != null && currentT != null ? `${priorT} → ${currentT}` : '—';
      document.getElementById('stat-new-towers').textContent =
        d.new_towers ?? m.new_towers ?? '—';
    }

    function renderCompareMap(containerId, geo, countyFeature, emptyNoteId, sharedBounds) {
      const emptyEl = emptyNoteId ? document.getElementById(emptyNoteId) : null;
      const hasCoverage = (geo.features || []).some(f => f.properties?.kind !== 'tower');
      if (emptyEl) emptyEl.style.display = hasCoverage ? 'none' : 'flex';

      const countyGeo = countyFeature
        ? { type: 'FeatureCollection', features: [countyFeature] }
        : { type: 'FeatureCollection', features: [] };

      const mapObj = new maplibregl.Map({
        container: containerId,
        style: {
          version: 8,
          sources: {
            carto: {
              type: 'raster',
              tiles: ['https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
                      'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
                      'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png'],
              tileSize: 256,
              maxzoom: 19,
              attribution: '© <a href="https://carto.com/">CARTO</a> © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            }
          },
          layers: [{ id: 'carto', type: 'raster', source: 'carto', paint: { 'raster-opacity': 1.0 } }]
        },
        center: [-98.5, 39.0],
        zoom: 8,
        maxZoom: 14,
        pixelRatio: 2,
        attributionControl: false,
        interactive: true,
        preserveDrawingBuffer: true
      });
      mapObj.on('load', () => {
        if (countyFeature) {
          mapObj.addSource('county', { type: 'geojson', data: countyGeo });
          mapObj.addLayer({
            id: 'county-fill',
            type: 'fill',
            source: 'county',
            paint: { 'fill-color': '#e8f0fe', 'fill-opacity': 0.18 }
          });
          mapObj.addLayer({
            id: 'county-outline',
            type: 'line',
            source: 'county',
            paint: { 'line-color': '#2563eb', 'line-width': 1.5, 'line-opacity': 0.7 }
          });
        }
        mapObj.addSource('coverage', { type: 'geojson', data: geo });
        mapObj.addLayer({
          id: 'coverage-fill',
          type: 'fill',
          source: 'coverage',
          filter: ['!=', ['get', 'kind'], 'tower'],
          paint: {
            'fill-color': ['get', 'color'],
            'fill-opacity': 0.85,
            // Match the fill color on the outline so adjacent same-signal hexes
            // blend into a smooth heatmap (no visible hex grid lines).
            'fill-outline-color': ['get', 'color']
          }
        });
        mapObj.addLayer({
          id: 'tower-points',
          type: 'circle',
          source: 'coverage',
          filter: ['all', ['==', ['get', 'kind'], 'tower'], ['==', ['get', 'in_county'], true]],
          paint: {
            'circle-radius': 6,
            'circle-color': [
              'match', ['get', 'site_class'],
              'new_site', TOWER_COLORS.new_site,
              'expanded_site', TOWER_COLORS.expanded_site,
              'prior_site', TOWER_COLORS.prior_site,
              '#38bdf8'
            ],
            'circle-stroke-width': 2,
            'circle-stroke-color': '#0f172a'
          }
        });
        mapObj.addLayer({
          id: 'tower-points-cross',
          type: 'circle',
          source: 'coverage',
          filter: ['all', ['==', ['get', 'kind'], 'tower'], ['==', ['get', 'in_county'], false]],
          paint: {
            'circle-radius': 7,
            'circle-color': [
              'match', ['get', 'site_class'],
              'new_site', TOWER_COLORS.new_site,
              'expanded_site', TOWER_COLORS.expanded_site,
              'prior_site', TOWER_COLORS.prior_site,
              '#38bdf8'
            ],
            'circle-stroke-width': 3,
            'circle-stroke-color': '#f8fafc'
          }
        });
        if (countyFeature) {
          mapObj.addLayer({
            id: 'county-outline-strong',
            type: 'line',
            source: 'county',
            paint: {
              'line-color': '#111827',
              'line-width': 3.5,
              'line-opacity': 1
            }
          });
        }
        const b = countyBounds(countyFeature) || sharedBounds || boundsFromGeoJSON(geo);
        if (b) mapObj.fitBounds(b, { padding: 36, maxZoom: 14, duration: 0 });
        requestAnimationFrame(() => mapObj.resize());
      });
      return mapObj;
    }

    async function loadCountyDetail(rec) {
      const pid = rec.provider_id || currentProvider;
      const svc = rec.service || currentService;
      const geoid = rec.geoid;
      const key = `${pid}/${safeServiceKey(svc)}/${geoid}`;
      if (detailCache[key]) return detailCache[key];
      const api = `/api/county?geoid=${encodeURIComponent(geoid)}`
        + `&provider=${encodeURIComponent(pid)}`
        + `&service=${encodeURIComponent(svc)}`;
      let res = await fetch(api);
      if (!res.ok) {
        const url = `${DATA_BASE}/details/${pid}/${safeServiceKey(svc)}/${geoid}.json`;
        res = await fetch(url);
      }
      if (!res.ok) return null;
      const data = await res.json();
      detailCache[key] = data;
      return data;
    }


    if (typeof maplibregl === 'undefined' || typeof h3 === 'undefined') {
      const err = document.getElementById('cdn-error');
      err.classList.add('visible');
      err.innerHTML = '<strong>Map libraries failed to load</strong> (MapLibre / h3 from <code>web/vendor/</code>). '
        + 'Serve from the <code>web/</code> directory: <code>python3 -m http.server 8000</code>.';
      document.getElementById('app-loading')?.classList.add('hidden');
      throw new Error('Vendor libraries missing');
    }

    const map = new maplibregl.Map({
      container: 'map',
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap'
          }
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }]
      },
      center: [-98.5, 39.0],
      zoom: 4,
      maxZoom: 12
    });

    let useSplitRecords = false;   // set to true if per-provider split files are used

    async function loadData() {
      const [metaRes, recRes, geoRes] = await Promise.all([
        fetch(`${DATA_BASE}/meta.json`),
        fetch(`${DATA_BASE}/records.json`),
        fetch(`${DATA_BASE}/counties.geojson`)
      ]);
      if (!metaRes.ok || !geoRes.ok) {
        throw new Error('Missing web bundle. Run: python -m fcc_audit.cli build-web');
      }
      meta = await metaRes.json();
      countiesGeo = await geoRes.json();
      // Prefer split records whenever meta says so OR the monolith is absent.
      // Never prefer a stale records.json over a present records/ tree.
      if (meta.use_split_records || !recRes.ok) {
        useSplitRecords = true;
        records = {};
      } else {
        // Probe one split path; if it exists, prefer split over monolith.
        const probePid = meta.providers?.[0]?.id;
        const probeSvc = meta.services?.[0];
        let splitOk = false;
        if (probePid != null && probeSvc) {
          try {
            const probe = await fetch(
              `${DATA_BASE}/records/${probePid}/${safeServiceKey(probeSvc)}.json`,
              { method: 'HEAD' }
            );
            splitOk = probe.ok;
          } catch {}
        }
        if (splitOk || meta.use_split_records) {
          useSplitRecords = true;
          records = {};
        } else {
          records = await recRes.json();
        }
      }
      updateScopeBanner();
    }

    function updateScopeBanner() {
      // Scope note folded into the header (banner removed).
      const el = document.getElementById('vintage-label');
      if (!el || !meta) return;
      const scope = String(meta.states_processed || '');
      const incomplete = !!meta.incomplete || (scope && scope !== 'all' && !scope.includes(','));
      const cur = formatVintage(meta.current_vintage) || meta.current_vintage || meta.current || '?';
      const pri = formatVintage(meta.prior_vintage) || meta.prior_vintage || meta.prior || '?';
      let base = `${cur} vs ${pri}`;
      if (scope) base += ` · ${formatScope(scope)}`;
      if (incomplete) base += ' · partial run';
      el.textContent = base;
    }

    async function ensureProviderRecordsLoaded(pid, svc) {
      if (!useSplitRecords) return true;
      if (records[pid] && records[pid][svc]) return true;
      const url = `${DATA_BASE}/records/${pid}/${safeServiceKey(svc)}.json`;
      try {
        const res = await fetch(url);
        if (res.ok) {
          records[pid] = records[pid] || {};
          records[pid][svc] = await res.json();
          return true;
        }
        const msg = res.status === 404
          ? `No records file for provider ${pid} / ${svc} (404). Re-run build-web, or pick another service.`
          : `Failed to load records for provider ${pid} / ${svc} (HTTP ${res.status}).`;
        document.getElementById('list').innerHTML = `<div class="err">${msg}</div>`;
        return false;
      } catch (err) {
        document.getElementById('list').innerHTML =
          `<div class="err">Network error loading records for ${pid}/${svc}: ${err.message || err}</div>`;
        return false;
      }
    }

    /** When meta lacks provider_services, probe which split record files exist. */
    async function resolveServicesForProvider(pid) {
      const declared = meta.provider_services?.[String(pid)];
      if (Array.isArray(declared) && declared.length) return declared;
      const loaded = records[String(pid)];
      if (loaded && Object.keys(loaded).length) return Object.keys(loaded);
      const candidates = meta.services || [];
      if (!useSplitRecords || !candidates.length) return candidates;
      const available = [];
      for (const s of candidates) {
        try {
          const res = await fetch(
            `${DATA_BASE}/records/${pid}/${safeServiceKey(s)}.json`,
            { method: 'HEAD' },
          );
          if (res.ok) available.push(s);
        } catch { /* ignore */ }
      }
      if (available.length) {
        meta.provider_services = meta.provider_services || {};
        meta.provider_services[String(pid)] = available;
        return available;
      }
      return candidates;
    }

    function providerRecords(pid, svc) {
      return (records[pid] && records[pid][svc]) ? records[pid][svc] : {};
    }

    function preferredProviderId() {
      if (meta.default_provider_id != null) return String(meta.default_provider_id);
      let best = null;
      let bestFlagged = -1;
      const svc = (meta.services || [])[0];
      (meta.providers || []).forEach(p => {
        const pid = String(p.id);
        const n = Object.values(providerRecords(pid, svc)).filter(r => r.flag).length;
        if (n > bestFlagged) {
          bestFlagged = n;
          best = pid;
        }
      });
      return best || String(meta.providers?.[0]?.id || '');
    }

    function openDemoCounty() {
      const geoid = meta.default_county_geoid;
      if (!geoid) return;
      if (meta.default_provider_id != null) {
        const pid = String(meta.default_provider_id);
        document.getElementById('provider').value = pid;
        currentProvider = pid;
      }
      const rec = providerRecords(currentProvider, currentService)[geoid];
      if (!rec) return;
      showDetail(rec);
      flyToCounty(geoid);
    }

    function fitMapToCounties() {
      if (!countiesGeo?.features?.length) return;
      const b = boundsFromGeoJSON(countiesGeo);
      if (b) map.fitBounds(b, { padding: 48, maxZoom: 9, duration: 0 });
    }

    function fitMapToUS() {
      // Zoom to the contiguous US on initial load so the user picks where to look.
      map.fitBounds([[-125.0, 24.5], [-66.5, 49.5]], { padding: 32, duration: 600 });
    }

    function servicesForProvider(pid) {
      const declared = meta.provider_services?.[String(pid)];
      if (Array.isArray(declared) && declared.length) return declared;
      const loaded = records[String(pid)];
      if (loaded && Object.keys(loaded).length) return Object.keys(loaded);
      return meta.services || [];
    }

    function fillServiceDropdown(pid, preferredService = null, servicesOverride = null) {
      const sSel = document.getElementById('service');
      const services = servicesOverride || servicesForProvider(pid);
      sSel.innerHTML = '';
      services.forEach(s => {
        const o = document.createElement('option');
        o.value = s;
        o.textContent = s;
        sSel.appendChild(o);
      });
      if (preferredService && services.includes(preferredService)) {
        sSel.value = preferredService;
      }
      return sSel.value;
    }

    async function fillDropdowns() {
      const pSel = document.getElementById('provider');
      pSel.innerHTML = '';
      (meta.providers || []).forEach(p => {
        const o = document.createElement('option');
        o.value = String(p.id);
        o.textContent = p.name;
        pSel.appendChild(o);
      });
      pSel.value = preferredProviderId();
      currentProvider = pSel.value;
      const services = await resolveServicesForProvider(currentProvider);
      currentService = fillServiceDropdown(currentProvider, null, services);
      updateScopeBanner();
      document.getElementById('n-providers').textContent = (meta.providers || []).length;
    }

    function colorForRecord(rec) { return colorForRecordImported(rec); }

    function applyFeatureStates() {
      if (!map.getSource('counties')) return;
      const recs = providerRecords(currentProvider, currentService);
      const flaggedOnly = document.getElementById('flagged-only').checked;
      countiesGeo.features.forEach(f => {
        const geoid = f.properties.geoid;
        const rec = recs[geoid];
        const hasData = !!rec;
        const show = hasData && (!flaggedOnly || rec.flag);
        map.setFeatureState(
          { source: 'counties', id: geoid },
          {
            hasData: show,
            flag: !!rec?.flag,
            priority: rec?.priority || 0,
            color: colorForRecord(rec)
          }
        );
      });
    }

    function setupMapLayers() {
      if (map.getSource('counties')) {
        applyFeatureStates();
        return;
      }
      map.addSource('counties', {
        type: 'geojson',
        data: countiesGeo,
        promoteId: 'geoid'
      });
      map.addLayer({
        id: 'counties-fill',
        type: 'fill',
        source: 'counties',
        paint: {
          'fill-color': ['case', ['feature-state', 'hasData'], ['feature-state', 'color'], '#1e293b'],
          'fill-opacity': ['case', ['feature-state', 'hasData'], 0.72, 0.25]
        }
      });
      map.addLayer({
        id: 'counties-outline',
        type: 'line',
        source: 'counties',
        paint: {
          'line-color': '#94a3b8',
          'line-width': 0.6,
          'line-opacity': ['case', ['feature-state', 'hasData'], 0.9, 0.3]
        }
      });
      map.on('click', 'counties-fill', e => {
        const geoid = e.features[0].properties.geoid;
        const rec = providerRecords(currentProvider, currentService)[geoid];
        if (rec) {
          document.querySelectorAll('.row').forEach(el => {
            el.classList.toggle('active', el.dataset.geoid === geoid);
          });
          showDetail(rec);
          flyToCounty(geoid);
        }
      });
      map.on('mouseenter', 'counties-fill', () => map.getCanvas().style.cursor = 'pointer');
      map.on('mouseleave', 'counties-fill', () => map.getCanvas().style.cursor = '');
      applyFeatureStates();
    }

    function renderList() {
      const recs = providerRecords(currentProvider, currentService);
      const q = document.getElementById('search').value.toLowerCase();
      let rows = Object.values(recs);
      if (!listShowAll) rows = rows.filter(r => r.flag);
      rows = rows
        .filter(r => !q || r.name.toLowerCase().includes(q))
        .sort((a, b) => (b.priority || 0) - (a.priority || 0));

      document.getElementById('n-scored').textContent = Object.keys(recs).length;
      document.getElementById('n-flagged').textContent = Object.values(recs).filter(r => r.flag).length;
      document.getElementById('list-head-label').textContent =
        listShowAll ? 'All analyzed counties' : 'Flagged for review';
      document.getElementById('list-mode-toggle').textContent = listShowAll ? 'Flagged only' : 'Show all';

      const list = document.getElementById('list');
      list.innerHTML = '';
      if (!rows.length) {
        const n = Object.keys(recs).length;
        const flaggedN = Object.values(recs).filter(r => r.flag).length;
        list.innerHTML = n
          ? (flaggedN === 0 && !listShowAll
            ? `<div class="info-msg">${n} ${n === 1 ? 'county' : 'counties'} analyzed — none flagged for this provider/service. Use <strong>Show all</strong> to browse every county.</div>`
            : '<div class="info-msg">No counties match your search.</div>')
          : '<div class="err">No county data for this provider/service yet. Run the pipeline with <code>--build-web</code>.</div>';
        return;
      }
      rows.forEach(r => {
        const div = document.createElement('div');
        div.className = 'row';
        div.dataset.geoid = r.geoid;
        div.setAttribute('role', 'button');
        div.tabIndex = 0;
        const sevLabel = r.explanation?.severity || '';

        const top = document.createElement('div');
        top.className = 'top';
        const nameSpan = document.createElement('span');
        nameSpan.className = 'name';
        nameSpan.textContent = r.name;
        const sevPill = document.createElement('span');
        sevPill.className = `pill ${severityPillClass(sevLabel)}`;
        sevPill.textContent = sevLabel;
        nameSpan.append(sevPill);
        if (r.flag) {
          const flagPill = document.createElement('span');
          flagPill.className = 'pill flag';
          flagPill.textContent = 'FLAG';
          nameSpan.append(flagPill);
        }
        // Decision badge
        const dKey = decisionKey(r.geoid, currentProvider, currentService);
        const dec = decisions[dKey];
        if (dec) {
          const db = document.createElement('span');
          db.className = `decision-badge ${dec.verdict}`;
          db.textContent = dec.verdict === 'needs-info' ? '? INFO' : dec.verdict.toUpperCase();
          nameSpan.append(db);
        }
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'score';
        scoreSpan.textContent = (r.priority || 0).toFixed(2);
        top.append(nameSpan, scoreSpan);

        const sub = document.createElement('div');
        sub.className = 'sub';
        sub.textContent = r.explanation?.headline || '';

        div.append(top, sub);
        const activate = () => {
          document.querySelectorAll('.row').forEach(el => el.classList.remove('active'));
          div.classList.add('active');
          currentDetailRec = r;
          showDetail(r);
          flyToCounty(r.geoid);
          updateTriagePanel(r);
        };
        div.onclick = activate;
        div.onkeydown = e => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); }
        };
        list.appendChild(div);
      });
    }

    function flyToCounty(geoid) {
      const f = countiesGeo.features.find(x => x.properties.geoid === geoid);
      if (!f) return;
      const bounds = new maplibregl.LngLatBounds();
      const extendRing = ring => ring.forEach(c => bounds.extend(c));
      const geom = f.geometry;
      if (geom.type === 'Polygon') {
        extendRing(geom.coordinates[0]);
      } else if (geom.type === 'MultiPolygon') {
        geom.coordinates.forEach(poly => extendRing(poly[0]));
      }
      if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 80, maxZoom: 9 });
    }

    function setTowerCompare(prior, current, newCount, crossBorder, rec) {
      const tc = document.getElementById('tower-compare');
      tc.style.display = 'grid';
      document.getElementById('tower-prior-label').textContent =
        formatVintage(meta.prior_vintage) || meta.prior_vintage || 'Prior map';
      document.getElementById('tower-current-label').textContent =
        formatVintage(meta.current_vintage) || meta.current_vintage || 'Current map';
      document.getElementById('tower-prior-count').textContent = prior ?? '—';
      document.getElementById('tower-current-count').textContent = current ?? '—';
      const delta = newCount != null ? newCount : (current ?? 0) - (prior ?? 0);
      document.getElementById('tower-delta-label').textContent =
        delta > 0 ? `inferred (+${delta} new)` : 'inferred (modeled)';
      const note = document.getElementById('tower-cross-border-note');
      const cb = crossBorder || {};
      const curX = cb.current || 0;
      const newX = cb.new || 0;
      const panel = formatTowerPanel(rec || {});
      const bits = [panel.inferredNote];
      if (curX > 0 || newX > 0) {
        bits.push(curX > 0
          ? `${curX} neighboring-county site${curX !== 1 ? 's' : ''} shown with a white ring.`
          : `${newX} new site${newX !== 1 ? 's' : ''} outside this county explain part of the growth here.`);
      }
      note.style.display = 'block';
      note.textContent = bits.join(' ');
      const asrNote = document.getElementById('tower-asr-note');
      if (asrNote) {
        if (panel.asrLine) {
          asrNote.style.display = 'block';
          asrNote.textContent = panel.asrLine;
        } else {
          asrNote.style.display = 'none';
          asrNote.textContent = '';
        }
      }
    }

    function showDetail(rec) {
      const reqId = ++detailRequestId;
      const d = document.getElementById('detail');
      d.classList.add('visible');
      fillDetailHeader(rec);
      document.getElementById('detail-headline').textContent = rec.explanation?.headline || '';
      const ul = document.getElementById('detail-bullets');
      ul.innerHTML = '';
      (rec.explanation?.bullets || []).forEach(b => {
        const li = document.createElement('li');
        li.textContent = b;
        ul.appendChild(li);
      });
      document.getElementById('detail-rec').textContent = rec.explanation?.recommendation || '';

      // Flag-math panel
      renderFlagMath(rec);

      const priorT = rec.metrics?.prior_towers;
      const currentT = rec.metrics?.current_towers;
      if (priorT != null || currentT != null) {
        setTowerCompare(priorT, currentT, rec.metrics?.new_towers, {
          current: rec.metrics?.current_towers_cross_border,
          new: rec.metrics?.new_towers_cross_border,
        }, rec);
      } else {
        document.getElementById('tower-compare').style.display = 'none';
        document.getElementById('tower-cross-border-note').style.display = 'none';
        const asrNote = document.getElementById('tower-asr-note');
        if (asrNote) asrNote.style.display = 'none';
      }

      const statusEl = document.getElementById('detail-map-status');
      statusEl.style.display = 'flex';
      statusEl.innerHTML = '<span class="spinner"></span><span>Loading coverage detail…</span>';
      statusEl.className = 'detail-loading';
      document.getElementById('compare-maps').style.display = 'none';
      document.getElementById('swipe-compare').style.display = 'none';
      document.getElementById('compare-toolbar').style.display = 'none';
      document.getElementById('signal-gradient-legend').style.display = 'none';
      destroyCompareMaps();

      // Show PNGs immediately (pipeline always writes prior.png / current.png).
      showPredictedRasterCompare(rec);
      statusEl.style.display = 'none';

      document.getElementById('map-prior-label').textContent =
        `${formatVintage(meta.prior_vintage) || 'Prior'} — ${rec.service} coverage`;
      document.getElementById('map-current-label').textContent =
        `${formatVintage(meta.current_vintage) || 'Current'} — ${rec.service} coverage`;
      document.getElementById('swipe-prior-label').textContent =
        formatVintage(meta.prior_vintage) || 'Prior';
      document.getElementById('swipe-current-label').textContent =
        formatVintage(meta.current_vintage) || 'Current';

      loadCountyDetail(rec).then(detail => {
        if (reqId !== detailRequestId) return;
        if (!detail) {
          statusEl.style.display = 'flex';
          const topN = meta?.top_n || 250;
          const outsideTop = rec.tier == null;
          statusEl.innerHTML = outsideTop
            ? `Coverage detail maps are only generated for the top ${topN} counties per provider×service. This county is outside that set — rankings and flag math above are still valid.`
            : 'Coverage detail not available for this county. Re-run <code>python -m fcc_audit.cli build-web</code> after batches complete.';
          statusEl.className = 'detail-missing';
          return;
        }
        fillDetailHeader(rec, detail);
        if (detail.towers_prior != null) {
          setTowerCompare(
            detail.towers_prior, detail.towers_current, detail.new_towers,
            { current: detail.current_towers_cross_border, new: detail.new_towers_cross_border },
            rec,
          );
        }
        document.getElementById('map-prior-label').textContent =
          `${formatVintage(detail.prior_vintage) || detail.prior_vintage || 'Prior'} — ${rec.service} coverage`;
        document.getElementById('map-current-label').textContent =
          `${formatVintage(detail.current_vintage) || detail.current_vintage || 'Current'} — ${rec.service} coverage`;

        if (detail.prior_map && detail.current_map) {
          showRasterCompare(detail, rec);
          return;
        }

        if (!h3lib?.cellToBoundary) {
          statusEl.style.display = 'flex';
          statusEl.innerHTML = 'Interactive map fallback unavailable; raster maps shown when present.';
          statusEl.className = 'detail-missing';
          return;
        }

        document.getElementById('compare-toolbar').style.display = 'none';
        setCompareMode('map');
        document.getElementById('compare-maps').style.display = 'grid';
        document.getElementById('signal-gradient-legend').style.display = 'grid';
        const estNote = document.getElementById('signal-estimated-note');
        if (detail.signal_flat && !detail.signal_estimated) {
          estNote.style.display = 'block';
          estNote.textContent = 'Coverage is binary (flat 0/1) with no usable signal band — hexes shown in neutral gray, not signal strength.';
        } else {
          estNote.style.display = detail.signal_estimated ? 'block' : 'none';
          if (detail.signal_estimated) {
            estNote.textContent = estNote.dataset.defaultText || estNote.textContent;
          }
        }
        const countyFeature = resolveCountyFeature(rec.geoid, detail);
        const flatOpts = { flat: !!detail.signal_flat && !detail.signal_estimated };
        const priorGeo = hexesToGeoJSON(detail.prior_hexes, detail.sites_prior, flatOpts);
        const currentGeo = hexesToGeoJSON(detail.current_hexes, detail.sites_current, flatOpts);
        const sharedBounds = unionBounds(
          countyBounds(countyFeature),
          boundsFromGeoJSON(priorGeo),
          boundsFromGeoJSON(currentGeo),
        );
        compareMaps.prior = renderCompareMap('map-prior', priorGeo, countyFeature, 'map-prior-empty', sharedBounds);
        compareMaps.current = renderCompareMap('map-current', currentGeo, countyFeature, 'map-current-empty', sharedBounds);
        setTimeout(() => {
          compareMaps.prior?.resize();
          compareMaps.current?.resize();
        }, 100);
      }).catch(() => {
        if (reqId !== detailRequestId) return;
        statusEl.style.display = 'none';
      });
    }

    async function loadTowers(show) {
      towerMarkers.forEach(m => m.remove());
      towerMarkers = [];
      if (!show) return;
      const pid = currentProvider;
      if (!towersCache[pid]) {
        try {
          const res = await fetch(`${DATA_BASE}/towers/${pid}.json`);
          towersCache[pid] = res.ok ? await res.json() : [];
        } catch { towersCache[pid] = []; }
      }
      const svc = currentService;
      towersCache[pid].filter(t =>
        (!t.service || t.service === svc) && (!t.vintage || t.vintage === 'current')
      ).forEach(t => {
        const el = document.createElement('div');
        const color = towerColor(t.site_class);
        el.style.cssText = `width:9px;height:9px;border-radius:50%;background:${color};border:1.5px solid #0f172a`;
        towerMarkers.push(new maplibregl.Marker({ element: el }).setLngLat([t.lng, t.lat]).addTo(map));
      });
    }

    async function onSelectionChange() {
      const nextProvider = document.getElementById('provider').value;
      if (nextProvider !== currentProvider) {
        const services = await resolveServicesForProvider(nextProvider);
        fillServiceDropdown(nextProvider, currentService, services);
      }
      currentProvider = nextProvider;
      currentService = document.getElementById('service').value;
      currentDetailRec = null;
      document.getElementById('detail').classList.remove('visible');
      updateTriagePanel(null);
      destroyCompareMaps();
      if (useSplitRecords) {
        await ensureProviderRecordsLoaded(currentProvider, currentService);
      }
      initThresholdSlider();
      applyFeatureStates();
      renderList();
      loadTowers(document.getElementById('show-towers').checked);
    }

    document.getElementById('provider').addEventListener('change', onSelectionChange);
    document.getElementById('service').addEventListener('change', onSelectionChange);
    document.getElementById('search').addEventListener('input', renderList);
    document.getElementById('flagged-only').addEventListener('change', e => {
      listShowAll = !e.target.checked;
      renderList();
      applyFeatureStates();
    });
    document.getElementById('show-towers').addEventListener('change', e => loadTowers(e.target.checked));
    document.getElementById('legend-toggle').addEventListener('click', () => {
      const legend = document.getElementById('map-legend');
      const collapsed = legend.classList.toggle('collapsed');
      document.getElementById('legend-toggle').setAttribute('aria-expanded', String(!collapsed));
    });
    document.getElementById('detail-close').addEventListener('click', () => {
      document.getElementById('detail').classList.remove('visible');
      destroyCompareMaps();
    });
    document.getElementById('list-mode-toggle').addEventListener('click', () => {
      listShowAll = !listShowAll;
      document.getElementById('flagged-only').checked = !listShowAll;
      renderList();
      applyFeatureStates();
    });
    document.getElementById('btn-side-by-side').addEventListener('click', () => setCompareViewLayout('side'));
    document.getElementById('btn-swipe').addEventListener('click', () => setCompareViewLayout('swipe'));
    document.getElementById('btn-download').addEventListener('click', downloadComparison);
    document.getElementById('swipe-slider').addEventListener('input', e => {
      updateSwipePosition(Number(e.target.value));
    });

    // ── Decisions: persistence + triage ─────────────────────────────────
    let decisions = {};
    let currentDetailRec = null;
    function decisionsStorageKey() {
      const cur = meta?.current_vintage || 'unknown';
      const pri = meta?.prior_vintage || 'unknown';
      return `fcc-audit-decisions:${cur}|${pri}`;
    }

    function decisionKey(geoid, pid, svc) {
      return `${pid}/${safeServiceKey(svc || '')}/${geoid}`;
    }

    function loadDecisions() {
      try { decisions = JSON.parse(localStorage.getItem(decisionsStorageKey()) || '{}'); } catch { decisions = {}; }
    }

    function saveDecision(verdict, note) {
      if (!currentDetailRec) return;
      const key = decisionKey(currentDetailRec.geoid, currentProvider, currentService);
      decisions[key] = {
        verdict, note: note || '',
        geoid: currentDetailRec.geoid,
        name: currentDetailRec.name,
        provider_id: currentProvider,
        service: currentService,
        priority: currentDetailRec.priority,
        flag: currentDetailRec.flag,
        timestamp: new Date().toISOString(),
      };
      localStorage.setItem(decisionsStorageKey(), JSON.stringify(decisions));
      updateTriagePanel(currentDetailRec);
      renderList();
    }

    function setDecision(verdict) {
      const note = document.getElementById('triage-note').value;
      saveDecision(verdict, note);
    }

    function saveCurrentNote() {
      if (!currentDetailRec) return;
      const key = decisionKey(currentDetailRec.geoid, currentProvider, currentService);
      const note = document.getElementById('triage-note').value;
      if (decisions[key]) {
        decisions[key].note = note;
        localStorage.setItem(decisionsStorageKey(), JSON.stringify(decisions));
      }
    }

    function updateTriagePanel(rec) {
      const section = document.getElementById('triage-section');
      const hint = document.getElementById('kbd-hint');
      section.style.display = rec ? '' : 'none';
      hint.style.display = rec ? '' : 'none';
      if (!rec) return;
      const key = decisionKey(rec.geoid, currentProvider, currentService);
      const dec = decisions[key] || {};
      ['accept', 'reject', 'needs-info'].forEach(v => {
        const btn = document.getElementById(
          v === 'accept' ? 'btn-accept' : v === 'reject' ? 'btn-reject' : 'btn-needsinfo'
        );
        btn.classList.toggle('active', dec.verdict === v);
      });
      document.getElementById('triage-note').value = dec.note || '';
    }

    function downloadDecisions() {
      const entries = Object.entries(decisions).map(([k, v]) => ({
        key: k,
        verdict: v.verdict,
        note: v.note,
        geoid: v.geoid,
        county_name: v.name,
        provider_id: v.provider_id,
        service: v.service,
        priority_score: v.priority,
        was_flagged: v.flag,
        decided_at: v.timestamp,
        current_vintage: meta?.current_vintage,
        prior_vintage: meta?.prior_vintage,
      }));
      const blob = new Blob([JSON.stringify(entries, null, 2)], {type: 'application/json'});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `fcc_audit_decisions_${new Date().toISOString().slice(0,10)}.json`;
      a.click();
    }

    // ── Keyboard triage ─────────────────────────────────────────────────
    function navigateList(direction) {
      const rows = Array.from(document.querySelectorAll('#list .row'));
      if (!rows.length) return;
      const active = document.querySelector('#list .row.active');
      let idx = active ? rows.indexOf(active) : -1;
      idx = Math.max(0, Math.min(rows.length - 1, idx + direction));
      rows[idx].click();
      rows[idx].scrollIntoView({ block: 'nearest' });
    }

    // ── Flag-math tooltip ────────────────────────────────────────────────
    function renderFlagMath(rec) {
      renderFlagMathInto({
        panel: document.getElementById('flag-math-panel'),
        scoreFill: document.getElementById('fm-score-fill'),
        threshLine: document.getElementById('fm-threshold-line'),
        scoreLabel: document.getElementById('fm-score-label'),
        badge: document.getElementById('fm-gates'),
        list: document.getElementById('fm-features-list'),
      }, rec?.flag_math);
    }

    // ── Display cutoff slider (does NOT change official pipeline flags) ──
    let displayCutoff = null;

    function applyLiveThreshold(threshVal) {
      displayCutoff = threshVal;
      const tv = document.getElementById('thresh-val');
      if (tv) tv.textContent = threshVal.toFixed(3);
      if (!currentProvider || !currentService) return;
      const recs = providerRecords(currentProvider, currentService);
      // Official flagged count always comes from pipeline `flag`.
      document.getElementById('n-flagged').textContent =
        Object.values(recs).filter(r => r.flag).length;
      applyFeatureStates();
      renderList();
    }

    function initThresholdSlider() {
      // Display-cutoff slider removed — official flag comes from the pipeline only.
      return;
      const recs = providerRecords(currentProvider, currentService);
      const scores = Object.values(recs).map(r => r.priority || 0).filter(v => v > 0).sort((a,b) => a - b);
      const slider = document.getElementById('thresh-slider');
      slider.min   = '0';
      slider.max   = '1';
      slider.step  = '0.001';
      const metaThresh = (meta && meta.flag_threshold != null) ? Number(meta.flag_threshold) : null;
      if (!scores.length && metaThresh == null) {
        slider.value = '0';
        document.getElementById('thresh-val').textContent = '—';
        displayCutoff = null;
        return;
      }
      const idx   = Math.min(Math.floor(scores.length * 0.9), Math.max(0, scores.length - 1));
      const pct90 = metaThresh != null && Number.isFinite(metaThresh) ? metaThresh : (scores[idx] ?? 0);
      slider.value = pct90.toFixed(3);
      applyLiveThreshold(pct90);
    }

    // Display-cutoff slider removed from the DOM.

    // Single keyboard handler
    document.addEventListener('keydown', e => {
      const tag = document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      const detailVisible = document.getElementById('detail').classList.contains('visible');
      if (e.key === 'j') { e.preventDefault(); navigateList(1); }
      else if (e.key === 'k') { e.preventDefault(); navigateList(-1); }
      else if (detailVisible && e.key === 'a' && currentDetailRec) setDecision('accept');
      else if (detailVisible && e.key === 'r' && currentDetailRec) setDecision('reject');
      else if (detailVisible && e.key === 'n' && currentDetailRec) setDecision('needs-info');
      else if (e.key === 'Escape') {
        const d = document.getElementById('detail');
        if (d.classList.contains('visible')) {
          d.classList.remove('visible');
          destroyCompareMaps();
          currentDetailRec = null;
          updateTriagePanel(null);
          document.querySelectorAll('.row').forEach(el => el.classList.remove('active'));
        }
      }
    });

    map.on('load', async () => {
      const loader = document.getElementById('app-loading');
      try {
        await loadData();
        loadDecisions();
        await fillDropdowns();
        setupMapLayers();
        if (useSplitRecords) {
          await ensureProviderRecordsLoaded(currentProvider, currentService);
          // Re-pick default provider now that flag counts are available (F36).
          if (meta.default_provider_id == null) {
            const best = preferredProviderId();
            if (best && best !== currentProvider) {
              document.getElementById('provider').value = best;
              currentProvider = best;
              const services = await resolveServicesForProvider(currentProvider);
              currentService = fillServiceDropdown(currentProvider, null, services);
              await ensureProviderRecordsLoaded(currentProvider, currentService);
            }
          }
        }
        initThresholdSlider();
        renderList();
        fitMapToUS();
        openDemoCounty();
        loader.classList.add('hidden');
        setTimeout(() => loader.remove(), 400);
      } catch (e) {
        loader.classList.add('hidden');
        document.getElementById('list').innerHTML =
          `<div class="err"><strong>Could not load data.</strong><br>${e.message}<br><br>`
          + 'Run the pipeline with <code>--build-web</code>, then serve from the <code>web/</code> folder:<br>'
          + '<code>python -m http.server 8000</code></div>';
      }
    });

    // Expose triage handlers for inline HTML onclick attributes.
    window.setDecision = setDecision;
    window.saveCurrentNote = saveCurrentNote;
    window.downloadDecisions = downloadDecisions;

