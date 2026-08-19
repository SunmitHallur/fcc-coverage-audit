# Mathematics of the coverage-change audit

This is the formula appendix for the pipeline in `src/fcc_audit/`.
The same content is summarized in the [README](../README.md#mathematics-of-each-step).

**What is from papers vs what is from this project.** Spatial indexing (H3),
equal-area projection, connected components, the discrete distance transform,
non-maximum suppression, k-d trees, and the Hungarian assignment method are
standard. Hex-area change metrics, relative-core site inference, cloverleaf
merge, same-site implausibility gates, and the bounded monotone score are
engineering choices. Weights and thresholds were calibrated on live Verizon
5G-NR 7/1 June 2025 → December 2025 filings (382 counties), preferring false
negatives over false positives. They are **not** a fit to a published fraud
model, and they are **not** frozen to old FCC slide labels when current
filings differ.

Symbols below match `config/pipeline.yaml` and the code. Distances and areas
use **NAD83 / Conus Albers** ([EPSG:5070](https://epsg.io/5070)).

---

## 0. Geographic primitives

### Why Albers, not lat/lng

Latitude–longitude is not an equal-area or equal-distance CRS. Area of a
claimed coverage jump and Euclidean distance between inferred masts would
stretch with latitude if computed in degrees. Snyder’s USGS manual is the
working reference for the Albers equal-area conic used as EPSG:5070
(NAD83 / Conus Albers):

- Snyder, J. P. (1987). *Map projections: A working manual.* USGS Professional Paper 1395. DOI [10.3133/pp1395](https://doi.org/10.3133/pp1395). Open PDF: <https://pubs.usgs.gov/pp/1395/report.pdf>
- EPSG:5070 (NAD83 / Conus Albers): <https://epsg.io/5070>

A coverage point $(\lambda,\varphi)$ (lon, lat, EPSG:4326) is projected to
meters $(x,y)$. Distances are Euclidean in that plane:

$$
d(a,b)=\sqrt{(x_a-x_b)^2+(y_a-y_b)^2}.
$$

### Why H3 hexagons

The FCC Broadband Data Collection already stores mobile coverage as H3
resolution-9 cells in the warehouse (`bbmap_mob_bb_mrgd_hex9_inter_*`). Using
the same discrete global grid means every later step (change, sites, flags)
is on the same cells the map product uses.

H3 is Uber’s implementation of a hexagonal discrete global grid (parent/child
aperture 7). The academic lineage is geodesic DGGS; the engineering spec is
H3:

- Sahr, K., White, D., & Kimerling, A. J. (2003). Geodesic Discrete Global Grid Systems. *Cartography and Geographic Information Science*, 30(2), 121–134. DOI [10.1559/152304003100011090](https://doi.org/10.1559/152304003100011090)
- Brodsky, I. (2018). H3: Uber’s Hexagonal Hierarchical Spatial Index. Uber Engineering. <https://www.uber.com/blog/h3/>
- Average hexagon area by resolution: <https://h3geo.org/docs/core-library/restable/>
- `polygonToCells` / Python `h3.geo_to_cells`: containment by **cell centroid**. <https://h3geo.org/docs/api/regions/>

Mean cell area at resolution $r$ (what the code calls
`h3.average_hexagon_area`, matching the H3 restable):

$$
A_{\mathrm{hex}}(r)=\texttt{h3.average\_hexagon\_area}(r)\;[\mathrm{km}^2].
$$

At $r=9$, $A_{\mathrm{hex}}=0.105332513\,\mathrm{km}^2$. Polygon backends
include a cell iff its centroid is inside the polygon. Redshift already did
that join; the pipeline skips polyfill.

Covered-area numerators therefore use H3’s spherical mean cell area; county
denominators use TIGER polygons projected to Albers (`geometry.area / 10^6`).
Those two CRS choices are mixed on purpose: hex counts stay on the FCC grid,
county land area stays equal-area.

FCC Broadband Data Collection: <https://www.fcc.gov/BroadbandData>

---

## 1. Acquire

No spatial formula. One layer is one `(provider, technology, mindown, vintage)`.
Warehouse rows already have `minsignal` (dBm). Cached parquet is `h3` +
`signal_dbm`.

**Why:** keep analysis independent of whether staff query Redshift or download
public shapefiles. Downstream formulas do not change.

---

## 2. Normalize (county join)

Each hex is tagged with Census TIGER county FIPS by point-in-polygon of the
cell center (Albers). Neighbor-state halo $\approx 50\,\mathrm{km}$ is
included on the Redshift path so a mast just across a state line can explain
in-county growth.

**Why not a spherical buffer in degrees:** 50 km is a physical search radius
for macro coverage, so it must be meters in an equal-area CRS.

TIGER/Line: <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>

---

## 3. Change detection (`changedetect.py`)

Let $P$ and $C$ be the sets of covered hexes in prior and current vintages,
with signals $s_P(h), s_C(h)$ in dBm when present.

$$
\begin{aligned}
\mathrm{new} &= C\setminus P,\\
\mathrm{lost} &= P\setminus C,\\
\mathrm{upgraded} &= \{h\in P\cap C: s_C(h)-s_P(h)\ge 5\,\mathrm{dB}\},\\
\mathrm{downgraded} &= \{h\in P\cap C: s_P(h)-s_C(h)\ge 5\,\mathrm{dB}\}.
\end{aligned}
$$

The 5 dB gate is an engineering threshold: smaller swings are ordinary band
noise, not a technology upgrade. It is **not** a 3GPP requirement.

County rollup (geoid $g$):

$$
\begin{aligned}
\mathrm{prior\_km}^2(g) &= |\{h\in P: \mathrm{county}(h)=g\}|\cdot A_{\mathrm{hex}},\\
\mathrm{current\_km}^2(g) &= |\{h\in C: \mathrm{county}(h)=g\}|\cdot A_{\mathrm{hex}},\\
\mathrm{added\_km}^2(g) &= \mathrm{current\_km}^2-\mathrm{prior\_km}^2,\\
f_{\mathrm{added}}(g) &= \frac{\mathrm{added\_km}^2}{\mathrm{area}(g)},\\
\rho(g) &= \begin{cases}
\mathrm{added\_km}^2/\mathrm{prior\_km}^2 & \mathrm{prior\_km}^2>0,\\
+\infty & \mathrm{prior\_km}^2=0\text{ and }\mathrm{added\_km}^2>0,\\
0 & \text{otherwise}.
\end{cases}
\end{aligned}
$$

**Why count hexes × mean hex area rather than dissolve polygons:** at national
scale the dissolve is expensive, and H3 cells already tile without overlap.
The bias vs true polygonal area is bounded by hex quantization
($\sim 0.1\,\mathrm{km}^2$ at res 9), which is small next to the
$10\,\mathrm{km}^2$ flag floor.

The binary **flag** uses this **net** `added_km2`. Attribution shares (next
sections) are counted only on hexes with status `new` or `upgraded`, so they
are not the same quantity.

---

## 4. Site inference (`towers.py`)

Cell sites are **not published**. Coverage from one sectorized site is a lobe
(or cloverleaf) with a strong core. The pipeline infers a pin from geometry,
then optionally snaps it onto FCC ASR.

This is **not** the Hata / COST-231 path-loss model. Those predict RSSI from
distance, frequency, and antenna height. Here the filing already *is* a
modeled heatmap; we invert shape → probable mast, then attribute growth.

Cellular sector geometry (why cloverleafs exist): typical macros use three
$\approx 120^\circ$ sectors. Rappaport’s textbook is the standard reference
for that architecture, not a formula we copy:

- Rappaport, T. S. *Wireless Communications: Principles and Practice.*

### 4a. Relative core

Providers file different dBm schemes (some cores at $-50$, others at
$-110$, some binary 0/1). A global cutoff would invent extra sites for hot
filers and miss cold ones. Keep the strongest fraction of **this**
provider × vintage layer:

Walk this layer’s distinct dBm bands from hottest to weakest and keep every
hex at or above the cutoff band $t$ such that the cumulative count first
reaches $\approx 35\%$ of the layer, without exceeding 60% when an earlier
band already reached 18%:

$$
\mathrm{core}=\{h: s(h)\ge t\},\qquad t=\texttt{\_relative\_core\_threshold}(s).
$$

If the layer has only one signal value (binary coverage), the core is the
**entire** footprint and peaks are taken from shape (depth), not dBm.

**Why a relative band cut, not Otsu:** Otsu assumes a bimodal histogram.
Filings are often flat or unimodal, and providers’ dBm origins differ. The
cut is on this layer’s own histogram so a $-50\,\mathrm{dBm}$ filer and a
$-110\,\mathrm{dBm}$ filer of the same shape seed the same blobs.

### 4b. Connected components

On the H3 graph, two cells are adjacent if they share an edge
(`h3.grid_disk(h, 1)`). Blobs are 6-connected components via DFS/BFS.

$$
\text{blob}(h)=\{h'\text{ reachable from }h\text{ staying in }\mathrm{core}\}.
$$

A huge single-tower footprint stays one blob. Density clustering (DBSCAN)
would split or merge with $\varepsilon$, which is the wrong inductive bias
for “one mast, large lobe.”

This is textbook DFS/BFS on the hex dual graph, not a named special-case
algorithm we imported. The digital-image analog (connected components of a
binary picture) is Rosenfeld (1970):

- Rosenfeld, A. (1970). Connectivity in digital pictures. *Journal of the ACM*, 17(1), 146–160. DOI [10.1145/321556.321570](https://doi.org/10.1145/321556.321570)

Drop blobs with fewer than $N_{\min}=35$ cells at res 9
($\approx 3.7\,\mathrm{km}^2$). That is a physical floor, not a statistical
test: smaller patches are noise or small-cell clutter the detector is not
trying to census.

### 4c. Discrete distance transform (binary / flat signal)

When `minsignal` is uninformative, score each core cell by grid distance to
the blob exterior (multi-source BFS). Depth 1 = edge; interior increases;
cap at 40 rings ($\approx 14\,\mathrm{km}$ at res 9).

$$
\delta(h)=\min\{\,d_{\mathrm{grid}}(h,e): e\notin\mathrm{blob}\,\}.
$$

This is the classical discrete distance transform on a grid:

- Rosenfeld, A., & Pfaltz, J. L. (1966). Sequential operations in digital picture processing. *Journal of the ACM*, 13(4), 471–494. DOI [10.1145/321356.321357](https://doi.org/10.1145/321356.321357)
- Borgefors, G. (1986). Distance transformations in digital images. *Computer Vision, Graphics, and Image Processing*, 34(3), 344–371. DOI [10.1016/S0734-189X(86)80047-0](https://doi.org/10.1016/S0734-189X(86)80047-0)

A 1-lobe disk has one depth maximum (the hub). A 3-sector cloverleaf has
three petal maxima.

### 4d. Local maxima + non-maximum suppression

A cell is a candidate peak if its score is $\ge$ every in-blob neighbor.
Plateaus collapse to the cell nearest the plateau centroid. Remaining peaks
are greedily kept in score order iff each is at least $d_{\mathrm{NMS}}$
from already accepted peaks:

$$
d_{\mathrm{NMS}}=\begin{cases}
500\,\mathrm{m} & \text{signal-field peaks},\\
2000\,\mathrm{m} & \text{depth-field peaks on blobs with }\ge 4000\text{ hexes}.
\end{cases}
$$

Greedy NMS is the standard object-detection suppression step:

- Neubeck, A., & Van Gool, L. (2006). Efficient Non-Maximum Suppression. *18th International Conference on Pattern Recognition (ICPR’06)*, 850–855. DOI [10.1109/ICPR.2006.479](https://doi.org/10.1109/ICPR.2006.479)

(The pipeline uses the usual greedy “keep if farther than $d_{\mathrm{NMS}}$”
rule, not Neubeck’s specific two-dimensional scan.)

500 m $\approx 1.4$ H3-9 cells so $\sim 0.8\,\mathrm{km}$ urban macros can
split. 2 km depth NMS stops county-scale fill from tiling fake sites every
500 m (observed on Sedgwick-scale blobs).

### 4e. Prominence (drop overlap shoulders)

Between peaks $a$ and $b$, sample the straight line in Albers and take the
minimum score on that path (the saddle). Drop $a$ if some stronger peak $b$
satisfies

$$
\mathrm{score}(a)-\mathrm{saddle}(a,b)<\pi_{\min},
$$

with $\pi_{\min}=3$ (rings or dB). True sector petals have a deep hub
saddle; overlap shoulders do not. This is the same idea as topographic
prominence, discretized.

### 4f. Cloverleaf / bowtie merge

If two or three peaks lie $1.5$–$10\,\mathrm{km}$ apart and a ring around
their hub has $n\in\{2,3\}$ occupied angular lobes with matching gaps
(18-bin histogram of $\mathrm{atan2}$), merge them to one site at the hub.

**Why:** 3GPP/macro sites are sectorized; treating petals as three masts
makes ordinary same-site growth look like new construction. This rule is
project-specific, calibrated on Albers-warped petals (e.g. Logan, UT), not
taken from a paper.

### 4g. Pin location

The pin is the **peak cell**, not the mass centroid. A centroid of a
cloverleaf sits in the hole; a centroid of an inflated blob walks out into
the new fill.

Joint inference runs on June $\cup$ December geometry so identity is shared.
A site is `new_site` if it has $\le 2$ prior hexes; `expanded_site` if hex
count grew by $\ge 20\%$; else `stable_site`.

---

## 5. Attribution (`attribute.py`)

Each hex with status $\in\{\texttt{new},\texttt{upgraded}\}$ is assigned to
the nearest current pin if it lies inside that site’s reach. Lost hexes and
`stable_site` hexes do not enter the growth shares.

**Reach.** Prefer the 95th percentile of distances from the pin to covered
hexes (`lobe_reach_m`), floored at $3\,\mathrm{km}$. If that is missing,
use $1.6\times$ the core `reach_m` (still floored at $3\,\mathrm{km}$).

Nearest-neighbor query is a k-d tree in Albers:

- Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), 509–517. DOI [10.1145/361002.361007](https://doi.org/10.1145/361002.361007)

Let $A_{\mathrm{new}}, A_{\mathrm{exp}}, A_{\mathrm{un}}$ be
$(\text{hex count})\times A_{\mathrm{hex}}$ tagged `new_site`,
`expanded_site`, `unattributed`. With
$A_+=A_{\mathrm{new}}+A_{\mathrm{exp}}+A_{\mathrm{un}}+\varepsilon$:

$$
\begin{aligned}
\mathrm{same\_site\_growth\_share} &= A_{\mathrm{exp}}/A_+,\\
\mathrm{new\_site\_share} &= A_{\mathrm{new}}/A_+,\\
\mathrm{unattributed\_share} &= A_{\mathrm{un}}/A_+.
\end{aligned}
$$

**Why nearest site within reach, not a Gaussian mixture:** reviewers need a
hard assignment (“this hex is claimed from that mast”). Soft clustering does
not produce an auditable share.

### Fallback vintage matching (when joint inference is unavailable)

Radius-gated linear assignment, cost $c_{ij}=d_{ij}$ if
$d_{ij}\le R$ else $10R$, with $R=2000\,\mathrm{m}$. SciPy then solves
the assignment problem

$$
\min\sum_{i,j} c_{ij} X_{ij}
$$

with $X$ a (possibly rectangular) matching.

- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2), 83–97. DOI [10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109) — the problem.
- Crouse, D. F. (2016). On implementing 2D rectangular assignment algorithms. *IEEE Transactions on Aerospace and Electronic Systems*, 52(4), 1679–1696. DOI [10.1109/TAES.2016.140952](https://doi.org/10.1109/TAES.2016.140952) — the modified Jonker–Volgenant solver SciPy documents for `linear_sum_assignment`.

One-to-one matching prevents two current pins from claiming the same prior
mast.

---

## 6. ASR snap (`attribute.anchor_sites_to_asr`)

Public ASR bulk file: <https://data.fcc.gov/download/pub/uls/complete/r_tower.zip>

Snap an inferred pin onto a registered structure if it is uniquely within
$750\,\mathrm{m}$: no second ASR inside $750\,\mathrm{m}$ unless that
second is at least $1.5\times$ farther. Matches out to $2\,\mathrm{km}$
are recorded but not moved. Several petals that snap to the **same** ASR
collapse to one pin.

**Why 750 m:** rural peak-vs-mast offset was $\sim 650\,\mathrm{m}$ median
in Pearl River calibration. **Why not flag on missing ASR:** ASR is tall
registered structures, not every Verizon rooftop or small cell. Weight of
`asr_no_new_structure` is **0**.

---

## 7. Boundary snap

Share of *new* hexes whose Albers distance to the county boundary is
$\le 1500\,\mathrm{m}$:

$$
\mathrm{boundary\_snap\_share}(g)=\frac{|\{h\in\mathrm{new}(g): d(h,\partial g)\le 1500\,\mathrm{m}\}|}{|\mathrm{new}(g)|}.
$$

**Why:** a tell for coverage drawn to an administrative outline rather than
radiating from masts. 1.5 km is a few H3-9 rings; it is a heuristic, not a
legal eligibility buffer.

---

## 8. Scoring (`score.py`)

### 8a. Features

Relative jump, squashed and **absolute-gated** so $0\%\to 1\%$ of a county
cannot look like a 100% increase:

$$
\begin{aligned}
\tilde{\rho}&=\frac{\rho}{1+\rho}\quad(\infty\mapsto 1),\\
\mathrm{coverage\_increase\_magnitude}&=\tilde{\rho}\cdot\min\!\left(\frac{f_{\mathrm{added}}}{0.05},\,1\right).
\end{aligned}
$$

$\tilde{\rho}=\rho/(1+\rho)$ is a rectangular-hyperbola squash (same shape
as a Michaelis–Menten curve): strictly increasing, bounded, no hard cap to
tune.

Blanket fill-in (low baseline $\to$ near-complete county):

$$
\mathrm{blanket\_fillin}=\bigl(f_C-f_P\bigr)_+\cdot(1-f_P),\qquad f=\frac{\mathrm{covered\,km}^2}{\mathrm{county\,km}^2}.
$$

A county that was already 90% covered cannot score a “blanket.” A county that
goes from empty to full scores 1.

### 8b. Bounded monotone score

Each feature $x_f$ is mapped with operating range $r_f$ (e.g.
$r_{\mathrm{added\_frac}}=0.15$):

$$
\hat{x}_f=\mathrm{clip}(x_f/r_f,\,0,\,1).
$$

Weights $w_f$ satisfy $|w_f|\le 0.25$. Let
$D=\max(\sum_f |w_f|,\,1)$. Contribution:

$$
c_f=\begin{cases}
w_f\,\hat{x}_f/D & w_f\ge 0,\\
|w_f|\,(1-\hat{x}_f)/D & w_f< 0.
\end{cases}
$$

$$
S=\sum_f c_f\in[0,1].
$$

**Why this instead of Isolation Forest / a learned ranker:** a reviewer must
see *why*. Isolation Forest is not monotone in “more same-site growth.” The
0.25 cap and $D\ge 1$ mean turning a weight off cannot inflate another
feature past the cap, and adding unrelated counties cannot change $S$
(no cohort z-score).

This is a **scorecard**, not a citation to a specific ML paper. Credit-risk
scorecards use the same idea (bounded additive points). We did not import
FICO’s tables.

Default weights (Verizon 7/1 J25→D25 calibration, FN preferred over FP):

| Feature | $w$ | Role |
|---------|------:|------|
| `added_frac_of_county` | $+0.25$ | Absolute in-county area |
| `coverage_increase_magnitude` | $+0.10$ | Relative jump, absolute-gated |
| `blanket_fillin` | $+0.14$ | Rural simultaneous fill |
| `same_site_growth_share` | $+0.22$ | Growth from existing pins |
| `unattributed_share` | $0$ | Off: inference misses look like 100% unattributed |
| `boundary_snap_share` | $+0.08$ | Outline hugging |
| `new_site_share` | $-0.22$ | Exculpatory build |
| `asr_no_new_structure` | $0$ | Snap only; not a census |

### 8c. Binary flag (not a percentile)

Let $I_{\mathrm{insuf}}$ be “site inference could not run.”

$$
\begin{aligned}
\mathrm{eligible} &= \mathrm{added\_km}^2 \ge 10,\\
\mathrm{implausible} &=
  (\mathrm{same\_site}\ge 0.50)\\
  &\quad\land\bigl(f_{\mathrm{added}}\ge 0.075 \lor \mathrm{blanket}\ge 0.20\bigr),\\
\mathrm{new\_build} &= (\mathrm{new\_site\_share}\ge 0.50 \land \mathrm{new\_towers}\ge 1)\\
  &\quad\lor (\mathrm{new\_site\_share}\ge 0.35 \land \mathrm{new\_towers}\ge 1 \land \mathrm{new\_towers\_cross\_border}\ge 1),\\
\mathrm{flag} &= \mathrm{eligible}\land\mathrm{implausible}\land\lnot\mathrm{new\_build}\land\lnot I_{\mathrm{insuf}}.
\end{aligned}
$$

(`unattributed` is in the code’s OR but the threshold is 1.01, so it cannot
fire.)

`flag_percentile = 0.95` is **only** a severity badge (Top 5% / Top 10%). An
all-ordinary state must produce **zero** binary flags.

**How we got the gates:** physics catalog in `tests/gaming_scenarios.py` plus
live Verizon 7/1 hexes. 7.5% of county catches Middlesex-scale urban fill
($\approx 7.8\%$) and misses 3–6% modest growth (Sullivan-scale) by design.
$10\,\mathrm{km}^2$ excludes empty-interior counties.

**Error preference:** false positives cost field teams; false negatives mean
a county is not on the list. The detector is a **priority list**, not a
finding of fraud.

---

## 9. Optional tile reconcile (off by default)

Jaccard / IoU between vector coverage and rendered tiles:

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

If $\mathrm{IoU}<0.80$, a CV fallback exists but is **not** invoked on the
national overnight path (runtime).

IoU here is the Jaccard index on two boolean tile masks
(`inter / union`; empty union $\mapsto 1$).

- Jaccard, P. (1901). Étude comparative de la distribution florale dans une portion des Alpes et du Jura. *Bulletin de la Société Vaudoise des Sciences Naturelles*, 37, 547–579. DOI [10.5169/seals-266450](https://doi.org/10.5169/seals-266450) (scans on e-periodica).

---

## 10. What this is not

- Not a likelihood ratio test, not Neyman–Pearson, not a trained classifier.
- Not a tower census. Interior masts inside a fully painted blob are
  unrecoverable from binary coverage.
- Not Okumura–Hata / 3GPP TR 38.901 path loss.
- Live filings can disagree with older slide labels; the formulas follow
  current hexes.

Code: `src/fcc_audit/{changedetect,towers,attribute,score,normalize}.py`.
Knobs: `config/pipeline.yaml`.

## 11. Link check (sources above)

Resolved against Crossref and/or a live GET (19 Aug 2026):

| Claim | Source URL |
|-------|------------|
| Snyder 1987 PP 1395 | https://doi.org/10.3133/pp1395 · PDF https://pubs.usgs.gov/pp/1395/report.pdf |
| EPSG:5070 | https://epsg.io/5070 |
| Sahr et al. 2003 DGGS | https://doi.org/10.1559/152304003100011090 |
| Uber H3 blog (Brodsky 2018) | https://www.uber.com/blog/h3/ |
| H3 restable (res-9 area $0.105332513\,\mathrm{km}^2$) | https://h3geo.org/docs/core-library/restable/ |
| H3 `polygonToCells` centroid rule | https://h3geo.org/docs/api/regions/ |
| Rosenfeld 1970 connectivity | https://doi.org/10.1145/321556.321570 |
| Rosenfeld & Pfaltz 1966 | https://doi.org/10.1145/321356.321357 |
| Borgefors 1986 | https://doi.org/10.1016/S0734-189X(86)80047-0 |
| Neubeck & Van Gool 2006 | https://doi.org/10.1109/ICPR.2006.479 |
| Bentley 1975 k-d tree | https://doi.org/10.1145/361002.361007 |
| Kuhn 1955 assignment | https://doi.org/10.1002/nav.3800020109 |
| Crouse 2016 (SciPy LAP) | https://doi.org/10.1109/TAES.2016.140952 |
| Jaccard 1901 | https://doi.org/10.5169/seals-266450 |
| Census TIGER/Line | https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html |
| FCC BDC program page | https://www.fcc.gov/BroadbandData |
| ASR bulk `r_tower.zip` | https://data.fcc.gov/download/pub/uls/complete/r_tower.zip |

`fcc.gov` / ACM Digital Library may return 403 to automated clients; the DOIs
and FCC URLs are the publishers’ identifiers. Rappaport’s textbook is an
architecture reference (3-sector macros), not a formula we copy, so it has
no DOI here.
