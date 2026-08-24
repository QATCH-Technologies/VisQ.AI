# Dataset Coverage & Sparsity Census

Source: `data/raw/formulation_data_072726_zp_descriptors_charges_added.csv`
Module: `src/visqai/analysis/coverage.py` + `coverage_plots.py`
Tests: `tests/test_coverage.py` (12/12 passing -- v1's 6 + v2's 6)

**Status: v2.** v1's census (Tasks 1-5) stands unchanged. v2 retires the class-A
scoring path (C-9), replaces `score = deficit x leverage` with either a Pareto
frontier or a rank-checked scalar depending on leverage mode (C-10, C-15, C-16),
collapses the remedy taxonomy to two values (C-11), gates class-C axes on
deliberate variation (C-12), and adds cross-bin `n_eff` comparability guardrails
(C-13). C-14 (Fv x format x viscosity-regime confound) is promoted from a
reported outcome to a standing constraint.

## v1 Task 0 gate (unchanged, frozen)

13 of 15 reproduced exactly. Two corrections, recorded per the plan's own
"Plan history" mechanism rather than silently substituted:

| # | Quantity | Expected | Observed | Status |
|---|---|---|---|---|
| 1 | Rows after blank-drop | 520 | 520 | confirmed |
| 2 | Non-placebo rows / clusters | 434 / 12 | 434 / 12 | confirmed |
| 3 | Placebo (`none`) rows | 86 | 86 | confirmed |
| 4 | Rows with all 5 shear channels populated | 434 | **520** | **C-7** |
| 5 | `nunique` per protein of the six C-1 descriptors | 1 for all six | 1 for all six | confirmed |
| 6 | Clusters w/ Arginine / Proline / Lysine | 8 / 5 / 4 | 8 / 5 / 4 | confirmed |
| 7 | Clusters w/ Sucrose / Trehalose | 13 / 5 | 13 / 5 | confirmed |
| 8 | Clusters w/ tween-80 / tween-20 | 11 / 5 | 11 / 5 | confirmed |
| 9 | Clusters w/ Histidine / PBS / Acetate | 13 / 10 / 10 | 13 / 10 / 10 | confirmed |
| 10 | `Salt_conc` rows at 140 / 0 / other | 261 / 252 / 7 | 261 / 252 / 7 | confirmed |
| 11 | `Temperature` rows in {25.0, 27.5} | 444 of 520 | 444 of 520 | confirmed |
| 12 | Clusters with any Fv charge value | 6 | 6 | confirmed |
| 13 | log10(v1000) min / max, non-placebo | -0.10 / 1.98 | -0.10 / 1.98 | confirmed |
| 14 | Clusters with log10 v1000 > 1.0 / > 1.5 | 7 / 6 | **9** / 6 | **C-8** |
| 15 | Design effect / n_eff at ICC 0.49 | 36.4 / 11.9 | 36.4 / 11.9 | confirmed |

## v2 Task 0 gate (corrected values + new quantities)

All reproduced exactly against the current CSV; halted-and-reported nowhere
in this pass -- every quantity below either matched or is reported as a
correction/finding through the normal C-numbering mechanism.

| Quantity | Expected | Observed | Status |
|---|---|---|---|
| Rows after blank-drop / non-placebo / clusters | 520 / 434 / 12 | 520 / 434 / 12 | confirmed |
| Rows with all 5 shear channels populated (all rows) | 520 | 520 | confirmed |
| Clusters with log10 v1000 > 1.0 / > 1.5 | 9 / 6 | 9 / 6 | confirmed |
| Fv-bearing clusters | 6 | 6 | confirmed |
| Overlap of Fv-bearing set with (v1000 > 1.5) set | 3 | 3 (adalimumab, nivolumab, pembrolizumab) | confirmed |
| n_eff, all 12 clusters | 11.94 | 11.94 | confirmed |
| n_eff, 11 clusters excluding poly-hIgG | 18.68 | 18.68 | confirmed |
| Protein-level r: kP<->HCI / kP<->C_Class / HCI<->C_Class | 0.88 / 0.91 / 0.86 | 0.884 / 0.909 / 0.861 | confirmed |
| Protein-level r: MW<->PI_mean | 0.83 | 0.8347 | confirmed (see C-9 note below on the 0.85 dedupe threshold) |
| `gap_ranking.csv` v1 rows tied to 6 d.p. on `score` | >= 8 | 17 groups / 39 rows | confirmed (C-9 signature) |

## C-9 through C-16 (this pass)

- **C-9.** `REQUIRED_CLUSTERS_GENERALIZATION`'s native scope is *feature
  presence across the full cluster set* (does a level/feature show up in
  enough of the 12 clusters), not *partition-cell membership*. v1 applied it
  at partition-cell scope to every class-A bin. Because class-A observed
  counts sum to exactly `N_CLUSTERS_TOTAL` by construction
  (`class_a_marginal`'s own assertion), total deficit for a k-bin axis is
  fixed at `10k - 12` regardless of the data, at most one bin per axis can
  ever reach deficit 0, and within-axis ranking degenerates to
  `-observed_clusters`. **Fix:** class-A deficit scoring is retired entirely
  (Task 9, `class_a_void_regions` replaces it; `class_a_deficit` now raises
  on call, Test 12). The threshold is wrapped in `ScopedThreshold` with an
  explicit `unit_of_comparison` (`feature_presence` / `partition_cell` /
  `condition_bin`); `compare_to_threshold` raises on a mismatch (Test 7,
  Requirement S1-b).

- **C-10 (rescoped).** `leverage="range"` was diagnosed as saturating on
  v1's full ranking (0.36-1.00, bulk above 0.65, `score` ~= 0.75 x deficit).
  That population was 34% class-A entries (26 of 77) -- re-measured on class
  B+C alone (post Task-9, the population any leverage mode now actually
  scores), `range`'s IQR is 0.408, not saturated. **The saturation claim
  does not survive as a general statement about `range` -- it was a
  class-A-specific artifact** (class-A leverage IQR alone: 0.086, min 0.418,
  max 1.0 -- see Test 9 below for the retrospective, honest re-check under
  the new rank-based guard). `range` remains `DEFAULT_LEVERAGE_MODE` in v2.

- **C-15 (validity, not tuning).** A `leverage="high_visc_share"` mode (the
  v2 plan's originally specified new default: fraction of a bin's own rows
  above the 1.5 decision threshold) was implemented, measured, and
  **rejected before being wired in as a selectable option** -- for two
  independent reasons:
  1. *Circularity.* A bin that has never been pushed into the
     high-viscosity regime scores exactly 0 -- not "low value," but "no
     observation." Weighting gaps by response observed *within the scored
     bin* penalizes sparse bins twice (once as the deficit, again as ~0
     leverage) for the same underlying reason.
  2. *C-14 confound.* The clusters that ever cross the high-viscosity cut
     are disproportionately the Fv-undefined formats. `high_visc_share`
     leverage would systematically deprioritize acquisitions in exactly the
     region this census exists to surface -- worse than uninformative.

  Measured on the real post-Task-9 B+C population (51 entries):
  `high_visc_share` IQR = 0.061 (28 of 51 entries score exactly 0),
  `corr(score, leverage) = 0.92` -- C-10's failure mirrored in the opposite
  direction (leverage, not deficit, would dominate). Not implemented as a
  selectable `leverage=` value; `_leverage_fn` only accepts `"range"` and
  `"logo_residual"`.

- **C-16.** `score = deficit x leverage` as a single scalar is retired for
  endogenous leverage (`"range"`): a two-factor product is governed by
  whichever factor has the larger relative RANK spread, and no leverage
  definition escapes that except by coincidence. Default presentation is
  now a **Pareto frontier** over `(deficit, leverage)` per remedy class
  (never combined across remedy classes -- Task 11). `leverage="logo_residual"`
  is exogenous (per-bin model error, not a function of the response
  observed in the bin) and remains the one mode that still produces a
  scalar `score` -- rank-checked via `check_score_non_degeneracy` before
  being trusted (Kendall's tau between score and each factor; raises if
  either exceeds `KENDALL_TAU_MAX=0.9`).

- **C-11.** `acquire_condition` was never a distinct remedy from
  `acquire_rows_within_cluster` in practice -- both were assigned purely
  from axis class, and "a new formulation of an existing protein" is the
  same remedy whether the empty cell is a condition bin or a shear channel.
  Taxonomy reduced to `{acquire_cluster, acquire_rows_within_cluster}`,
  assigned by what would close the gap (Task 11).

- **C-12.** `Temperature` admitted to class C in v1 without checking it was
  ever deliberately varied. 85.4% of rows sit on one of two setpoints (below
  the 95% row-scope degenerate-continuous flag, so it didn't trip that
  check), but only 3 of 12 clusters ever record >= 3 distinct raw
  Temperature values (vs. 12/12, 12/12, 12/12 for Protein_conc, Buffer_pH,
  Whole_Antibody_Charge_at_Buffer_pH). Gate added (`class_c_design_variation_gate`,
  Task 12.1): Temperature fails via the second clause, is reported in
  `not_varied_by_design.csv`, and never ranked.

- **C-13.** `n_eff` is not monotone in row count (Kish's design effect
  penalizes imbalance via `Sum(n_i^2)/Sum(n_i)`, not row count directly).
  `BinCount` gained a `balance` field (`(Sum n_i)^2 / (k * Sum n_i^2)`,
  scope `between_protein`); every function that returns `n_eff` now returns
  `balance` alongside it, and `compare_n_eff` requires both. Regression
  test 8 encodes the non-monotonicity directly (`n_eff(12 clusters)=11.94 <
  n_eff(11, excl. poly-hIgG)=18.68`, despite 434 -> 281 rows) so a future
  "fix" to n_eff fails the suite rather than landing silently.

- **C-14 (promoted to standing constraint).** Unchanged from v1's E7: Fv
  availability (6 clusters) and the log10 v1000 > 1.5 set (6 clusters)
  overlap in only 3 (adalimumab, nivolumab, pembrolizumab). The
  high-viscosity-only clusters (etanercept, bsa, poly-hIgG) are exactly the
  Fv-undefined formats (Fc-fusion, other, polyclonal). This is now cited
  directly as the reason C-15's `high_visc_share` leverage was rejected on
  validity grounds, not just measured and found lacking.

## v2 deliverables

| File | Contents |
|---|---|
| `src/visqai/analysis/coverage.py` | v1 Tasks 1-5 unchanged. v2: `ScopedThreshold`/`compare_to_threshold` (12.2), `BinCount.balance`/`compare_n_eff` (12.3), `class_a_descriptor_space`/`class_a_collinearity_groups`/`class_a_void_regions`/`class_a_deficit` (9), `_pareto_frontier_mask`/`check_score_non_degeneracy` (10/16), `class_c_design_variation_gate` (12.1), `build_gap_report` (replaces `build_gap_ranking`; 6/9/11/12/16) |
| `src/visqai/analysis/coverage_plots.py` | P1/P3/P4 unchanged. P2 kept as companion. New: P2b, P5' (replaces P5), P6b (replaces P6), P9 |
| `tests/test_coverage.py` | 18/18 passing (v1's 6 + v2's 12, Task 15) |
| `reports/coverage/gap_ranking.csv` | **Frozen v1 artifact** (77 rows) -- kept as evidence for the C-9 tie-signature check (Task 0), superseded by the four files below |
| `reports/coverage/void_regions.csv` | 5 class-A void regions (Task 9.3) |
| `reports/coverage/acquire_cluster.csv` | 7 class-B + Fv-pseudo-level gaps (Task 11) |
| `reports/coverage/acquire_rows_within_cluster.csv` | 32 class-C condition/shear gaps, Temperature-gated out (Task 11/12.1) |
| `reports/coverage/not_varied_by_design.csv` | 1 row (Temperature), reported, never ranked |
| `reports/coverage/p1_logo_support_matrix.png` ... `p9_deficit_leverage_frontier.png` | Rendered + visually checked (P5' title/panel overlap and the P9.3 void-region-overlap defect both caught and fixed on first render -- see below) |

## Pre-registered outcomes (v2)

- **E8.** Confirmed with a correction to how "collapse" was achieved. The
  five v1 entries that all encoded "BSA is alone" collapse into **exactly
  one** void region (rank 1, nearest_protein=bsa, void_score=2.69) -- but
  only after fixing a real defect in the first implementation: `void_score`
  grows with distance from the nearest protein, so a naive top-K over raw
  grid points with a small non-max-suppression radius picked **four of five**
  regions all pointing away from BSA along the same radial direction (up to
  88% descriptor-range overlap on MW), failing the acceptance test outright
  on first render. Fixed by capping to one region per nearest-protein
  Voronoi cell (guarantees <= 1 region per protein by construction) and
  decoupling the back-projection box width from the NMS separation radius
  (a box tied to the separation constant was ~25% of the whole descriptor
  range wide). After the fix: max pairwise per-axis overlap across the 5
  final regions is 14.6% (well under the 50% bar). The five regions are
  centered on bsa, etanercept, poly-hIgG, nivolumab, and adalimumab -- the
  highest-ranked is the low-pI/low-kP/low-MW direction held by BSA alone,
  as predicted.
- **E9 (partially refuted, not forced).** Effective axis count (participation
  ratio) over the six class-A descriptors is **2.80**, and 3 of 6 components
  reach 90% cumulative variance -- confirming "2-3, not 6." The collinearity
  dedupe confirms {kP, HCI, C_Class} collapse to one representative (pairwise
  |r| 0.884/0.909/0.861, all >= 0.85). **{MW, PI_mean} do NOT collapse**:
  their pairwise |r| is 0.8347, under the 0.85 threshold, and the threshold
  was not lowered to force the merge. This is not a contradiction with the
  plan's own worked example for a void region ("kP <= 2.5, MW <= 100 kDa"
  reported alongside "pI 5" as a *separate* constraint) -- that example is
  consistent with MW and PI_mean staying distinct, not merged. Both remain
  singleton representatives; the void-region descriptor ranges (`void_regions.csv`)
  report MW, PI_mean, PI_range, and kP as four separate columns, not three.
- **E10 (confirmed, expected null).** Retiring class-A deficit changes no
  class-B result. Fv stays at 6 of 10 required clusters (`fv_defined`/`fv_undefined`
  both n_clusters=6). tween-20=4, trehalose=4, lysine=4, proline=5 --
  identical to v1's cluster counts.
- **E11 (expected-null, honestly reported, not tuned).** The rank-based
  guard was re-run against the historical class-A `range`-leverage scalar
  (deficit=10-n_clusters, leverage=`_leverage_range`, score=deficit x leverage,
  n=26) and **does not trip** at `KENDALL_TAU_MAX=0.9`
  (tau_score_deficit=0.468, tau_score_leverage=0.262). This does not overturn
  C-10's original diagnosis (leverage's IQR on that same population is
  genuinely tiny, 0.086) -- it means the rank-based reformulation catches a
  *different* failure mode (one factor's rank order fully determines score's
  rank order) than the IQR check did (one factor has no spread to
  discriminate with). Both `deficit` (5 distinct values, 11 of 26 bins tied
  at deficit=7) and `leverage` are heavily tied on this 26-point set, and
  Kendall's tau is tie-corrected -- it gets damped by low-cardinality
  clustering on *either* side, independent of how lopsided the two factors'
  relative spread is. `KENDALL_TAU_MAX` was not lowered to force a trip;
  Test 9 asserts the true (non-raising) result and separately re-confirms
  the original IQR signal (`leverage` IQR < 0.15) so C-10's actual finding
  stays checked by the suite even though the new guard doesn't reproduce it
  under this specific statistic.
- **E12 (confirmed).** `Temperature` fails the 12.1 design-variation gate
  (3 of 12 clusters reach >= 3 distinct raw values, vs. `DESIGN_VARIATION_MIN_CLUSTERS=6`)
  and is reported in `not_varied_by_design.csv`, never ranked. `Protein_conc`
  is prominently represented at the top of `acquire_rows_within_cluster.csv`
  (2nd-ranked overall, tied on the Pareto frontier: `Protein_conc=etanercept`,
  deficit=2, leverage=0.933) -- not literally rank 1 (`Buffer_pH=nivolumab`,
  deficit=3, leverage=0.914, holds that spot), but Protein_conc appears
  repeatedly in the top 8 rows where it was absent from v1's ranking
  entirely (Temperature had displaced it).
- **E13 (confirmed, expected null).** The set of six clusters with no rows
  below 25 mg/mL (adalimumab, bevacizumab, bgg, pembrolizumab, trastuzumab,
  vudalimab) is unchanged from v1 and appears in `acquire_rows_within_cluster.csv`
  on the `Protein_conc` axis. The instrument changed ranking, not the
  underlying coverage facts.
- **E14 (confirmed).** Test 8 passes on the current data: 11.94 < 18.68.
- **E15 (genuinely open -- reported below, no directional prediction).**
  Temperature tail provenance.

## Task 13: Temperature tail provenance (investigation, not a fix)

41 distinct `Temperature` values total; two dominant setpoints (25.0, n=235;
27.5, n=209) cover 85.4% of rows. The remaining ~14.6% splits into two
distinct groups with different, checkable explanations:

1. **Placebo stress-test setpoints** (15.0/17.5/20.0/30.0/35.0/36.5,
   36 rows) sit **entirely on `Protein_type == "none"` rows** (IDs
   181-216ish, contiguous) -- these are deliberate round-number stress
   conditions for the placebo stratum, not noise.
2. **A block of ~30 high-precision decimal values** (e.g. 25.216364,
   24.801818, 25.246364) confined almost entirely to two clusters:
   Nivolumab (16 rows, contiguous IDs 488-503) and poly-hIgG (14 rows,
   contiguous IDs 505-518), plus 2 Adalimumab rows (630-631).
   - **Lattice check:** Nivolumab's decimals fit a plain 2-decimal-place
     (hundredths) pattern -- consistent with routine finer-grained instrument
     logging, nothing anomalous. poly-hIgG's decimals do **not** fit any
     small-denominator fraction (tested every integer denominator 1-200,
     and specifically 7/9/11/13/22/33/44/99 on a hunch from the repeating-digit
     look of a few values -- all left residuals near 0.5, i.e. no fit).
   - **Correlation check:** within the 30-row weird-decimal subset, Temperature
     does not correlate significantly with `Protein_conc` (r=-0.175, p=0.355)
     or with `Viscosity_1000` / log10(Viscosity_1000) (r=-0.166, p=0.380 /
     r=-0.257, p=0.171). No evidence these values were back-calculated from
     the response.
   - **Source/ID check:** the weird-decimal rows sit in tight, contiguous ID
     blocks per protein (488-503, 505-518), strongly suggesting a specific
     data-collection batch or source rather than scattered instrument noise
     spread across the whole dataset.

**No directional conclusion drawn.** The ID-contiguity is a real, checkable
provenance signal (worth flagging to whoever owns data ingestion for
poly-hIgG/Nivolumab specifically), but the absence of any response
correlation argues against these being downstream of viscosity -- i.e.
against a leakage concern tied to the model's own target. Reported as
findings only, per the plan's own instruction; nothing proposed.

## Plan history

- **C-7, C-8.** Unchanged from v1 -- see the frozen table above.
- **C-9.** `REQUIRED_CLUSTERS_GENERALIZATION` applied at partition-cell scope
  when its native scope is feature presence. Class-A deficit retired
  (Task 9); threshold scope-typed (12.2, `ScopedThreshold`/`compare_to_threshold`).
- **C-10 (rescoped).** `leverage="range"` saturation was a class-A-specific
  artifact, not a general property of `range` -- does not hold on the
  post-Task-9 B+C population (IQR 0.408). *Origin: identified by the plan
  author during this session's leverage-design discussion, not by the
  agent's initial (overbroad) framing.*
- **C-11.** `acquire_condition` was not a distinct remedy. Taxonomy reduced
  to two (Task 11).
- **C-12.** `Temperature` admitted to class C without a design-variation
  gate. Gate added (12.1).
- **C-13.** `n_eff` treated as cross-bin comparable. `balance` field added
  (12.3); regression test 8 encodes the non-monotonicity.
- **C-14.** Fv availability x molecular format x viscosity regime confound,
  surfaced by v1's E7 membership check. Promoted from a reported outcome to
  a standing constraint; now cited as the substantive reason (not just a
  measurement) for rejecting C-15's leverage candidate.
- **C-15 (new).** `leverage="high_visc_share"` -- the v2 plan's originally
  specified new default -- rejected on validity grounds (circularity + the
  C-14 confound) before being wired in as a selectable option, not merely
  found to fail a dispersion check. *Origin: plan author's critique during
  this session, sharpening the agent's initial dispersion-only framing into
  a validity argument.*
- **C-16 (new).** `score = deficit x leverage` as a single scalar retired
  for endogenous leverage. Pareto frontier per remedy class; `logo_residual`
  the only scalar, rank-checked when available. *Origin: plan author,
  replacing the agent's initial IQR-based Task 10.1 guard, which the plan
  author identified as data-dependent-threshold-tuning in spirit even when
  the specific IQR floor (0.15) was chosen before looking at outcomes.*
- **E9 addendum.** {MW, PI_mean} pairwise |r|=0.8347 sits just under the 0.85
  collinearity threshold; not adjusted to force the merge the plan's prose
  anticipated. Reported as a partial refutation, consistent with the
  Fv/charge session's precedent (report discrepancies, don't force-match).
- **E11 addendum.** The Task 10.1 guard, restated as rank-based (Kendall's
  tau) per the plan author's correction, does not retrospectively trip
  against the exact class-A population that motivated C-10. `KENDALL_TAU_MAX`
  was not lowered to force it. Reported as an honest limitation of the
  rank-based statistic for tie-heavy, low-cardinality factors -- not a
  reversal of C-10's underlying finding, which Test 9 re-confirms via the
  original IQR signal directly.
