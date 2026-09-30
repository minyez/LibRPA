# Strict 2D GW Feature Manifest

This document is the integration checklist for the single active strict-2D
development branch. Historical source snapshots are immutable behavioral
references. They are not production branches and must not be selected directly
for new calculations.

## Active Lineage

| Item | Value |
|---|---|
| Upstream base | `master_ghj@9ce52212` |
| Active branch | `codex/strict2d-gw-main` |
| Analytic head/wing introduction | `c3e2152d` |
| Internal PW-to-auxiliary normalization | `f87f2f2a` |
| Direct complete-`Wc` Gamma route | `6c7e8e97` |
| Final fixed-Gamma-basis integration | `6c498441` |

Only `6c498441` (or a later commit that contains it and passes the same gates)
is an admissible production source.  The earlier commits in this table are
lineage checkpoints: `6c7e8e97` routed the production call but did not yet
project the numerical Ewald external legs into the fixed Gamma Coulomb basis.

The internal normalization commit reads the in-plane cell area and the
constant auxiliary-basis moment from producer metadata. The resulting scale is
computed, not supplied as an adjustable physical parameter.

## Production Feature Gates

| Feature | Production call site | Focused test | Runtime evidence | Status and action |
|---|---|---|---|---|
| Reader-v1, symmetry, and shrink support | inherited from `master_ghj`; `driver/read_data.cpp` and GW API | existing reader/symmetry tests | reader versions, symmetry options, and artifact validation | Present in base; every producer remains an independent numerical gate. |
| Analytic 2D head/wing response | `diele_func::cal_headwing` and strict-2D helpers | prefactor, Schur, radial-integral, dense-inverse tests | analytic head/wing marker | Present. |
| Direct Gamma-cell average of complete `Wc` | `compute_Wc_freq_q_blacs` -> `rewrite_strict_2d_wc` | route-selection, all-block dense inverse, radial quadrature tests | direct-complete-`Wc` marker | Present after `6c7e8e97`; the active integration additionally projects the numerical Ewald legs into the fixed Gamma Coulomb basis before the average. |
| Numerical Ewald external legs in the fixed Gamma Coulomb basis | `compute_Wc_freq_q_blacs`: `U^dagger sqrt(V_Ewald) U` | dense projection regression | `Projected full-Ewald regular Wc legs...` marker | Present in the active integration. This step was explicit in the historical hybrid implementation but missing from `6c7e8e97`; binaries made before this fix are not admissible production references. |
| Skip ordinary factorized `q=0` Coulomb legs | `compute_Wc_freq_q_blacs` | route-selection test | `Skipping factorized q=0 Coulomb multiplication...` | Present after `6c7e8e97`. |
| All finite-`q` screening and `Wc` use full Ewald Coulomb | GW API selects `vq` for `use_fullcoul_eps/wc` | `test_strict_2d_gw_uses_full_coulomb_at_all_q` | input echo plus all-q full-Ewald marker | Enforced by the production GW API; strict 2D rejects either cut-Coulomb choice. |
| Strict route fails closed when analytic data are absent | GW API and `compute_Wc_freq_q_blacs` | strict-request/runtime regression | fatal message rather than fallback | Present; strict 2D rejects missing analytic data and the non-ScaLAPACK `Wc` path. |
| Internal PW-to-auxiliary Coulomb normalization | `read_data.cpp` -> `configure_strict_2d_coulomb_head`; `rewrite_strict_2d_wc` | metadata and transformation tests | metadata area/moment and computed scale | Present after `f87f2f2a`; no manual head coefficient is allowed. |
| Finite-`q` L1 behavior and bounded Gamma average | `dielecmodel.cpp` strict-2D functions | finite-`q` reference, bounded-average tests | analytic finite-`q` marker | Present. |
| Cartesian Voronoi-cell reference for Gamma average | test-only numerical integration | analytic radial average versus clipped Cartesian subcells | test result | Restored in the active integration. |
| Fixed-Gamma-basis finite-`q` diagnostics | `output_2d_finite_q_diagnostics` | 37/19-column schema, Gamma-first ordering, and projection tests | `strict2d_finite_q_scaling.csv` and `strict2d_gamma_wc_blocks.csv` | Connected as opt-in diagnostics; disabled state leaves the production q ordering unchanged. |
| `q`-shell classification | post-processing of diagnostic CSV | post-processing tests | shell-resolved table/plot | Diagnostic only; keep out of default physics path. |
| Alpha-response and omega-zero overrides | archived diagnostic snapshots | archived tests only | explicitly named diagnostic files | Diagnostic experiments only; do not merge environment overrides into production. |
| ABACUS high-`L` Ewald central term | ABACUS producer executable, outside LibRPA | producer regression | executable hash and `librpa_2d_coulomb_head.dat` | External producer gate; audit exact executable for every material. |

## Historical Snapshot Fingerprints

All snapshots below descend from the historical `05928161` line. The hashes
identify the inspected key files, not a claim that the directories are clean Git
commits.

| Snapshot | `epsilon.cpp` SHA256 | `dielecmodel.cpp` SHA256 | Role |
|---|---|---|---|
| `strict2d-directwc-20260722` | `7ea67092f9b3312c3717ce84bab47dd23501b9b834a38bd8fdaa481b64bc9b69` | `4f715b68a87c23c3e4bd31ca1baea3ba7a6ad3fb3e7610b6d23462751170c186` | First direct complete-`Wc` Gamma route. |
| `strict2d-hybridwc-20260724` | `6862a7ff2387bfd448b26e8841e5667228f77d7869909b0c2cb55e70e19c86d0` | `cb1339e8094896545c58442cc5264f378bdb906aa79063e9eeb8f68bb8dae52e` | Full-Ewald finite `q` with separate Gamma regular body. |
| `strict2d-allqewald-diag-loginfix2-20260726` | `15f4d298ba007c2ce3dab8e10331fe73290542cb6678b408ad6dcfc1a1de1ef7` | `05947f7b01da13e160cff4a9e37f2a46b5f9343fd284da6a91bdb5de69cfebd7` | Fixed-Gamma-basis diagnostics and lifecycle regression. |
| `strict2d-rawnorm-propagationfix-20260727` | `15f4d298ba007c2ce3dab8e10331fe73290542cb6678b408ad6dcfc1a1de1ef7` | `c282619cb10baaaa265a9d40b6349937a9801eae1cdf1ae539d2dd6fd19352d8` | Historical manual normalization; superseded by internal metadata normalization. |
| `strict2d-rawnorm-qshell-diag-20260728` | `4181ded77c847860eacca1f13f95fc7ba3be9fbfcd445ec29d326659a6feba43` | `c282619cb10baaaa265a9d40b6349937a9801eae1cdf1ae539d2dd6fd19352d8` | Diagnostic shell labels only. |
| `strict2d-omega0-override-20260801` | `2190ac45ef36d6061f5e7edd11eb619452bcb1228acc138c1a8e82138f65b6de` | `d803ffd6485c0eee375576eea2127318f1c5a028aeeffcfffc11138d7518ee8f` | Diagnostic response overrides and dumps; not production behavior. |

## Branch Reconciliation

The following branch-tip audit prevents a historical worktree from becoming
an implicit second production line.  "Present" means that the behavior is in
the active lineage or in its `master_ghj` base; it does not require retaining
the old commit hash when the change was ported or superseded.

| Historical branch | Production behavior checked | Active-line evidence | Disposition |
|---|---|---|---|
| `codex/headwing-activek-master-ghj` | active k list for PyATB and reader-v1 data | `12ea0beb` plus current reader-v1 tests | Present; the old branch is not selected. |
| `codex/rpa-headwing-sym-ibz-regression` | full-BZ Gamma-cell measure, symmetry atom/k-star parsing, k-parallel head/wing restoration | `a033ec4c`, `133a6061`, `84a62bfc`, and current symmetry/headwing tests | Present through equivalent or later ports. |
| `codex/headwing-sym` | q-average/head-only controls, symmetry-expanded head and wings, shrink and k-parallel support | `17bc7828`, `c34961c8`, `4c302ffa`, `12ea0beb`, and inherited reader/shrink support | Present; old optimization and regression commits are references. |
| `codex/rpa-headwing-qavg` | RPA q-average trace-log correction and head/wing response replacement | current `rpa_headwing_mode`, q-average code, and focused tests | Present; the old epsilon-average GW route is superseded by direct complete-`Wc`. |
| `codex/2d-gamma-headwing-stageA` | early analytic 2D formulas and tests | `c3e2152d` followed by the stricter `6c498441` route | Superseded; do not build from this old base. |
| `codex/2d-gw-headwing-cut-masterghj-20260720` | experimental cut-Coulomb GW path | no commits unique relative to the active lineage | Not a production feature; strict 2D rejects cut Coulomb. |
| dated omega-zero, alpha, q-shell, and finite-q dump snapshots | observability and limiting controls | opt-in 37/19-column diagnostics and post-processing | Diagnostics only; no environment override changes production physics. |

Every production behavior in this table is also represented in the gate table
above.  A future branch is merged only after adding its behavior, focused test,
runtime evidence, and disposition to this manifest.

## Required Calculation Identity

A strict-2D production result is admissible only when all of the following are
recorded together:

1. LibRPA source commit and executable SHA256 from a clean build.
2. ABACUS source/executable SHA256, including the high-`L` 2D Ewald correction.
3. Reader-v1, symmetry, shrink, and same-producer full-grid PyATB validation.
4. `replace_w_head=t`, `option_dielect_func=3`, `use_2d_dielectric=t`, and full
   Ewald Coulomb for both epsilon and finite-`q` `Wc`; no 2D `vq_cut` route.
5. Direct complete-`Wc` Gamma marker and skipped-factorized-`q=0` marker.
6. Solver success, finite band/self-energy output, and clean fatal/NaN scan.
7. Exact frequency grid, analytic-continuation order, basis/ABFS chain, vacuum,
   and k mesh.

Passing a build, producer, PyATB, or output-file check alone does not establish
the numerical or physical validity of a GW result.
