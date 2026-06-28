# ROADMAP — Restore & prove the content+network novelty (greedy-reward teacher) [2026-06-27]

**Goal (north star):** an ABR that **combines semantic content + network signals** in the decision, and
**demonstrate** the combination helps — benchmarked the **same way as the existing comparison** (closed-loop
ns-3 NTN, % of DP-oracle, head-to-head vs the 6 classic heuristics + the current throughput-only model).

**Why this can work (validated offline):** the **greedy-reward causal teacher** (per-segment argmax of the
full QoE reward `w_vmaf·sem·VMAF − w_stall·rebuf − w_switch·|ΔVMAF|·(1−cut_disc·is_cut)`, using current
throughput estimate + buffer) scored **34.9% of oracle** as a policy (vs throughput-only 22.9%, and best
heuristic SVAA 30.3%), and **differs from throughput-only on 72% of decisions** → it *forces* the model to
use content + full network state. The content benefit is **bandwidth-efficiency on saturating segments**
(trim where VMAF is maxed) → strongest on saturating / mixed-VMAF content, **not** high-motion (motion_score
is decoupled from VMAF-curve shape — see `HANDOFF.md` and the headroom-stratified probe).

**Benchmark integration (the key requirement):** every model variant below is scored by the **identical
existing pipeline** so it drops straight into the same table/figures as the heuristics:
`run bridge sims → decisions_to_throughput(+meta) → dp_oracle(ALGO_SHORT=<arm>) → decisions_to_trainrows →
score_v2_perceptual (writes results/<arm>.txt) → make_paper_table / make_paper_figs`. Add each arm to the
`ALGOS` registry in `benchmark/paper_style.py`.

---

## STAGE 0 — Build greedy-reward teacher + offline gate (cheap, no sims)
- **Build** `benchmark/relabel_greedy.py`: per training row, compute the greedy-reward rung from
  `observation.throughput_estimate_bps` (causal) + `observation.buffer_level_s` + ladder `[size,VMAF]` +
  `teacher.semantic_weight` + `teacher.is_scene_cut`; reward weights from `config/oracle_reward.json`.
  (train_jsonl rows already carry semantic_weight/is_scene_cut/ladder-VMAF.) Output `train_jsonl_greedy/`.
- **Gate (offline):** reproduce the probe — teacher differs from throughput-only ~70% of decisions AND scores
  ~35% as a policy (replay on `cache/v2bench_shards`). **Proceed only if confirmed.**

## PROGRESS LOG
- **Stage 0 DONE (gate passed):** `relabel_greedy.py` built; greedy teacher differs from throughput-only on
  **80% train / 73% val** of decisions (content+network dependent). [`train_jsonl_greedy/`]
- **Stage 1 DONE (gate passed):** trained 3 arms on greedy labels. **Best val-F1: net-only 0.388, +vmaf 0.381,
  +semantic 0.426.** ⇒ the **semantic features** clearly improve the fit (+0.038 over net-only); ladder-VMAF
  alone does not (b≈a). The target carries content info — half-proof of the novelty. [`checkpoints_greedy_{net,vmaf,sem}`]
- **Stage 2 DONE [2026-06-28] — content helps online (nuanced positive):** 108/126 cells scored (18 cells
  crashed bridge at startup → coverage net 37/42, vmaf 33/42, sem 38/42). Online %-of-oracle on the congested
  18/9/4.5 matrix (mean over runs per cond), scored by the identical heuristic path:

  | arm | 18 | 9 | 4.5 | **overall** |
  |---|---|---|---|---|
  | (a) net-only | 2.1 | 0.0 | 0.0 | **0.6** |
  | (b) +vmaf | 44.6 | 10.5 | 5.9 | **22.7** |
  | (c) +semantic | 51.8 | 14.0 | 4.1 | **21.5** |
  | throughput-only V2 (AiDash) | 28.2 | 18.6 | 9.9 | 18.7 |
  | best heuristic SFTM | 26.5 | 28.3 | 15.2 | 23.3 |
  | SVAA | 30.3 | 2.9 | 9.7 | 14.3 |

  **Verdict:** (b)/(c) **≫** (a) — content inputs are *decisive*: without them the model can't reproduce the
  content-dependent greedy target and **collapses to the floor (0.6%, low mean-VMAF)**. Content arms (22.7/21.5)
  **beat the throughput-only model (18.7)** and tie the best heuristic (SFTM 23.3). BUT **(c)+semantic ≈ (b)+vmaf
  overall** (sem −1.2pp); semantic only wins where there is **headroom** (18Mbps 51.8>44.6, 9Mbps 14.0>10.5) and
  loses under extreme congestion (4.5Mbps 4.1<5.9). ⇒ the **ladder-VMAF channel is the workhorse content signal
  online**; semantic adds a *headroom-concentrated* boost. Note the offline≠online flip: offline val-F1 had
  +vmaf≈net and +semantic best; online +vmaf is decisive and +semantic≈+vmaf. Results in
  `benchmark/results/Greedy_{net,vmaf,sem}.txt`.
- **Stage 2 FULL DONE [2026-06-28] — combined model is the headline; it WINS overall.** User decision: drop
  the confounded net/vmaf input-ablation arms (train/serve mismatch — inputs zeroed at train but served real;
  this also turned out to be the real cause of the "dead cells", see below) and carry forward only the **clean
  combined model** (`checkpoints_greedy_sem`, network+VMAF+semantic, all inputs real at both train & serve). Ran
  its full **14×5** matrix (added run_1=100Gb/s, run_2=36Mbps + 4 reruns; 70/70 cells). %-of-oracle:

  | algo | 100Gb/s | 36 | 18 | 9 | 4.5 | **overall** |
  |---|---|---|---|---|---|---|
  | **V2+content (ours)** | 51.1 | **53.0** | **46.5** | 13.7 | 3.8 | **33.6** |
  | V2 (throughput-only) | 53.5 | 43.8 | 28.2 | 18.6 | 9.9 | 30.3 |
  | SVAA | **61.3** | 47.2 | 30.3 | 2.9 | 9.7 | 30.3 |
  | SFTM | 40.7 | 35.8 | 26.5 | **28.3** | **15.2** | 29.3 |

  **Headline:** ours **33.6% = best of all** (+3.3 over throughput-only V2 and best heuristic). Pareto-efficient:
  meanVMAF 41.2 @ 3.50 Mbps with **0.6s stall** vs SVAA's similar VMAF (42.8) but **11.6s stall**; lowest overall
  regret (7.3, tied). Dominates the headroom/mid-congestion regime (36/18 Mbps) but **loses to buffer-based SFTM
  under extreme congestion** (9/4.5 Mbps) — honest, sensible (content-aware allocation pays when bandwidth
  exists; under scarcity it's all about not stalling). Artifacts: `benchmark/results/Greedy_sem.txt`,
  `paper/table1_oracle.{tex,md}`, `paper/fig1_pareto.*`, `paper/fig2_curve.*`, `paper/inline_stats.txt`.
  Registered in `paper_style.ALGOS` as the red-diamond "ours"; throughput-only V2 kept as the "V2 (thrpt)"
  baseline.
- **Infra fix [2026-06-28]:** the 18 "dead cells" were NOT a bridge crash — concurrent `waf --run` jobs raced on
  `build/compile_commands.json` via the `clang_compilation_database` post-build hook (`JSONDecodeError` killed the
  build before ns-3 ran). Fixed in `waf-tools/clang_compilation_database.py` (broaden `except IOError` →
  `except (IOError, ValueError)`); the full-matrix re-run had **0 deaths**. No rebuild needed (hook runs from
  source per job).
- **Stage 4 NEXT (optional):** headroom-stratified figure (where the content gain concentrates). Stage 3
  (teacher-constant tuning) + the model-improvement levers (DAgger/asymmetric-loss/window>1) remain the backlog
  if we want to push past 33.6 and/or fix the extreme-congestion crossover vs SFTM.

## STAGE 1 — Semantic ablation: train 3 arms on the SAME greedy-reward labels
Same target, different **inputs** (isolates the content contribution):
- **(a) net-only** — network features only; **ladder = size only (no VMAF)**, semantic branch zeroed.
- **(b) +vmaf** — network + ladder `[size,VMAF]` (content-quality via manifest); semantic branch zeroed.
- **(c) +semantic** — network + ladder `[size,VMAF]` + semantic features (motion, saliency,
  quality_sensitivity, scene_cut).
- **Impl:** add training-time input-ablation flags to `train.py` (`--no-semantic` to zero the semantic
  branch; `--ladder-size-only` to drop the VMAF channel from the ladder context) — the model already has
  separate branches (cf. `evaluate.py --ablate`). Train `checkpoints_greedy_{net,vmaf,sem}` (window=1, LoRA,
  agro-b/lgpu).
- **Cheap gate (val-F1, no sims):** if **(a) ≪ (c)** in val-F1, the labels *require* content to fit — already
  half the proof (the target carries content the network can't reproduce). If (a)≈(c), content is irrelevant
  to the target → stop and rethink.

## STAGE 2 — Benchmark the arms the SAME way as the heuristics (the comparison)
- **Cheap online screen first** (cost control): run each arm on a **reduced matrix** — the 14 test videos ×
  the 3 congested rates {18,9,4.5 Mbps} (skip the trivial 100Gb/s tie initially) via
  `bench_v2_causal_smoke.sbatch CKPT=<arm> OUT=Quic_artifacts/AiDash_greedy_<arm>/…`. Score with the standard
  chain → `results/Greedy_{net,vmaf,sem}.txt`.
- **Decision gate:** does **(c) ≥ (b) > (a)** on % of oracle, and does the best arm **beat the heuristics**
  (SVAA 30.3 / SFTM 29.3) and the current throughput-only model (30.3)? If yes → content+network proven.
- **Full matrix for the winner only:** run the winning arm on the **full 14×5** matrix and regenerate
  `table1_oracle` + `fig1_pareto` + `fig2_curve` with it added alongside the 6 heuristics + throughput-only
  V2. *(Cost: ~70 bridge sims/arm @ ~4–5 h; that's why we screen on the reduced matrix first and only run the
  full matrix for the winner. squeue is flaky here — track via slurm-log `Simulator Time:` + decision counts.)*

## STAGE 3 — Tune the teacher constants (idea #3), then re-benchmark the best
Tune the **teacher's** reward constants (not the eval oracle's — keep the ceiling fixed):
- lower **w_switch** (the greedy teacher dips at 100Gb/s, 52 vs 64, from over-penalising switches);
- raise **scene_cut_discount → ~1.0** (free switches at scene cuts → content-aware switching at boundaries);
- turn on **w_quality_sensitive** (currently 0 — weights quality by per-segment `quality_sensitivity`);
- sweep **w_stall** (stall-vs-quality).
**Gate each offline** (relabel → teacher %-oracle on shards) before retraining; retrain + benchmark only the
best config (Stage-2 chain).

## STAGE 4 — Content-stratified reporting (honest content figure)
Report the headline %-oracle table AND a **VMAF-headroom-stratified** breakdown (content-aware gains
concentrate on saturating content: +12.2 vs −0.9 in the probe). This is the figure that *shows* where/why
content helps. Cheap (reuse `results/*` + per-video VMAF headroom). Frame as **content-aware bandwidth
efficiency**, not quality-prioritisation.

## STAGE 5 — (optional, novel) Content-lookahead teacher
Future **content** (upcoming segments' VMAF curves) is known at decision time from the manifest, even though
future **throughput** is not. Build a teacher/model that allocates bits over a short horizon using
known-future-content + current-network budget (give the model window>1 on the ladder/content axis). This is a
*causal* way to capture the cross-segment allocation the myopic teacher can't — the planning-payoff angle.
Bigger build; do only if Stages 1–3 land. Benchmark via the Stage-2 chain.

## STAGE 6 — (optional) Stronger baselines for the table
Run the already-coded `BolaClient` (dash.js default) on the benchmark matrix and add it to the table; BBA/MPC
if desired. Strengthens the "vs the literature" comparison. (No learned-model re-implementation — Strategy B.)

---

## Sequencing & gates (do in order; each gate can stop the line)
0. greedy teacher offline gate → 1. ablation train + val-F1 gate → 2. reduced-matrix benchmark gate
   ((c)≥(b)>(a) and beats heuristics) → full-matrix for winner → 3. constant tuning (offline gate → retrain →
   benchmark) → 4. stratified reporting → 5/6 optional.

## Files
| File | Stage | Change |
|---|---|---|
| `benchmark/relabel_greedy.py` | 0 | NEW — greedy-reward causal teacher relabeler (+ offline gate print) |
| `scratch/ai_llm/train.py` | 1 | add `--no-semantic` / `--ladder-size-only` input-ablation flags |
| `benchmark/paper_style.py` | 2 | register new arms (name/color/marker) in `ALGOS` |
| reuse | 2–4 | `bench_v2_causal_smoke.sbatch`, `decisions_to_throughput`, `dp_oracle`, `decisions_to_trainrows`, `score_v2_perceptual`, `make_paper_table`, `make_paper_figs` |
| `benchmark/relabel_greedy.py` (params) | 3 | constant sweep (w_switch / scene_cut_discount / w_quality_sensitive / w_stall) |

## Verification / success criteria
- **Novelty proven** if, scored by the identical benchmark: **(c) +semantic ≥ (b) +vmaf > (a) net-only** on
  % of oracle, AND the best content+network arm **≥ the best heuristic** and **> the throughput-only model** —
  with the VMAF-headroom-stratified figure showing the gain concentrated on saturating content.
- All arms scored by the same `score_run` path; same videos/conditions/seeds; results land in `results/*.txt`
  and regenerate `paper/table1_oracle` + figures with the new arms included.

## Cost & caveats
- Bridge sims are the bottleneck (~4–5 h × 70 cells/arm) → screen on the reduced (congested-only) matrix,
  full matrix for the winner only. No new ns-3 sims for Stages 0/4; Stages 1/3 are GPU training.
- Honest framing: content benefit = **efficiency on saturating content**; offline F1 only proves the target
  needs content — **online %-oracle is the decisive metric** (offline≠online; see `HANDOFF.md` §6).
- Don't curate toward high-motion (motion ≠ VMAF-curve shape); aim for VMAF-curve diversity if adding data.
