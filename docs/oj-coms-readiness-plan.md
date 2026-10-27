# OJ-COMS Submission Readiness Plan

Target: **IEEE Open Journal of the Communications Society** (open-access, rigorous peer review).
Paper: *Content-Aware ABR Streaming with a Small Language Model over Satellite 5G IAB Networks*
(`latex-article/slm_abr.tex`).

## Guardrails (non-destructive — applies to every phase)
- **Never overwrite; always archive.** Corrupt-VMAF results, current checkpoints, and the current
  paper stay as `*_prevmaf` / `*_v<n>` / archived dirs so we can diff old vs new.
- **Branch per phase** off `feat/pensieve-benchmark` (e.g. `fix/vmaf-alignment`); merge when validated.
- **Commit at the end of every phase**; large binaries (tars, checkpoints, media) go to a release/
  archive location or `.gitignore`, never bloated into git history.
- No new ns-3 sims or retraining happen until the data-integrity fixes land (Phases 1–2 gate 3–8).

## Readiness assessment (why this plan, not just polishing)
A journal reviewer will scrutinize correctness. Current blockers, in priority order:
1. **VMAF corrupt on 6/10 benchmark videos** — it is the eval metric AND a model input AND the
   teacher-label basis AND the video-selection key. Disqualifying if found.
2. **Permanent-freeze rebuffering booked as 0 s** for the LLM arm (decisions.jsonl finalizes stall
   only on resume). Distorts the QoE/congestion story.
3. **Statistical significance** — Ours vs Pensieve currently p=0.07 (not significant at 0.05).
4. **No content-aware baseline** — Ours is the only content+network policy; the content claim is
   hedged. A reviewer will ask for a content-aware comparator.
5. **Reproducibility** — the entire `scratch/ai_llm/` code tree is git-ignored (untracked).

---

## Phase 0 — Preserve & version-control current state (do first)
- Commit tracked edits (`.gitignore`, `scratch/run_ai_dash_llm.py`) + `docs/` (incl. this plan).
- `.gitignore` the large regenerable artifacts (`viewer_bundles.tar`, bundle dirs); record their
  build command instead.
- **Decide code-tracking for `scratch/ai_llm/`**: force-add the *source* (`*.py`, `*.md`, `config/*.json`,
  `benchmark/`, `pensieve/`, `slurm/`) while keeping checkpoints/data/shards ignored. This is the
  single most important reproducibility fix. (Needs a careful `git add -f` with an explicit include
  list so no large binaries slip in.)
- Decide checkpoint storage: `checkpoints_*` → an archive dir or IEEE DataPort/Zenodo release, with a
  manifest in-repo (hashes + provenance), not raw binaries in git.
- **Commit:** "chore: preserve viewer bundle exporter, docs, and OJ-COMS plan; track ai_llm source".

## Phase 1 — Fix data-integrity blockers (GATING)
**EXECUTED 2026-10-26 (cheap path):** full re-investigation confirmed the VMAF bug = 32/242 catalog
videos (test 4/26, val 5/26, train 20/128; 3 of the paper's 10) whose six rung encodes were
copy-segmented with a merged 4 s first segment vs the 2 s `original`, offsetting the VMAF reference and
semantic frames by one segment. Fix applied: `abr/scripts/realign_original.py` re-splits `original`
onto the rung timeline (rung media untouched, `*_prealign` archived, argmax-over-offset verified);
all 32 aligned (abr commit ccdb313). Recompute of frames→VMAF→semantic→metadata for the 32 running as
`abr/realign_affected.slurm` job 8124852. The stall fix (`extract_true_stall.true_stall_per_ue`) and
the 2.0 s segment-size fix landed in ns3 commit 49a51c9. Remaining Phase-1 item: validate job
8124852 output (top-rung VMAF median > 55 on the 32), then proceed to Phase 2.

Branch `fix/vmaf-alignment` + `fix/stall-accounting`.
- **VMAF:** root cause = per-segment temporal offset between encoded and "original" segments (matching
  frame counts, shifted window) in abr `_04_compute_quality_metrics.py`. Fix the segment alignment
  (verify with a known-good clip: motion-heavy segment VMAF should be high at high rungs), recompute
  VMAF for all affected videos, rebuild `output/metadata/*.csv` (archive old as `*_prevmaf`).
- **Stall:** fix `extract_true_stall.py` model path to use the ns-3 `FinalizeInterruptionTime`
  "Updated" value (terminal freeze) instead of `decisions.jsonl` `interruption_time_s`; distinguish
  genuine mid-content freeze from end-of-content idle (played < segment_count).
- **Validate:** re-run the VMAF-health scan (expect >55 median, monotonic) and the freeze audit
  (expect frozen UEs now carry real rebuffer).
- **Commit** each fix with before/after numbers.

## Phase 2 — Rebuild the pipeline on clean data
- Recompute `quality_sensitivity` + semantic features (depend on VMAF); refit feature scalers
  (archive old).
- Re-derive teacher labels (DP/greedy) on corrected VMAF.
- **Keep the held-out split fixed; evaluate on the FULL test set (26 videos), not the shrunk 10.**
  The 10-video shrink was for lgpu wall-time and cherry-picked by corrupt-VMAF headroom — running all
  26 removes that sub-selection bias. Recompute per-video headroom on clean VMAF for binning/reporting
  only (not re-selection).
- **Retrain the LLM** (offline imitation, documented budget) on the clean dataset → new
  `checkpoints_clean/`. Keep the old checkpoint archived for the record.
- **Commit:** "feat: rebuild dataset + retrain on corrected VMAF (clean pipeline)".

## Phase 3 — Baselines (fair, published)
Branch `feat/mpc-baselines`.
- Add **RobustMPC** (Yin, SIGCOMM'15) — the de-facto strong network-only baseline you currently lack.
- Add **quality-aware MPC** (RobustMPC with a VMAF QoE objective) — the content-aware comparator that
  properly tests the thesis: *content-aware LLM vs content-aware control*. Slot both into the serving
  bridge like Pensieve's `--policy` hook; no new data collection.
- Keep Pensieve (content-blind learned control) and the heuristics.
- **Risk to accept up front:** offline content effect was weak (v0≈v2, ΔR²≈0.05); Ours may not beat
  quality-MPC. That is the real test — plan the narrative for either outcome.
- **Commit:** "feat: RobustMPC + quality-aware MPC baselines".

## Phase 4 — Re-run benchmark on the FULL test set + statistical rigor
- Run the **full 26-video test set × 5 rates = 130 cells per arm** (vs the old 10-video/50-cell shrunk
  benchmark), clean VMAF, retrained LLM, all baselines. This is the headline change: full held-out
  test coverage, no sub-selection bias.
- **Compute note:** 130 LLM cells × seeds is heavy (lgpu wall-time was the reason for the original
  shrink). Budget more jobs / stagger; heuristics are cheap (C++), the LLM + quality-MPC dominate.
- **Multiple seeds** (≥3–5 RNG runs per cell) → report mean ± 95% CI and paired significance tests;
  target p<0.05 for the headline claim (fixes the current p=0.07). Larger test N also tightens CIs.
- Resolve the **mean-vs-median** instability and the **9 Mbps dip** honestly (report both; explain).
- Regenerate all figures/tables from the clean results (archive old `results/` as `results_prevmaf/`).
- **Commit:** "results: clean 50-cell benchmark with CIs + significance".

## Phase 5 — Content-vs-network ablation
- With VMAF fixed, run the ablation a journal will expect: network-only vs content+network variant of
  *our* policy, on the clean metric, to quantify the content contribution directly (the thing the
  "use-case framing" currently sidesteps). Report ΔQoE with CI.
- **Commit:** "results: content-vs-network ablation".

## Phase 6 — Manuscript (OJ-COMS specifics)
- Reformat to the **OJ-COMS template** (IEEEtran `open-access`); resolve the `IEEEtran.cls`/compile
  issue (compile in Overleaf or install the class).
- **Expand related work** to journal depth: MPC/RobustMPC, Pensieve, Oboe, Fugu/Puffer, Minerva,
  QARC, Comyco, BOLA — position the content-aware SLM precisely.
- Add a rigorous **Methodology/reproducibility** section: architecture dims, training budget (LLM and
  each baseline), reward weights, ladder, trace pool, seeds.
- Add **Limitations & threats to validity** (2.0 s media vs 3.3 s player de-rating; single testbed;
  content effect magnitude; simulation vs real traces).
- **Graphical abstract**, ORCID, author bios, index terms. Vector figures, colorblind-safe, readable
  at print size.
- Keep measured tone / de-AI prose (no em-dashes/filler), LLM-as-use-case framing.
- **Commit:** manuscript revisions.

## Phase 7 — Reproducibility package
- Public code release (tagged commit) with the tracked `scratch/ai_llm/` source + a top-level
  `REPRODUCE.md` (build on login, run matrix, score, regenerate figs).
- Data/checkpoint release (IEEE DataPort or Zenodo): corrected metadata, checkpoints, a small sample
  of run bundles; DOI cited in the paper.
- **Commit:** "docs: reproducibility package + release manifest".

## Phase 8 — Submission checklist
- Overleaf compiles cleanly on the OJ-COMS template; page/APC budget noted.
- All claims trace to a clean-pipeline number; no corrupt-VMAF figure remains.
- Significance stated with CIs; limitations present; related work complete.
- Figures/tables referenced, captioned, vector. Graphical abstract + bios + ORCID in.
- Cover letter; suggested reviewers; novelty statement.

## Dependency order
`0 → 1 → 2 → {3} → 4 → 5 → 6 → 7 → 8`. Phases 1–2 gate everything; no dataset expansion (we test on
the existing FULL 26-video held-out set, not a shrunk 10). 3 can be implemented in parallel with 2
but scored only after 2. Writing (6) proceeds in parallel but numbers are locked only after 4–5.
