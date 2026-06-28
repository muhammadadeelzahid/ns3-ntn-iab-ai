# V2 (LLM-ABR) Training Log — getting the frozen-LLM policy to learn

Running log of every attempt to get **V2** (encoder → soft-tokens → frozen Qwen2.5-1.5B + LoRA →
pool → head) to train on the congestion dataset. "Our work" arm = the LLM; V0 (encoder-only) is
**permanently off the table** as a fallback — if V2 collapses we keep tuning/re-architecting the LLM
from the literature.

**Symptom of failure ("collapse"):** `train_acc ≈ 0.329` (the majority-rung fraction), `train_loss ≈ 1.69`
(≈ entropy of the label prior), `val_macroF1 = 0.093`, recall `[0,0,0,0,0,1]` (predicts rung-5 always).
The model outputs the class prior **independent of input, even on the training set**.

**Key diagnostic (established 2026-06-24):** V0 (encoder→head, *no* LLM) trains fine (val macro-F1 ~0.44)
at the *same* `--lr-encoder 3e-2`. Inserting the frozen 1.5B LLM between encoder and head collapses it
across every config below. A *fresh* (untrained) V2 is input-sensitive (logit diff ~0.37) but the
*trained* checkpoint is input-invariant (diff ~0.0) → the encoder/soft-tokens are **driven to a constant
during training**. Conclusion: the frozen LLM is a **gradient barrier** — the encoder receives near-zero
useful gradient through it, so it never learns to produce informative soft tokens; the head then settles
on the prior. This is a *training-dynamics / architecture* problem, not a hyperparameter one.

Dataset: congestion campaign (job 7100419, 524/525 traces), linkRate sweep, numUes=3. Pure-QoE oracle
labels (w_rung=0). Backbone Qwen2.5-1.5B-Instruct, bf16 LoRA r=16 α=32 on q/k/v/o/gate/up/down.
Partition lgpu (L40s). Train script: `slurm/train_llm.sbatch v2`.

---

## Attempts

| # | Date | Change | Config | Ckpt dir | Result |
|---|------|--------|--------|----------|--------|
| 1 | ≤06-21 | Baseline V2 (fused vector reshaped → k soft tokens, **no LayerNorm**) | seeds 41/42/43; window 1 & 8; balanced on/off; soft-tokens 4/16/32 | checkpoints/, _old, _w1nb, _k16, _k32, _w8 | **COLLAPSE** (all). Soft tokens ~11.5× larger than real LLM embeddings (0.237 vs 0.0205) → frozen LLM saturates. |
| 2 | 06-24 | **LayerNorm on the fused soft-token projection** (NetLLM §4 cites LN for stability). soft-token |elem| 0.237→0.799, per-token std 1.0; fresh input-sensitivity 0.37 | 7101835 w8 no-bal; 7101836 w8 cwp0.5 | **COLLAPSE.** LN fixes token *scale* but not the gradient barrier — encoder still gets no signal through the frozen LLM. |
| 3 | 06-24 | **Encoder residual skip**: concat encoder fused embedding `e_fused` with pooled LLM output before the head (`encoder_residual=True`, ablatable). Gives the head a direct, un-throttled path to the discriminative encoder signal so it cannot collapse to the prior; LLM+LoRA still trains on top. (IAA inner-adaptor / multimodal-adapter residual, arXiv:2408.12902.) Smoke: grads flow to soft_proj/head/encoder; v0 untouched. | 7101837 w8 seed42 PATIENCE12 EPOCHS40 | checkpoints_resid | **✅ SOLVED.** Collapsed epochs 1-3 (F1 0.093); escaped epoch 4; climbed to **best val_macroF1=0.403** (ep10), val_acc 0.53, train_acc 0.67, spread recall across all 6 rungs. Early-stopped ep22. Comparable to V0 (~0.44) but it's the LLM and it TRAINS. Saved `checkpoints_resid/abr_v2_seed42.pt` (+ `.lora`). |

**OUTCOME: V2 collapse solved by the encoder residual skip (attempt #3).** We have a trainable V2 = the
"our work" benchmark arm. Caveat for the research story: the residual gives the head a direct encoder path,
so it likely leans on the encoder; an `encoder_residual=False` ablation (or zeroing the pooled LLM output)
is needed to measure the LLM's *marginal* contribution. If the LLM must be in the critical path (not a
bypass), Time-LLM reprogramming (lever #1) is the principled follow-up. But the immediate blocker —
"V2 won't train at all" — is cleared; the benchmark can proceed with this checkpoint.

---

## Candidate next levers (literature + reasoning), in priority order
*(implement only if the current attempt fails; one change at a time so the result is attributable)*

1. **Time-LLM reprogramming layer (KEEPS THE LLM IN THE CRITICAL PATH — best research story).**
   Instead of projecting features into *arbitrary* embedding vectors (out-of-distribution → frozen LLM
   saturates, our root cause), Time-LLM (ICLR'24, arXiv:2310.01728) *reprograms*: cross-attention with
   encoder features as queries against a small learned set of **real LLM vocab-token embeddings** as
   keys/values. The injected tokens are then convex combinations of actual vocabulary embeddings →
   **guaranteed in-distribution** for the frozen LLM. Directly targets the saturation/OOD problem that
   LayerNorm only partially patched. This is the principled alternative to the residual skip if we want
   the LLM (not a bypass) to do the work.
2. **Reparameterize SoftTokenProjection as a bottleneck MLP, not a single Linear.** Prefix-tuning
   (Li & Liang 2021) found *directly* optimizing soft prefixes is unstable and collapses; the standard fix
   is to generate them via a small MLP with a low-rank bottleneck + nonlinearity. Ours is one
   `Linear(d_fuse → k·H)` — swap for `Linear(d_fuse → r) → SiLU → Linear(r → k·H) → LayerNorm`. Cheap,
   well-supported. (HF PEFT docs; arXiv:2101.00190.)
3. **GPT4TS / "One Fits All" (NeurIPS'23, arXiv:2302.11939): freeze the LLM but fine-tune the
   LayerNorm affine params + positional embeddings.** Lets the frozen backbone's *entry/residual norms*
   adapt to the new modality without touching attention/MLP weights. Concretely: set `requires_grad=True`
   on the LLM's `*.norm`/`*_layernorm` params (and pos-emb if any) in addition to LoRA. Strong, simple,
   widely-replicated for frozen-LLM-on-non-text.
4. **Per-modality multi-token encoder (genuine NetLLM multimodal encoder).** NetLLM projects *each*
   modality/feature group into its *own* token (each LayerNorm'd) and feeds the LLM a *sequence*, rather
   than reshaping one fused vector into k arbitrary tokens. Richer/varied input tokens → LLM output varies
   more with input + more positions for the head to pool. (arXiv:2402.02338 §4.)
5. **Networking prediction head over per-token hidden states** instead of mean-pool. Try last-token /
   attention-pool over the soft-token positions (Time-LLM uses the final positions + linear projection).
6. **Higher LoRA capacity / more aggressive LLM-path LR** so the LLM adapts before the head settles on the
   prior (lr-llm 3e-4 → 1e-3; r 16→32). Cheap config test, combine with above.
7. **Two-stage / curriculum:** pretrain encoder+head as V0, then graft the LLM and fine-tune (warm-start the
   encoder so it never learns through the barrier from scratch).
8. **Gradient-flow instrumentation (DIAGNOSTIC):** log per-group grad norms (encoder vs soft_proj vs LoRA)
   to confirm where the signal dies — turns guesses into measurements. Do this regardless; it validates the
   gradient-barrier hypothesis directly.

## Benchmark prep (post-fix)
- **Offline checkpoint validation (06-24):** `checkpoints_resid/abr_v2_seed42.pt` saved with
  `policy_head.trunk.0.weight = (256, 1792)` = hidden 1536 + d_model 256 (residual head). All keys
  present (encoder 18, soft_proj 4 incl. `.norm`, policy_head 6). `create_model` defaults
  `encoder_residual=True`, so `evaluate.py` + live `inference.py` rebuild the matching arch and
  `load_state_dict` populates fully — **no bridge code change needed**. Checkpoint is window=8.
- **window=1 residual retrain launched (job 7101839 → checkpoints_resid_w1):** the live-bridge-proven
  recipe is window=1; this gives the benchmark arm the live-validated window AND tests that the residual
  fix reproduces across window settings (robustness check for this log).
- **window=1 residual retrain (7101839) DONE — reproduces the fix:** best val_macroF1=**0.418**,
  spread recall; early-stopped ep35. Confirms the encoder-residual fix is robust across window settings
  (w8=0.403, w1=0.418). → `checkpoints_resid_w1/abr_v2_seed42.pt` (the benchmark V2 arm; live-proven window).
- **SCORING PATH VALIDATED END-TO-END (06-24):** `ALGO_SHORT=Bola` extract_throughput (7101840) →
  dp_oracle (7101841, w_rung=0) → `dataset_builder --algo-short Bola` (49 rows, oracle attached 0 missing)
  → `perceptual_qoe.py --run-dir Quic_artifacts/Bola --teacher-dir /tmp/bola_teacher`. Produces the full
  metric table (%oracle, regret, meanVMAF, bitMbps, stall, switch). The benchmark infrastructure works.
- **BOLA baseline finding (smoke @ 18Mbps/3UEs):** pooled **6.5% of oracle**, one run 106s stall over
  27 segs (~89s content). `BolaClient` **over-asks** — oscillates to rung 5 (12Mbps) when per-UE share is
  ~5.5Mbps → stalls. This is a known BOLA weakness in high-RTT regimes; its `BufferTargetSeconds=20`/`Gp=5`
  defaults likely need NTN tuning, OR it stands as a legitimately weak NTN baseline. **Decide before/within
  the full matrix.** (Scoring is correct; this is a baseline-tuning question, not a metric bug.)

## FULL BENCHMARK MATRIX LAUNCHED (06-24, user: skip BOLA, full scope)
- **6 classic arms via dash_trace_launch** (SKIP_BUILD=1 to avoid waf-lock race; binary pre-built once):
  Fdash 7101842, Osmp 7101843, Aaash 7101943, Svaa 7102055, Raahs 7102056, Sftm 7102148. Each = 14 test
  videos × 5 linkRates {100Gb/s,36,18,9,4.5 Mbps} × 1 run = 70 sims → **420 classic sims**, numUes=3,
  SIM_STOP_CAP=120, aiEnable=false. Test videos: `scratch/ai_llm/benchmark/test_videos.txt` (held-out split).
- **V2 (LLM) arm** = `checkpoints_resid_w1/abr_v2_seed42.pt` via `run_ai_dash_llm.py` bridge. GPU is the
  bottleneck (lgpu=2 GPUs → 70 GPU sims would serialize ~87h), so run inference on **CPU** (ns-3 PHY
  dominates wall time, ~81 LLM calls/sim) → parallelizes across skylake like the classic arms. Bridge has
  per-job-unique shm names (concurrent-safe) + writes standard `Quic_artifacts/AiDash/<video>/run_<run>/`
  (scorable via ALGO_SHORT=AiDash). **V2 smoke 7102149** (pexels_18296909 @18Mbps, numUes=3, CPU) validates
  the residual-arch load-in-bridge + CPU feasibility before the full sweep. For the sweep, assign
  run=link_idx+1 per video so the 5 linkRates get distinct run dirs (avoid run_1 collision).
- **V2 BRIDGE VALIDATED (direct login-node test, no queue):** `inference_server.py` loads
  `checkpoints_resid_w1` on CPU, emits `{"ready": true}`, returns decisions via `reason:
  "llm_policy_model"` (real LLM path, not fallback). Residual architecture works in the bridge on CPU.
- **V2 SWEEP LAUNCHED:** 70 `bench-v2` jobs (`slurm/bench_v2_run.sbatch`, NEW) = 14 videos × 5 rates,
  run=link_idx+1, numUes=3, CPU, standard AiDash artifact paths. Map: `benchmark/v2_run_map.txt`.
  Smoke 7102149 cancelled (used old custom-path wrapper). **FULL MATRIX = 420 classic + 70 V2 = 490 sims.**
- **V2 ARM HEALTHY & ADAPTIVE AT SCALE (06-24 ~09:00):** all 70 bench-v2 running on CPU; decisions stream
  via `reason=llm_policy_model` (never fallback). Model ADAPTS to conditions — e.g. seg1→rung3 (6Mbps),
  seg2→rung1 (1.5Mbps) when RTT jumped to ~300ms. Input-dependent back-off, NOT the old majority-class
  collapse — closed-loop confirmation the residual fix works live. 0 failures across the matrix.
- Launcher edits: `dash_trace_launch.slurm` += `SKIP_BUILD` guard + `${SLURM_JOB_ID}` suffix on the
  job_matrix CSV (prevents same-second filename collision across parallel launchers).
- **Scoring (per arm, after sims finish):** `ALGO_SHORT=<X> extract_throughput → dp_oracle (w_rung=0) →
  dataset_builder --algo-short <X> → perceptual_qoe.py --run-dir Quic_artifacts/<X> --teacher-dir <built>`.

## SCORING (results land in scratch/ai_llm/benchmark/results/<ALGO>.txt)
- **Contamination fix:** `Quic_artifacts/<ALGO>` also holds ~91 leftover OLD training-campaign videos.
  The benchmark = the 14-video held-out test split only (the V2/AiDash arm ran exactly those). `score_report.sh`
  now prunes the teacher-dir to `benchmark/test_videos.txt` so perceptual_qoe scores only the 14 test
  videos × 5 conditions (70 runs/arm) → all arms comparable. Verified: Fdash scored exactly 70 runs.
- **Fdash (FIRST RESULT):** pooled **18.1% of oracle**, regret 13.1/seg, VMAF 25.5, stall 557s (≈ cached
  FDASH 19%). Per-condition gradient as expected: run_1 (100Gb/s) 38-73%, run_5 (4.5Mbps) →0% + stalls.
- TODO for final table: aggregate per-condition (group by run-index 1..5 = the 5 linkRates) across arms;
  the per-run rows already carry it. Score remaining arms at 70/70.

### Results (pooled %-of-oracle over 14 test videos × 5 conditions; results/<ALGO>.txt)
**ALL 6 CLASSIC BASELINES DONE.** Ranking: Svaa ≈ Sftm > Raahs > Fdash > Aaash ≫ Osmp.
| Arm | %oracle | regret/seg | meanVMAF | stall(s) | note |
|-----|--------:|-----------:|---------:|---------:|------|
| Svaa  | 30.3 | 12.3 | 42.8 | 812 | strongest classic |
| Sftm  | 29.3 |  8.8 | 32.8 | 156 | lowest regret+stall |
| Raahs | 20.5 | 11.1 | 32.9 | 453 | |
| Fdash | 18.1 | 13.1 | 25.5 | 557 | trace-collection policy |
| Aaash | 11.9 | 16.4 | 23.3 | 1180 | scored on 68/70 (2 hung run_1 cells excluded) |
| Osmp  |  0.6 | 687  | 37.1 | 53295 | ANOMALY: non-adaptive over-ask → catastrophic replay-stall; verify behavior-vs-artifact |
| **AiDash (V2)** | **pending** | | | | the LLM arm — re-running on fixed bridge (0/70, ETA ~16-17h) |

Hung: Aaash jobs 7101985_6/_66 still running (~7.5h, frozen) — uncongested run_1; not cancelled (user's call).

**V2 walltime issue — RESOLVED (06-24, user call).** CPU inference is ~6-7s/decision (heavy under 70-job
CPU contention) → V2 sims advance ~17 sim-s/h, needing ~7h to reach simStop=120. The first V2 sbatch set
only 6h walltime and running jobs can't be extended (scontrol permission denied), so they'd be killed
~18:35 at ~85-95% playback. Per user: **cancelled the running V2 jobs, cleaned logs + AiDash artifacts,
set `bench_v2_run.sbatch` walltime to 24h (`1-00:00:00`, skylake max), resubmitted all 70.** They now run
to full completion (no kill); ETA ~22:45.

**V2 SCORING PATH DIFFERS FROM CLASSIC (06-24) — score from decisions.jsonl, NOT traces.**
`dash_trace_worker` does `cd RUN_DIR` before ns-3, so classic arms get per-run sim.log + RxPacketTrace
(→ extract_throughput → dp_oracle → perceptual_qoe). The bridge (`run_ai_dash_llm.py`) runs ns-3 from the
repo ROOT, so all 70 V2 jobs write `./RxPacketTrace.txt`/`sim.log` there, CLOBBERING each other. **Sim is
still valid** (trace files are output-only, never read back); each run's **`decisions.jsonl` is correct
per-run** with chosen_rung + chosen_bitrate_bps + throughput_estimate_bps + buffer + interruption_time_s
per segment. So V2 must be scored from decisions.jsonl (old sweep used `eval_qoe.py`, but that gives raw
pillars, not %-oracle). **Plan (when V2 done):** compute V2 %-oracle from decisions.jsonl — chosen_rung =
policy, throughput_estimate_bps → dp_oracle feasibility, per-rung VMAF from metadata, same VMAF-QoE math as
perceptual_qoe. NO V2 re-run needed. (Cleanup: remove clobbered `./RxPacketTrace.txt` etc. from repo root after V2 finishes.)

**OFF-BY-ONE INDEXING BUG (06-24) — ONLINE-ONLY; FIXED.** Metadata + all 6 baselines + the training
data are 0-indexed (segments 0..N-1); ns-3's AiDashClient sends the bridge **1-indexed** segment ids
(1..N). `inference_server` did `metadata_index.get(seg_id)` → off-by-one: each live decision was fed the
NEXT segment's encoding ladder and the last segment overflowed (the "fallback" I earlier MISDIAGNOSED as
looping; the modulo patch was wrong and is removed). **Scope = online serving only.** Offline is correct:
verified a teacher row for seg 3 has ladder vmaf[6000k]=95.828 == metadata[3] — dataset_builder pairs each
segment S with metadata[S], so training + evaluate.py used aligned ladders (V2 val-F1 0.418 valid). **Fix:**
`inference_server` now `get(seg_id-1)` (tolerant fallback to `get(seg_id)` for a 0-indexed source).
Verified: seg 1→idx0, seg 9→idx8, no fallback. **The current online V2 run used the shifted ladder** → for
a clean online benchmark number, re-run V2 with the fix (decision pending user). Note: V2 choices are
network-state-dominated (top rung ~80% across all conditions), so the corrected run should move modestly.

**KEY FINDING — V2 model is healthy OFFLINE; the rung collapse is ONLINE-only (06-24).**
Offline eval on the held-out TEST split (1792 decisions, GPU, job 7104349, correct 0-indexed ladders)
gives a SPREAD, adaptive predicted-rung distribution; the buggy online run collapsed to the top rung:
| rung | offline test | online (buggy) |
|------|-------------:|---------------:|
| r0 | 24% | 3% |
| r1 | 4% | 0% |
| r2 | 10% | 2% |
| r3 | 10% | 0% |
| r4 | 18% | 15% |
| r5 | **34%** | **80%** |
(offline acc 0.438, macro-F1 0.408.) So V2 trained well — it uses all rungs and picks the lowest rung 24%
of the time; the online ~80%-r5 skew is a SERVING problem, not the model. Off-by-one ladder fix applied.
**CORRECTED V2 RE-RUN launched (70 jobs, fixed bridge, 24h walltime).** When it lands, MUST compare the
online rung distribution to the offline spread: if it moves toward offline → fix worked; if it still
collapses to r5 → deeper online/offline serving gap (network-feature scaling / window assembly) to chase
before V2's online number is trustworthy.

**ONLINE COLLAPSE ROOT CAUSE FOUND (06-24) — `last_segment_bitrate_bps` train/serve mismatch.**
Diagnosis of why online collapses to r5 (72%) vs offline spread (34%):
- Throughput feature is CORRECT online (tracks congestion: 174→10→6→3.3→1.93 Mbps per condition). Not it.
- At the SAME ~6 Mbps throughput, offline picks mean-rung ~2.5 but online ~4.4 → a DIFFERENT feature is shifted.
- Found it: the bridge's `CPP2PY_FIELDS` (run_ai_dash_llm.py:34) sends `current_bitrate_bps`/`prev_bitrate_bps`
  but NOT `last_segment_bitrate_bps` — the exact name the model's `_build_observation_dict` reads
  (`getattr(state,"last_segment_bitrate_bps",0.0)`). So online it's ALWAYS 0.0. It's a LOG-feature
  (scaler log_features) → 0 maps to an extreme z-score → 1 of 7 network features is constant-OOD every
  decision → skewed predictions. Offline it had real values (median 0.8 Mbps).
- Offline semantics (extract_trace.py:257): last_segment_bitrate = `metadata[seg-1].actual_bitrate_kbps[
  prev_rung]*1000` (VBR-actual bitrate of the PREVIOUS segment at the rung chosen for it; 0 at startup).
**FIX (to implement):** make `inference_server` replicate it — track per-node last chosen rung, and before
predict inject `last_segment_bitrate_bps = prev_segment_meta.actual_bitrate_kbps[LADDER_LEVELS[last_rung]]
*1000`. SegmentMetadata.actual_bitrate_kbps is available. Then smoke (1-2 sims) → check online rung dist
moves toward offline spread; if yes, final re-run. **The current re-run STILL has this bug (collapsed 72% r5)
→ stop it; it won't yield a usable V2 number.** (May not be the ONLY shift — also check the buffer/last-rung
self-reinforcement — but this is the clear, large OOD bug.)

**last_segment_bitrate hypothesis DISPROVED (06-25).** Decisive offline ablation
(`benchmark/ablate_last_bitrate.py`, GPU): zeroing last_segment_bitrate_bps on the test states gives
IDENTICAL predicted-rung dist (r5=34% both) → the model effectively IGNORES this feature. So the
train/serve mismatch is real but immaterial to the online collapse. (Kept the inference_server fix as
hygiene.) → online r5-collapse is NOT this serving bug. Running a systematic per-feature ablation (zero
each of the 7 net features) to find what the model actually relies on; if no single feature explains it,
the collapse is the imitation→control DISTRIBUTION SHIFT (online V2 visits OOD states FDASH never
generated) = the known live-loop null [[live-loop-qoe-result]], not a fixable serving bug.

**VALID ABLATION (evaluate.py --ablate, 06-25): the SEMANTIC channel is implicated.**
My earlier per-feature ablation (zeroing row['observation'][f]) was INVALID — dataset._features uses
load-time cached/windowed features, so the edits never reached the model (all-identical results). The
trusted evaluate.py --ablate (zeros the assembled tensor) gives:
- baseline: r5=34% (spread)
- zero NETWORK: collapses to r3=60%, zero r5  (model uses network; without it → middle rung)
- zero SEMANTIC: collapses to **r5=59%** ≈ the online 72% collapse.
→ The content/semantic channel is what pulls V2 OFF the top rung; online behaves ~semantic-blind.
BUT online `segment_meta.semantic_raw` IS populated ({motion,saliency,scene_cut,object,quality_sens}),
matching offline values — so it's not simply missing. Online re-derives semantic_norm from raw+scaler
(features.build_model_input, parity-by-design). off-by-one fix did NOT resolve (still 72%). REMAINING
suspects for the semantic-blindness online: semantic scaler mismatch, window/cold-start assembly, or the
fundamental imitation→control distribution shift (the known null [[live-loop-qoe-result]]). DECISION POINT
for user: accept V2 online over-ask as the honest negative result + score as-is, vs one bounded semantic
parity check. (Ruled out: last_segment_bitrate — model ~ignores it; off-by-one — fixed, immaterial here.)

## DAgger / covariate-shift FIX (06-25) — addressing the proven online collapse
Proven cause: training lacks low-buffer-under-congestion states (FDASH protects its buffer) → model
extrapolates "low buffer → high rung" online. Fix = add oracle-labeled low-buffer/congested states to
training + retrain. Stages, cheapest-first:
- **Stage B (RUNNING, cheap probe):** Osmp over-asker (drains buffer) on the 77 TRAIN videos × {18,9,4.5
  Mbps} (job 7105664, 231 fast classic sims, ~3h). Then ALGO_SHORT=Osmp extract_throughput → dp_oracle →
  dataset_builder → filter to train videos → aggregate with train_jsonl/train → retrain V2 → validate
  OFFLINE with `state_response.py` (does low-buffer→LOW-rung now?). No slow online run needed to validate.
  Video list: `scratch/ai_llm/benchmark/train_videos.txt`. Diagnostics: `benchmark/parity_check.py`
  (parity=exact), `benchmark/state_response.py` (the low-buffer→high-rung smoking gun).
- **Stage A (prereq for C):** fix bridge to write per-run traces (currently clobbers repo root).
- **Stage C (true DAgger, if B short):** iterate with V2's OWN online-induced states, 2-3 rounds.

## DAgger Stage B RESULT (06-25) — SUCCESS on held-out mid-stream low-buffer
**Diagnosis refinement (user catch):** we DO have low-buffer/low-throughput training data, but 96% is
STARTUP (seg 0, cold-start throughput artifact) where the oracle correctly picks HIGH (no real stall).
The genuinely-missing regime is MID-STREAM low-buffer under sustained congestion (FDASH protects its
buffer, never visits it). Validated: Osmp mid-stream low-buf states get oracle r0 (mean 0.30, 87% r0)
vs startup r5 (mean 3.69). So the model learned a STARTUP "low buffer→high rung" rule and misapplies it
mid-stream online.
**Metric fix:** state_response.py on the test split is the WRONG validator (its low-buffer is ~all
startup → conflates). Correct validator = `benchmark/validate_midstream.py` on HELD-OUT test-video Osmp
states (seg>2, buf<5s, thr<5M; 344 states, excluded from DAgger train).
**Result (model vs oracle at mid-stream low buffer):**
| ckpt | mean rung | r0% | r5% |
|------|----------:|----:|----:|
| oracle | 0.15 | 92 | 0 |
| baseline resid_w1 | 3.27 | 31 | **57** (over-ask bug) |
| **DAgger** | **0.01** | **100** | 0 (matches oracle) |
DAgger (checkpoints_dagger, augmented FDASH+Osmp, val-F1 **0.447** ≥ 0.418 baseline) now picks r0 at
mid-stream low buffer = the covariate-shift fix at its root, offline imitation intact, startup behavior
preserved. (Slightly over-conservative: r0 100% vs oracle 92% — safe direction.)
**NEXT:** online smoke (V2-dagger via bridge on a few congested videos) to confirm the closed-loop
collapse is actually reduced. NOTE: startup-exclusion alone was a red herring (its before/after was on
the flawed test-split metric).

## V2-DAgger ONLINE smoke (06-25, in progress)
Jobs 7106556-61: 3 test videos × {9Mbps run_4, 4.5Mbps run_5}, CKPT=checkpoints_dagger, CPU bridge.
EARLY (44 decisions, all llm_policy_model): DAgger mean rung 3.35 @9Mbps (baseline 4.16), 2.83 @4.5Mbps
(baseline 4.26) — ~1 rung more conservative already. INCONCLUSIVE yet: early/startup segments dominate
(buffer not drained). Right online success metric: DAgger should MAINTAIN buffer + not collapse to r5 +
stall less (if conservative, it avoids the low-buffer regime, so won't necessarily show r0 online).
Need fuller data + buffer-maintenance/stall comparison (DAgger vs baseline online).

**FULLER (86 decisions, in low-buffer regime) — FIX WORKS, partial:** online rung mean 2.47 @9Mbps
(baseline 4.16), 2.32 @4.5Mbps (baseline 4.26), pooled 2.41 vs ~4.2; r5 DROPPED 72-80%→~15%. Mechanism
confirmed online: at buf<5s (80 decisions) DAgger picks r0 44% (mean 2.38) where baseline picked r5 57%.
The closed-loop r5 collapse is BROKEN. PARTIAL: residual r4 ~38% under severe congestion → buffer still low
(median 0.3s, p10 -3.8) + some stall (max 25s cumulative). Note 4.5Mbps/3UE ≈ 1.5Mbps/UE share so low
buffer is partly genuine severity. END-TO-END VALIDATED (offline held-out r0 + online no-collapse).
Path to COMPLETE fix: 2nd DAgger round (iterate on V2-dagger's OWN online states) or more conservative
tuning. Optional: full-matrix re-score of the dagger arm for the headline %-of-oracle table.

## DAgger ROUND 2 (06-25, in progress) — true DAgger on V2's own states, decisions-based
Bridge can't write per-run traces cleanly (Experiment hard-coupled to cwd=ns3-root), so round-2 is
DECISIONS-BASED (no bridge surgery). Decisions logging EXTENDED (run_ai_dash_llm.py) to record full
6-field net obs (added buffer_differential_s, segment_fetch_time_s); last_segment_bitrate reconstructed
offline. NOTE: this logging change does NOT affect the model's online decisions (bridge always sent the
full state to the server) — earlier online tests remain valid; it only enables reusing online runs as
training data.
Pipeline BUILT + validated (923 rows on partial data):
- `benchmark/decisions_to_throughput.py` (A): decisions.jsonl → throughput_trace.jsonl (seg 0-indexed).
- dp_oracle ALGO_SHORT=AiDash (reuse) → per-VIDEO oracle_labels_state.jsonl.
- `benchmark/decisions_to_trainrows.py` (B): decisions+oracle(per-video, key run_id,node,seg0)+metadata
  → train_jsonl rows (obs incl. reconstructed last_bitrate, semantic_raw, ladder, teacher.chosen_bitrate_class).
COLLECTION: 90 V2-DAgger sims (30 train videos × {18,9,4.5Mbps}, CKPT=checkpoints_dagger), ~5-7h.
WHEN DONE: A→oracle→B on full data → aggregate train_jsonl_dagger + round-2 rows → train_jsonl_dagger2 →
retrain checkpoints_dagger2 (agro-b) → validate_midstream.py (held-out) + online smoke. Goal: close
residual r4 (round-1 online mean 2.41, 38% r4 @severe congestion). If good → full-matrix re-score.

**ROUND-2 retrain LAUNCHED (job 7107208, agro-b):** proceeded on partial collection (469 V2-DAgger
mid-stream low-buf states captured, ≈ round-1's 553 Osmp). Converter A→dp_oracle→B → 2453 round-2 rows
(83 runs). Aggregate train_jsonl_dagger2 = 699 shards / 22361 rows (FDASH 14448 + Osmp 5460 + V2-own 2453).
checkpoints_dagger2. Remaining round-2 sims left running (extra data, not awaited). NEXT: validate_midstream
on held-out + online smoke vs round-1 (online mean 2.41, 38% r4); if better → full-matrix re-score.

**ROUND-2 retrain DONE (val-F1 0.422). HELD-OUT mid-stream:** dagger2 r0=96% mean 0.21 (oracle 0.15;
round-1 dagger was r0=100% mean 0.01). Offline both round-1 and round-2 match the oracle (round-2 slightly
closer to 0.15). Online smoke (3 test videos × {9,4.5}Mbps, checkpoints_dagger2) launched — the decider for
whether round-2 closes round-1's residual 38% r4 (online mean 2.41). Pending.

**CRITICAL (06-26): V2-DAgger OSCILLATES online — mean-rung hid it.** The decisions-based %-of-oracle
scorer (`benchmark/score_v2_decisions.py`, reuses replay_session) reveals V2-DAgger's online policy
flip-flops r5↔r0 (e.g. rungs [5,0,5,0,0,5,0,0,5,0] vs oracle stable [5,4,4,4,...]). Huge switch penalty +
stalls → policy QoE -867 vs floor -0.2 → 0% of oracle. The "online mean rung 2.41" was an AVERAGE of r5 &
r0 picks — masked the oscillation. So the DAgger fix changed the failure mode (r5 lock-in → r5↔r0
oscillation) but did NOT yield a good online policy; on the QoE/%-oracle metric (what the baselines are
judged on) V2-DAgger is poor. Consistent with the fundamental online-control difficulty for this
imitation-trained LLM. Full-matrix re-score (70 sims, checkpoints_dagger) running → will quantify the
honest headline %-oracle on the 14 test videos. Scorer validated (correctly captures the oscillation cost).

**CONCRETE V2 SCORER PLAN (interfaces confirmed):** reuse `evaluate.replay_session(rungs, rows, ladders,
reward_cfg)` exactly as perceptual_qoe does. Per (node,seg) build a teacher-row dict with:
`teacher.throughput_bps_replay` = decisions.jsonl `throughput_estimate_bps`; `teacher[LABEL_FIELD_CLASS]`
= dp_oracle's optimal rung on that throughput+ladder; `ladder` = metadata per-rung (bitrate,VMAF) via
`features.ladder_context_from_row`. policy rungs = decisions.jsonl `chosen_rung`; floor = [0]. Then the
SAME %-oracle = (pol-floor)/(oracle-floor) clip+normalize. **Sanity checks:** (1) oracle-replay (pol=oracle)
must give ~100%; (2) rung histogram sane; (3) cross-check ladder/oracle construction by reproducing a
classic arm's %-oracle from metadata-ladders+trace-throughput vs its dataset_builder-teacher score. This is
the natural completion of scoring (not a re-run), same QoE math → comparable to baselines.

**Per-condition aggregator (`benchmark/aggregate_table.py`):** parses results/<ALGO>.txt per-run rows,
groups by run-index→linkRate, emits the mean %-of-oracle table (algo × condition). Verified: 'all' column
matches pooled. Per-condition (100Gb/s→4.5Mbps): Svaa 61→10, Sftm 41→15 (most congestion-robust, holds 28
@9Mbps), Raahs 41→9, Fdash 35→7, Aaash 15→1, Osmp ~0 everywhere. **Final step when V2 done:**
`score_report.sh AiDash` → `aggregate_table.py` for the headline V2-vs-field table.

**Open issues:** (1) Osmp 0.6% — likely real (rate-based heuristic over-asks in NTN congestion, like BOLA)
but the unbounded replay-stall (53k s) warrants a behavior sanity-check. (2) 2 Aaash run_1 sims hung
(uncongested, frozen sim.log @ 5.5h) — surfaced for the user; not cancelling per the reserved-decision rule.
(3) **V2 arm slower (~5h/sim, CPU inference); ETA ~13:30-14:00.** Progressing, NOT hung (sim-time ~75/120).
(4) **V2 loop-overflow metadata bug — FIXED + V2 RE-RUN (06-24 ~12:30).** Short videos replay past their
metadata under simStop=120; ns-3 keeps incrementing segment_id (seg≥num_segments). The C++ arms loop
natively (per-run `dash_metadata_segments.csv` records looped segs WITH real bitrates → teacher/oracle
covers them), but the Python bridge indexed source metadata by raw segment_id and returned
`fallback_missing_metadata` (a fixed 1.5Mbps rung) on overflow. Because the teacher DOES cover looped segs,
those fallback rungs WERE scored → V2 contaminated (not just uncongested ties). **Fix:** `inference_server.py`
maps overflow ids onto the replayed segment by position (`seg_keys[seg_id % len(seg_keys)]`) so the LLM makes
a REAL decision, matching the C++ arms; added `MetadataIndex.segment_ids()`. Verified directly: seg 9/15 now
return `llm_policy_model`, no `missing_metadata`. **Classic arms UNAFFECTED** (native looping) → Fdash/Svaa/Osmp
results stand. V2 (70 jobs) cancelled + AiDash artifacts cleared + resubmitted on fixed bridge (ETA ~17:30).

## STATUS: V2 blocker cleared; benchmark infra complete & validated
Trainable V2 arm = `checkpoints_resid_w1/` (val-F1 0.418, window=1). BOLA + 6 classic clients runnable;
metric proven on a real run. **Remaining: the full benchmark matrix** (V2 + BOLA + 6 classics × linkRate
conditions × test videos) — a multi-hour, hundreds-of-sims commitment. Open questions before launch:
(a) BOLA NTN param tuning vs leave-as-weak-baseline; (b) V2 live-arm runs via run_ai_dash_llm.py bridge
(checkpoint load-path already verified offline; per-UE feature assembly validated earlier in task #34).

### Research sources logged
- NetLLM — adapting LLMs for networking (multimodal encoder + networking head): arXiv:2402.02338
- Time-LLM — reprogramming via vocab-prototype cross-attention: arXiv:2310.01728 (ICLR'24)
- GPT4TS "One Fits All" — freeze LLM, tune LayerNorm + pos-emb: arXiv:2302.11939 (NeurIPS'23)
- Prefix-Tuning — reparameterize prefixes via MLP for stability: arXiv:2101.00190
- IAA Inner-Adaptor — residual adaptor empowering frozen LLM (basis for attempt #3): arXiv:2408.12902

## FINAL — V2-DAgger full-matrix re-score (06-26): HONEST NEGATIVE RESULT

Scored `checkpoints_dagger` (round-1 DAgger) online via the decisions-based %-of-oracle pipeline
(`decisions_to_throughput.py` → `dp_oracle ALGO_SHORT=AiDash` → `score_v2_decisions.py`) on the held-out
14 test videos × 5 linkRate conditions. The mmWave-PHY-bound sims flatlined well short of the 120s sim cap
(congested arms ~50% through; +38 decisions in the last 45 min) so the table is on **1568 collected
segments** — but the verdict is condition-robust (identical to the earlier 1530-seg preliminary).

### Headline: % of DP-oracle (VMAF-QoE) — V2-DAgger vs the field
| Algorithm        | % of oracle |
|------------------|-------------|
| SVAA             | 30.3 |
| SFTM             | 29.3 |
| RAAHS            | 20.5 |
| FDASH            | 18.1 |
| AAASH            | 11.9 |
| OSMP             |  0.6 |
| **V2-DAgger (ours)** | **0.0** |

Per condition (V2-DAgger): 100Gb/s 0.0% · 36Mbps 0.0% · 18Mbps 0.0% · 9Mbps 0.0% · 4.5Mbps 0.0% ·
POOLED 0.0% (meanVMAF 30.2, stall ~7s total). **0.0% even at 100Gb/s** (uncongested, where every
heuristic ties near the top): stalls are near-zero, so it is the *switching penalty* (`w_switch·|ΔVMAF|`),
not rebuffering, that drags policy QoE down to the all-min floor.

### Why: r0↔r5 oscillation (the DAgger fix changed the failure mode, didn't cure it)
| cond     | switch% | mean&#124;Δrung&#124; | rung-dist (0..5) |
|----------|---------|-----------|------------------|
| 100Gb/s  | 26.4% | 3.07 | 0:112 1:1 4:54 5:134 |
| 36Mbps   | 37.6% | 4.20 | 0:212 1:6 4:56 5:68 |
| 18Mbps   | 36.4% | 4.49 | 0:223 1:4 4:36 5:62 |
| 9Mbps    | 38.8% | 4.39 | 0:209 1:4 2:6 3:4 4:33 5:59 |
| 4.5Mbps  | 34.1% | 4.50 | 0:215 1:2 4:28 5:41 |

The distribution is **bimodal r0↔r5** — middle rungs (r1/r2/r3) are nearly empty — and mean |Δrung| ≈ 3–4.5
confirms the policy bounces between min and max segment-to-segment. Round-1 DAgger broke the original r5
lock-in (online r5 share 72–80% → ~15%; online mean rung 4.2 → 2.4), but the "mean rung 2.4" was an average
of r5 and r0 picks — it masked oscillation, not adaptation. Mid-stream low-buffer states now (correctly)
label r0, but the model still over-asks r5 elsewhere, so it whipsaws.

### Conclusion
**Imitation-trained LLM-ABR (frozen Qwen2.5-1.5B + LoRA + numeric soft-tokens) is not competitive online on
QoE.** Offline it spreads across rungs and matches the oracle on held-out states (val-F1 0.418→0.447 with
DAgger; held-out mid-stream r0 matches oracle); online it suffers covariate shift — first as r5 top-rung
lock-in (proven self-reinforcing, memory [[v2-online-collapse-mechanism]]), then, after DAgger, as r0↔r5
oscillation. Both failure modes put it at 0.0% of the DP-oracle, below all six classic causal heuristics.
The acausal DP-oracle is the ceiling; the heuristics' low absolute %-of-oracle (≤30%) reflect how hard the
NTN congestion regime is, but they remain **stable** policies whereas the LLM does not. This is an honest
negative result: closed-loop control demands temporal stability that single-step imitation does not supply,
and neither the residual-head architecture nor one round of DAgger closed that gap. Per [[llm-arm-is-the-work]]
the next lever is control-aware training (multi-step / smoothness-penalized objective), not a fallback to V0.

## PHASE A (06-26): post-hoc smoothing probe — GATE FAILS, oscillation is a symptom not the cause

`benchmark/smooth_probe.py` re-scores the SAME collected V2-DAgger decisions (182 sessions) after de-jittering
the chosen-rung sequence per (run,node), reusing `evaluate.replay_session` (identity filter reproduces the
0.0% baseline exactly; oracle = 100% — both sanity-pass).

| filter      | %oracle | switch% | mean&#124;Δrung&#124; | meanVMAF |
|-------------|---------|---------|---------|----------|
| identity    | 0.0%    | 34.8%   | 4.20    | 30.2 |
| dwell2/3/4  | 0.0%    | 12.0 / 6.9 / 5.3% | 4.5–4.8 | 28–38 |
| median3/5   | 0.0%    | 17.7 / 11.2% | 4.1 | 25–29 |
| ratelimit1  | 0.0%    | 62.6%   | 1.00    | 35.2 |
| ORACLE      | 100.0%  | 21.9%   | 1.32    | 55.9 |

**Every realistic smoother collapses switch rate (34.8%→5.3%) but %-oracle stays 0.0% in EVERY condition.**
meanVMAF *rises* under de-jittering (30.2→~38): the r0↔r5 oscillation locks onto the HIGH rung, and a stable
over-ask under congestion stalls as badly as the jitter. So oscillation is a **symptom**; the disease is
**wrong rung levels** — the model commits to high rungs the oracle never picks. The oracle is not constant
(switch 21.9%, |Δrung| 1.32): it adapts *smoothly at the right level*, which no post-hoc filter on bad choices
can synthesize. → Serving-side smoothing (B1) is NOT warranted; the indicated path is **B2 control-aware
retraining** (smoothness/switch-penalized sequence loss in `train.py`) that learns correct + stable levels
jointly. Reconvene with user at this gate before launching the retrain.

## ROOT CAUSE (06-26): the acausal DP-oracle is a MIS-SPECIFIED EXPERT for causal control

Inspecting the round-2 DAgger corrective rows (`train_jsonl_dagger2_rows`, 2453 rows = `checkpoints_dagger`'s
OWN online states on train videos, oracle-labeled) revealed why iterated DAgger plateaus:

- Oracle-label rung dist over ALL corrective rows: {0:295, 1:53, 2:101, 3:318, 4:631, **5:1055**} — 69% r4/r5.
- Even restricted to the FAILURE regime (mid-stream seg>2 AND buffer<5s, n=625): {0:**6**, 2:46, 3:148,
  4:261, 5:164} — **92% r3–r5**, almost no r0. The corrective signal *reinforces* over-asking.
- Smoking gun: on **87%** of those failure states the oracle's chosen bitrate is **ABOVE the current
  throughput estimate** (median 1.61×, p90 2.56×, max 4.9×).

The DP-oracle is **acausal** — it sees future throughput, so in a draining-buffer state it bets on recovery
and labels a HIGH rung. A causal policy can't see that, so imitating these labels = systematic over-asking =
online stall/collapse. This is the true mechanism behind the whole offline↔online gap and the DAgger plateau:
**the expert itself produces the bad behavior.** Iterated DAgger with this oracle cannot correct over-asking
(rounds 1→2 went from online mean 2.41 to 2.80, i.e. MORE aggressive). The fix must change the EXPERT to a
causal / stall-conservative one (high-`w_stall` oracle, or a causal MPC/throughput-sustainable rule that
never asks above current throughput), OR move to closed-loop RL whose QoE reward penalizes the stalls the
acausal oracle ignores. Reconvene with user — this overturns the plain "iterated DAgger" plan.

## METHOD CAVEAT (06-26): decisions-based %-oracle is DEGENERATE on truncated bridge sessions

While probing the throughput-sustainable causal expert (`benchmark/causal_expert_probe.py`) it scored 0.0%
at every safety factor too — which triggered a look at the raw QoE components and exposed a scoring problem:

- The 70 rescore sims are TRUNCATED (flatlined/PHY-bound, still running) so decisions sessions are only
  **5–7 segments**, mostly inside the player's 10s startup window (`startup_threshold_s=10`, seg_dur 3.3s).
- With `w_switch=0.8·|ΔVMAF|` (one r0↔r5 switch ≈ 44 penalty ≈ 68 VMAF-pts of quality), the DP-oracle on a
  5-segment trace just picks a FLAT LOW rung to avoid switches. In several sessions the labeled "oracle"
  scores BELOW the all-min floor (e.g. ora −16.0 vs flr 0.0) — impossible for a true ceiling → the
  floor-normalized %-oracle is degenerate here. **No stalls anywhere**: the 0% is switch-penalty + startup
  accounting on tiny sessions, not rebuffering.

Consequences:
- The **0% magnitudes are NOT trustworthy** — neither the causal-rule probe nor the earlier "V2-DAgger 0%"
  headline. The 6 heuristics (18–30%) were scored on COMPLETE full-video sessions via `perceptual_qoe`;
  V2/causal were scored on TRUNCATED 5–7-seg bridge sessions via decisions-replay. Apples-to-oranges.
- **Still solid (length-independent):** V2 oscillates online (switch 35%, |Δrung| 4.2 — direct measure) and
  the acausal oracle over-asks (bitrate>throughput on 87% of failure states — training-data analysis,
  [[acausal-oracle-misspecified-expert]]). The causal-teacher fix remains mechanism-justified; the probe
  simply can't adjudicate teacher quality on a broken instrument.
- ACTION: a fair %-oracle requires COMPLETE closed-loop sessions. Validate the causal-teacher retrain on
  complete online smoke runs, and re-score the headline V2 arm only on complete sessions.

## CAUSAL-EXPERT RETRAIN (06-26): launched — fixes the over-ask at the SOURCE (labels)

User chose: iterated DAgger with a CAUSAL expert. Built `benchmark/relabel_causal.py` (rung = max r with
actual_bitrate[r] ≤ throughput_estimate × sf, sf=0.9) and relabeled the base congestion training set:
`train_jsonl/{train,val}` → `train_jsonl_causal/{train,val}` (14448 train / 2433 val). Causal rung-dist is
healthy/throughput-driven: {0:582,1:2009,2:3093,3:2730,4:1822,5:4212} (vs the acausal oracle's r5-heavy
{...5:4752}). Retrain LAUNCHED: job **7108047** (agro-b A30, `checkpoints_causal`, window=1 recipe via
`train_llm.sbatch`, DATA_DIR=train_jsonl_causal). Epoch 1 val-F1 0.072 (cold start, climbing).

### Validation plan — over-ask rate is the PRIMARY (length-robust) metric, not %-oracle
The decisions-based %-oracle is unreliable on the bridge arm for a SECOND structural reason beyond
truncation: test videos are SHORT (4–21 segs) so ns-3 LOOPS them to fill the 120s cap; the C++ heuristic
arms score looped segs via per-run `dash_metadata_segments.csv`, but the V2 scorer uses SOURCE metadata and
DROPS looped segs → V2 sessions are capped at 4–21 scorable segs by construction (apples-to-oranges vs the
heuristics' full sessions). Fixing that is separate scorer infra. So the causal retrain is validated by the
DIRECT, length-robust metric that measures the actual disease:

**over-ask rate** = fraction of online decisions with chosen_bitrate > current throughput_estimate (the
proven collapse driver), plus switch rate / |Δrung|.

**BASELINE to beat — `checkpoints_dagger` online (test-video decisions):**
| cond     | over-ask% | switch% | mean&#124;Δrung&#124; |
|----------|-----------|---------|---------|
| 100Gb/s  | 11.3% | 26.4% | 3.07 |
| 36Mbps   | 14.5% | 37.6% | 4.17 |
| 18Mbps   | 20.6% | 36.4% | 4.49 |
| 9Mbps    | 28.6% | 39.5% | 4.42 |
| 4.5Mbps  | 24.7% | 35.1% | 4.53 |
| POOLED   | **19.9%** | **35.2%** | **4.21** |

Target: causal model drives over-ask → ~0% (teacher never asks above thr×0.9) and cuts switch rate. Online
smoke on a few test videos under congestion once `checkpoints_causal` finishes, same metric.

## CAUSAL RETRAIN RESULT (06-26): offline over-ask HALVED — strong green light

Training converged: `checkpoints_causal` val-macroF1 **0.86** (vs ~0.42 acausal) — the causal label is a near
-deterministic function of throughput so it's highly imitable, balanced recall 0.71–0.96 across all rungs.
Snapshot `checkpoints_causal_snap.pt` (best, F1 0.860) used for eval to avoid the still-running job's writes.

OFFLINE covariate-shift over-ask check (`benchmark/overask_predict.py`: predict on the SAME 1568 online
states checkpoints_dagger visited; `decisions_to_trainrows.py` → `AbrJsonlDataset` → argmax; over-ask =
predicted bitrate > current throughput):

| pooled (same states) | DAGGER | CAUSAL |
|----------------------|--------|--------|
| over-ask %           | 19.5%  | **8.4%** |
| mean &#124;Δrung&#124;        | 3.88   | **1.77** |
| switch %             | 47.6%  | 48.4%  |

(dagger re-prediction reproduces the 19.5% baseline → method validated.) Per-condition over-ask falls
across the board, most at congestion (9M 28.3→9.2, 4.5M 24.1→18.2; the 4.5M residual is partly structural —
throughput below r0). Rung-dist flips from bimodal r0↔r5 to GRADED throughput-tracking (18M→mostly r3,
4.5M→r0–r2, 100Gb→r5). The causal expert fixed the over-ask at the SOURCE. Residual: switch FREQUENCY still
~48% — frequent SMALL adjustments (rate-tracker chattiness), a candidate for a later hysteresis/buffer term.

CLOSED-LOOP smoke LAUNCHED to confirm the over-ask collapse survives online (no buffer drain → no collapse):
jobs 7108229-32 = {pexels_5257861, pexels_35139203} × {9, 4.5}Mbps, SIMSTOP=60, CKPT=checkpoints_causal_snap,
separate `Quic_artifacts/AiDash_causal/` root (does NOT clobber the dagger baseline). Validate same metric on
the causal model's OWN online states. NOTE: %-oracle is NOT the metric (looped-segment scorer gap); over-ask
/ switch rate is.

## CAUSAL closed-loop smoke (06-26, PARTIAL ~25%): over-ask collapse SURVIVES; residual = STARTUP

Jobs 7108229-32 running (SIMSTOP=60). Over-ask/switch from the causal model's OWN online decisions:
| cond    | over-ask: dagger→causal | &#124;Δrung&#124;: dagger→causal | med buf |
|---------|-------------------------|------------------------|---------|
| 9Mbps   | 28.3% → **13.6%**       | ~3.9 → **1.62**        | 6.4s |
| 4.5Mbps | 24.1% → **2.3%**        | ~4.0 → **1.69**        | 7.4s |

The mid-stream collapse is GONE — graded throughput-tracking, median buffer healthy. BUT the closed loop
exposes a residual: **startup over-ask**. Sample `seg=3 rung=5 buf=-3.3 thr=1269k` → picked r5 (~20Mbps) at
1.3Mbps throughput → buffer negative → 42 interruptions. 9Mbps run logged 361 interruptions / 14.2s stall,
nearly all in seg 2-5; 4.5Mbps only 5 / 0.4s. Cause: the cold-start throughput ESTIMATE is optimistic, so
the causal relabel inherited "startup→high rung" there (same startup artifact as the original low-buffer
analysis). CAVEAT: partial data is startup-HEAVY (runs ~25% into a 60s sim, 10s startup window) so this read
is pessimistic on stalls; steady-state will dominate the complete run. Residual is a narrow, tractable
sub-problem (conservative startup guard / startup-aware throughput floor), NOT the mid-stream collapse.
Let smoke finish → final steady-state-dominated read + verdict.

## CAUSAL closed-loop FINAL-ish read (06-26, ~55%): mid-stream collapse FIXED; startup is the residual

Startup(seg≤2) vs mid-stream(seg>2), interruption_time_s is CUMULATIVE (max per session = total stall):
- **MID-STREAM over-ask 5–6%** (vs dagger 28/24%), |Δrung| 1.4–1.6 (vs ~4.0), median buffer **9.8s healthy**,
  rung-dist graded+low (9M→mostly r2, 4.5M→r1–r2). The r0↔r5 collapse is GONE.
- Per-session stall: **pexels_35139203 — ALL 5 sessions 0.0s stall**, final buf 13–26s (both 9M & 4.5M
  clean). **pexels_5257861 — 3 of 6 sessions stalled** 4.3–7.0s (the nodes that over-asked early → buffer
  deficit → stall → slower progress); other 3 clean (buf 8–17s).

VERDICT: the throughput-sustainable causal expert **fixes the mid-stream over-ask collapse online** — graded
tracking, healthy steady-state buffer, one whole video fully stall-free. Residual = **startup over-ask** (a
few seg2–5 r5 picks at low cold-start throughput cascade into stalls on some sessions). Narrow + tractable,
NOT the universal dagger collapse. Next step (small follow-up): a STARTUP GUARD — conservative cold-start
throughput floor / cap the initial rung until a real throughput estimate forms. See memory
[[causal-expert-fixes-online-collapse]].

### COMPLETE-run final (06-26): 8/11 sessions stall-free
Mid-stream: 9Mbps over-ask 5% |Δrung| 1.53 medBuf 9.8s; 4.5Mbps over-ask 7% |Δrung| 1.36 medBuf 9.7s.
Per-session stall: **pexels_35139203 all 5 stall-free** (buf 14–26s); **pexels_5257861 3/6 stalled** 4.3–7.0s
(the seg-16 nodes that over-asked at startup). Startup over-ask 9M 17% / 4.5M 0%. VERDICT CONFIRMED: causal
expert fixes the mid-stream collapse online (8/11 sessions perfect); the only stalls are startup-driven on
one video → task #56 startup guard.

## FOLLOW-UPS (06-26): startup guard + looped-segment scorer fix + cleanup

(1) **Cancelled** the 13 stale `bench-v2` dagger-rescore jobs (flatlined, verdict settled) — queue clear.

(2) **STARTUP GUARD** added to `inference_server.py` (`_sustainable_rung` + buffer-aware clamp): when
`buffer_level_s < --guard-buffer-s` (default 5s) the chosen rung is capped to the throughput-sustainable
rung (`actual_bitrate ≤ throughput × --guard-safety`, default 0.9) — the same constraint the causal expert
was trained on; above the threshold the model is trusted (may spend buffer). Default ON; bridge spawns the
server so it auto-applies. OFFLINE sim on the causal smoke decisions: **startup over-ask 27.3%→4.5%**,
mid-stream 6.2%→2.6% (residual is structural: throughput below r0). Confirming guard-ON smoke LAUNCHED (jobs
7108901-04, `Quic_artifacts/AiDash_causal_guard/`, same snapshot ckpt for a clean A/B vs guard-off
`AiDash_causal`). Expect the 3 startup-stalled pexels_5257861 sessions to go stall-free.

GUARD-ON smoke (PARTIAL ~30%, 06-26): guard FIRES correctly — sample reason
`llm_policy_model+guard(buf0.3<5.0:r5->r0)` (caught the exact startup r5-at-empty-buffer pathology). The 3
pexels_5257861 sessions that stalled 4.3/7.0/6.8s guard-OFF now show **0.0/1.0/0.0s** — startup stalls
eliminated. Startup over-ask 9M 16.7%→8.3%; steady-state unharmed (mid over-ask 4.8%→2.6%). (over-ask metric
uses nominal ladder vs the guard's actual-bitrate budget, so the % reduction understates it — the STALL
elimination is the real proof.) Let complete for the final steady-state confirmation.

GUARD v2 — DYNAMICS-AWARE (06-26): the fixed buf<5s threshold guard has a GAP. Guard-ON smoke exposed a NEW
9.2s stall (4.5M node7): at seg3 buf=7.9s (ABOVE the 5s threshold so guard inactive) the model picked r5 at
2610k throughput → a ~25s download drained the 7.9s buffer to negative → stall. A single egregious over-ask
drains even a healthy buffer. FIX: replaced the threshold with a buffer-DYNAMICS-aware no-stall cap
(`_feasible_rung`): `actual_bitrate ≤ throughput * max(buffer,seg_dur)/seg_dur * safety` — always on; low
buffer ⇒ ~sustainable, high buffer ⇒ proportional spend. Offline-validated on guard-OFF decisions: startup
over-ask 27.3→4.5%, mid 6.2→2.6%, AND the seg3 case now clamps r5→r3 (download fits buffer) while at
buf=9.8s it still allows r5 (steady-state NOT over-constrained). Args: `--guard-safety` (0=off, default 0.9),
`--guard-seg-dur-s` (3.3). Confirming dynamics-guard smoke launching to `Quic_artifacts/AiDash_causal_guard2/`.

DYNAMICS-GUARD smoke (PARTIAL ~30%, 06-26): the critical clamp `seg3 buf7.9 r5->r3` FIRES (the exact
over-ask that crashed the threshold guard). Per-session stall — ALL eliminated vs both prior runs:
| pexels_5257861 session | guard-OFF | threshold | DYNAMICS |
|------------------------|-----------|-----------|----------|
| 4.5M node6 | 6.8s | 0.0s | **0.0s** |
| 4.5M node7 | 0.0s | 9.2s(gap) | **0.0s** |
| 9M node6   | 4.3s | 0.0s | **0.0s** |
| 9M node7   | 7.0s | 1.0s | **0.0s** |
pexels_35139203 all clean. The stall-danger window (startup + seg3 over-ask) is captured; all sessions past
it at 0 stall. Over-ask (vs raw thr, which the dynamics guard intentionally allows above thr at high buffer):
9M startup 8.3% mid 6.1%, 4.5M startup 0% mid 2.9%. Letting it complete for the rigorous final close.

### DYNAMICS-GUARD FINAL (06-26): online over-ask collapse FIXED end-to-end ✅
Near-complete run (all 11 sessions seg 11–20, past the seg1–5 stall window):
- **STALLS: 0/11 sessions, 0.0s total** (vs guard-OFF 3/11, 18.1s; threshold-guard had a NEW 9.2s).
- **Steady-state NOT over-constrained**: mid |Δrung| 1.58 vs 1.53 (9M), 1.12 vs 1.36 (4.5M); medBuf 8.8/10.8s
  vs OFF 9.8/9.7s — comparable. Startup over-ask 9M 8.3% / 4.5M 0%.
The buffer-dynamics guard eliminates every stall (the 3 guard-off ones AND the threshold-guard's regression)
without harming steady-state buffer-spending. **VERDICT: the online over-ask collapse is resolved end-to-end:**
(1) causal-expert retrain fixes the mid-stream collapse (over-ask 28→5%, healthy buffers, graded tracking),
(2) buffer-dynamics no-stall guard fixes the startup + healthy-buffer egregious over-asks (stalls 18.1→0s),
(3) looped-segment scorer fix makes short looped videos scorable. The LLM ABR arm now produces a stable,
non-collapsing online policy. (Metric = stall / over-ask / buffer-health; %-oracle deferred — needs the
scorer + longer sessions for a fair headline.)

(3) **LOOPED-SEGMENT SCORER FIX** (`scripts/dp_oracle.py` + `benchmark/score_v2_decisions.py`): short test
videos loop to fill the 120s cap; the bridge scorer/oracle used SOURCE metadata only and silently dropped
looped segs (capping V2 sessions at the video length — apples-to-oranges vs the C++ arms' full looped
sessions). Added a guarded modulo map (`seg_id % N_source`) so looped playback gets its source-segment ladder
VMAF/sizes; the throughput converter already emits all segs. Verified: non-looped scoring unchanged; synthetic
looped test maps 21→0, 24→3, 42→0. CAVEAT: %-oracle still has short-session/startup degeneracy beyond
looping → a fully fair headline also needs complete/longer sessions (deferred).

## %-ORACLE COMPARISON (06-26): scoring pipeline FIXED + V2 benchmark matrix launched

Goal: put V2 (causal+guard) in the SAME %-of-oracle table as the 6 heuristics. Fixed the scoring path so
V2 is scored IDENTICALLY to the heuristics (`perceptual_qoe.score_run`: per-(video,run) floor-normalized
%-oracle, mean per condition via aggregate_table) — the previous `score_v2_decisions` did ratio-of-sums
(mean-of-ratios ≠ ratio-of-sums → unfair).

THREE bugs fixed (all were producing garbage):
1. **Throughput basis** — `decisions_to_throughput.py` used the ABR's logged `throughput_estimate_bps`,
   which is ~10× BELOW the realized rate (size/fetch). `replay_session` re-simulating downloads on that
   estimate invented huge phantom stalls (19060s in a 60s sim). FIX: reconstruct realized throughput =
   chosen rung's ACTUAL bytes / `segment_fetch_time_s` (needs metadata_dir arg; looped-aware).
2. **Missing `throughput_bps_replay`** — Converter B never emitted it → `replay_session` fell back to the
   10 kbps floor → same phantom stalls. FIX: Converter B now emits realized `throughput_bps_replay`.
3. **Path/format** — `perceptual_qoe` gained `_algo_rungs_from_decisions` + `score_run(decisions_jsonl=)`
   so V2 delivered rungs come from `decisions.jsonl chosen_rung`; new driver `score_v2_perceptual.py`;
   Converter B video-split made root-agnostic (`/AiDash/` hardcode removed).

VALIDATION (guard2 4 cells): stalls now SANE (0.0/4.3s, matching decisions' interruption_time); V2 raw QoE
(pol −338) CLOSE to oracle (−314). The 0.0% on these 2 short videos is the SAME floor-degeneracy the
heuristics hit on short/startup-dominated sessions (oracle ≈ floor under the switch penalty — heuristic
per-run rows show many 0.0% too). So %-oracle only discriminates on longer videos / uncongested conditions;
will also report REGRET (per-seg oracle−policy QoE, no floor degeneracy).

LAUNCHED: 70 V2 sims (14 test videos × 5 conditions, `checkpoints_causal` + dynamics guard, SIMSTOP=120,
`Quic_artifacts/AiDash_v2bench/`, jobs in /tmp/v2bench_jobs.txt) — matches the heuristic matrix exactly.
~half a day. Score when done → V2 row vs SVAA 30.3 / SFTM 29.3 / RAAHS 20.5 / FDASH 18.1 / AAASH 11.9 /
OSMP 0.6, + per-condition + regret.

## %-ORACLE HEADLINE (06-27): V2 not competitive on efficiency, but ZERO stalls + lowest regret

Full 14×5 matrix (67/70 cells scored; causal+guard, SIMSTOP=120, scored via the IDENTICAL
`perceptual_qoe.score_run` path as the heuristics). %-of-oracle (mean over runs per condition):

| Algorithm | 100Gb/s | 36Mbps | 18Mbps | 9Mbps | 4.5Mbps | **all** |
|-----------|---------|--------|--------|-------|---------|---------|
| Svaa      | 61.3 | 47.2 | 30.3 | 2.9  | 9.7  | **30.3** |
| Sftm      | 40.7 | 35.8 | 26.5 | 28.3 | 15.2 | **29.3** |
| Raahs     | 41.1 | 25.0 | 17.6 | 9.7  | 9.2  | **20.5** |
| Fdash     | 34.9 | 27.2 | 13.5 | 7.2  | 7.4  | **18.1** |
| Aaash     | 15.0 | 14.1 | 22.5 | 6.8  | 1.3  | **11.9** |
| Osmp      | 0.0  | 0.0  | 0.0  | 3.2  | 0.0  | **0.7**  |
| **V2 (ours)** | **0.0** | **0.0** | **0.0** | **7.1** | **35.7** | **9.0** |

V2 is near the bottom on %-oracle (9.0%, beats only Osmp), leading only at the most severe congestion
(4.5M 35.7%). BUT the diagnostics flip the story:
- **REGRET (per-seg oracle−policy QoE, lower=better) — V2 is BEST in every condition**: V2
  4.2/5.8/5.8/3.8/1.9 vs Svaa 4.6/7.8/15.2/20.0/13.7, Sftm 8.9/9.7/10.3/7.4/7.5. So V2 tracks the oracle
  CLOSEST in absolute QoE.
- **ZERO stalls everywhere** (stall=0.0 all conditions) — the over-ask collapse is fully cured online.
- The %-oracle/regret CONTRADICTION resolves to **chattiness**: with no stalls, V2's QoE still lands BELOW
  its all-min floor at high/moderate BW purely from the **switch penalty** (w_switch·|ΔVMAF|=0.8·|ΔVMAF|).
  V2 switches ~4–12×/run (2–3× the heuristics) — even at 100Gb/s where it mostly picks r5, frequent
  switches drag QoE under the stable floor → floor-normalized %-oracle clips to 0%.

CONCLUSION: causal-expert + guard made V2 STABLE (no collapse, no stalls, oracle-tracking levels = lowest
regret), but it is NOT competitive on the switch-penalized %-oracle efficiency metric because it is CHATTY.
The underlying rung LEVELS are good; the missing piece is temporal SMOOTHNESS. Clear next lever (not started,
needs user OK): a smoothness/hysteresis layer (serving-side dwell, or the B2 switch-penalized training loss)
to cut switch frequency — should lift %-oracle substantially since regret is already lowest and stalls are 0.
Honest caveat: %-oracle floor-normalization is harsh/clippy on short test videos; regret is the more robust
absolute metric and favors V2. Both reported. Scoring pipeline: [[causal-expert-fixes-online-collapse]].

## %-ORACLE CORRECTED (06-27): scoring bug fixed → V2 TIES FOR #1; chattiness was a red herring

The 9.0% headline was a SCORING BUG, not a V2 deficiency. `replay_session`'s quality reward is
`w_vmaf·teacher.semantic_weight·vmaf`; Converter B (`decisions_to_trainrows.py`) dropped `semantic_weight`
(→0.0) so V2's quality term was zeroed (only switch/stall penalties survived → negative QoE → 0%). The
heuristics' shards carry it (`dataset_builder.py:155`), so the comparison was unfair. FIX: Converter B now
keeps the full dp_oracle label and emits `semantic_weight`/`is_scene_cut`/`vmaf_chosen`. Re-scored the
existing 70-cell `AiDash_v2bench` decisions (NO new sims):

| Algorithm | 100Gb/s | 36M | 18M | 9M | 4.5M | **all** |
|-----------|---------|-----|-----|----|----|---------|
| **V2 (ours)** | 53.5 | 43.8 | 28.2 | 18.6 | 9.9 | **30.3** |
| Svaa      | 61.3 | 47.2 | 30.3 | 2.9  | 9.7  | **30.3** |
| Sftm      | 40.7 | 35.8 | 26.5 | 28.3 | 15.2 | **29.3** |
| Raahs     | 41.1 | 25.0 | 17.6 | 9.7  | 9.2  | **20.5** |
| Fdash     | 34.9 | 27.2 | 13.5 | 7.2  | 7.4  | **18.1** |
| Aaash     | 15.0 | 14.1 | 22.5 | 6.8  | 1.3  | **11.9** |
| Osmp      | 0.0  | 0.0  | 0.0  | 3.2  | 0.0  | **0.7**  |

**V2 = 30.3% overall, TIED FOR #1 with Svaa** (Svaa's all-cap is dragged down by a 9M collapse to 2.9%;
V2 holds 18.6% there). V2 leads at 9M, comparable at 18M/4.5M, slightly behind Svaa at 100Gb/s/36M.

STAGE-2 GATE — chattiness NOT worth fixing. Under the corrected reward, output dwell-smoothing HURTS:
identity 30.3%/regret7.3/sw9.9 → dwell4 29.7%/regret11.3/sw1.1. It cuts switches 9.9→1.1 but %-oracle dips
and regret worsens, because with quality credited V2's switches are mostly PRODUCTIVE (throughput-tracking
to capture quality); smoothing loses more quality than it saves in switch penalty. (The earlier "dwell lifts
9→31%" was an artifact of the penalty-only broken reward.) → Do NOT add hysteresis / switch-penalized loss.

VERDICT: the causal-expert + dynamics-guard V2 is COMPETITIVE — co-leader on perceptual %-oracle, with NO
stalls and the over-ask collapse cured. The only "fix needed" was the scoring bug. Minor optional residuals:
severe-congestion tail (9M/4.5M V2 18.6/9.9 vs Sftm 28.3/15.2 — a condition-aware smoother could help only
there) and the startup guard's needless first-segment clamp at high BW (Finding 3); neither is essential.

## Notes
- All collapsed checkpoints are architecture-incompatible with the residual head (head input dim changed
  from H to H+d_model) — fine, they were degenerate anyway.
- Never fall back to V0. See memory [[llm-arm-is-the-work]].
