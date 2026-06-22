# Dataset Expansion Plan

Current state: **25 videos × 5 ns-3 runs = ~24,228 decision rows**.

Goal: reach ~100k rows with broader content and network-condition diversity to improve rung-1 and rung-2 recall, reduce variance across seeds, and enable a credible unseen-content generalisation split.

---

## EXECUTION LOG — 2026-06-18 (catalog scaled 25 → 105 videos)

Executed the 100+-video expansion. Outcome:

- **Cleanup:** wiped `train_jsonl/`, fitted scalers, all ABR `output/`, logs, and Slurm artifacts for a fresh run. Kept the 25 existing `input/*.mp4`. Slurm queue cancelled to empty first.
- **Discovery:** expanded `discover_pexels_4k.py` `SEARCH_QUERIES` from 14 → 36 queries (added gaming, news, concert, sports, timelapse, animation, snow, desert, volcano, fireworks, space, fire, equestrian, scuba, ballet, motorsport, etc.). Ran `discover_pexels_4k.py --count 80 --add-to-catalog` → **80 new Pexels 4K entries appended**, catalog now **105 enabled videos** (103 Pexels + 2 Blender).
- **Blender films NOT viable:** Spring / Sprite Fright / Charge / Coffee Run / Cosmos Laundromat are behind Blender Cloud, not on `download.blender.org/demo/movies/`. Only BBB + Tears of Steel exist there as direct 4K (already in catalog). `Sintel_4k/` is TIFF sequences only. **Dropped the 5-Blender-film step**; reached 105 via Pexels alone.
- **Bug fixed:** discovery wrote emoji (🎥📸✪) from Pexels photographer names into `attribution` fields; the cluster's **system Python 3.6 (narrow Unicode)** yaml loader rejected them, failing the launcher. Stripped emoji from the catalog and added `_ascii_safe()` sanitization to `discover_pexels_4k.py` so future runs are clean. Lesson: **always validate the catalog with `/usr/bin/python3`, not the venv** — the launchers use system Python.
- **Download:** submitted `MAX_PARALLEL=10 sbatch download_videos_launch.slurm` (no FORCE → 25 existing skipped, 80 new fetched).

**Audit:** `run_audit_video_resolution.sh` → **105/105 KEEP**, 0 below-4K, 0 errors. All downloaded videos meet the 2160 long-edge gate.

**ABR pipeline:** submitted `MAX_PARALLEL=20 sbatch run_batch_launch.slurm` → array job **7086274** (`1-105%20`) → writes `abr/output/metadata/<id>_segments.csv` per video.

### Full remaining chain (each stage gates the next)

The ns-3 simulation traces (`Quic_artifacts/Fdash/<video>/run_*`) currently exist for **only the original 25 videos**. The 80 new videos need the full trace campaign:

| # | Stage | Script (repo) | Input | Output |
|---|-------|--------------|-------|--------|
| 1 | ✅ Download | `download_videos_launch.slurm` (abr) | catalog | `input/*.mp4` |
| 2 | ⏳ ABR pipeline | `run_batch_launch.slurm` (abr) | mp4 | `output/metadata/*_segments.csv` |
| 3 | ⬜ ns-3 sim campaign | `dash_trace_launch.slurm` (ns3) | metadata CSVs (reads `abr/output/metadata` directly, `RUNS_PER_METADATA=5`) | `Quic_artifacts/Fdash/<video>/run_*/sim.log` + `trace_decisions.jsonl` |
| 4 | ⬜ Throughput traces | `extract_throughput_traces.slurm` (ns3) | sim.log | `throughput_trace.jsonl` |
| 5 | ⬜ DP oracle | `dp_oracle.slurm` (ns3) | throughput | `oracle_labels_state.jsonl` |
| 6 | ⬜ Build JSONL | `dataset_builder.py` (ns3) | traces + oracle + metadata | `train_jsonl/*.jsonl` |
| 7 | ⬜ Refit scalers + retrain | `train.py` + `slurm/sweep.sh` (ns3) | JSONL | checkpoints |

**Stage 3 is the heaviest:** 105 videos × 5 runs = **525 ns-3 simulation tasks** (worker walltime 24h each, `dash_trace_worker.slurm`). The docs recommend running `dash_trace_smoke.slurm` (one end-to-end run) to validate before launching the full array.

**Decision (2026-06-18):** user chose **auto-launch the full campaign**. When ABR batch 7086274 drains, the agent will submit `dash_trace_launch.slurm` (RUNS_PER_METADATA=5, reads `abr/output/metadata` → 525 tasks), then chain throughput extraction → DP oracle → `dataset_builder.py` → scaler refit → `sweep.sh` retrain. Background monitor `b1hc66gkd` is the trigger.

---

## Lever 1 — More ns-3 Runs Per Video (highest ROI, no new ABR pipeline cost)

Each additional run per video = ~193 rows (24,228 / 125 shards). Going from 5 → 10 runs doubles the dataset to ~48k rows with zero new video downloads or ABR pipeline work.

**What to vary across new runs:**

| Trace parameter | Current range | Expand to |
|----------------|--------------|-----------|
| Satellite elevation angle | fixed | low (15°), mid (45°), high (85°) |
| Rain-fade attenuation | none | light (3 dB), heavy (10 dB) |
| LEO handover mid-stream | none | add 1–2 per run |
| Background TCP cross-traffic | light | 20%, 50%, 80% channel load |
| UE mobility speed | static | 0, 30, 120 km/h |

This enriches the `observation` feature distribution (throughput, RTT, loss) without touching the video catalog or the ABR pipeline.

**How:** edit `scratch/ai_llm/slurm/train_llm.sbatch` `--num-runs` argument and the ns-3 run-generation script. Add entries `run_6` … `run_10` to `build_job_matrix.py`.

---

## Lever 2 — Expand the Video Catalog (25 → 50 videos)

The existing `../abr/scripts/discover_pexels_4k.py` already automates Pexels discovery.

### Missing content categories (none currently in catalog)

| Category | Why it matters for ABR |
|----------|----------------------|
| **Gaming / esports** | High frame rate, high motion, text overlays — stresses rung selection at low buffer |
| **News / talking-head broadcast** | Very low motion, prominent text, face-heavy — tests `text_score` + `face_score` signals |
| **Concert / music video** | Rapid cuts, crowd, lighting effects — high `scene_cut_count` |
| **Sports broadcast** | Scoreboard text, fast motion, mixed dark/bright scenes |
| **Timelapse (day→night)** | Extreme motion score range within one clip |
| **Animated series (non-Blender)** | Flat colour regions → high compressibility; different VMAF curve from live action |
| **Snow / winter landscape** | Low texture complexity, low motion — easy to compress |
| **Desert / arid landscape** | Similar, but tests sand-grain texture |
| **Screen recording / tutorial** | Text-dominated, static — near-zero motion |

**Recommended additions — 25 new Pexels 4K IDs to search for:**

Add these queries to `discover_pexels_4k.py`'s `SEARCH_QUERIES` list:

```python
"4k esports gaming screen",
"4k news anchor broadcast studio",
"4k concert crowd stage lights",
"4k football soccer match aerial",
"4k timelapse city sunrise sunset",
"4k cartoon animation colorful",
"4k snow blizzard winter",
"4k sahara desert sand dunes",
"4k volcano eruption lava",
"4k space satellite orbit earth",
"4k fireworks night sky",
"4k marathon running crowd",
```

**Additional free 4K sources (no API key needed):**

| Source | Notes |
|--------|-------|
| Blender Open Movie — *Cosmos Laundromat* | CC-BY-4.0, 12 min, 4K; download from blender.org |
| Blender Open Movie — *Agent 327* | CC-BY-4.0, animation, high motion |
| Coverr.co | Free MP4s, many 4K, no key required |
| Mixkit.co | Free 4K stock, no key |
| Internet Archive — Ephemeral Films | Public domain, varied content |
| Xiph.org — *Crowd Run* 4K | YUV raw, needs transcode; classic ABR benchmark |

Running `FORCE_DOWNLOAD=1 sbatch download_videos_launch.slurm` after updating the catalog pulls all new videos automatically.

---

## Lever 3 — Extend Clip Duration (120s → 240s)

The catalog caps each video at `max_duration_s: 120`. Doubling to 240 s would ~2× rows per video. Cost: re-run the ABR pipeline on all 25 (or 50) videos with the new cap, which takes ~2.5 GPU-hours per video.

Do this only after Lever 1 is exhausted — it has the same row yield as Lever 1 but costs ABR pipeline recomputation.

**How:** change `max_duration_s` in `config/video_catalog.yaml` and re-submit `run_batch_launch.slurm`.

---

## Lever 4 — Use More Semantic Features From the ABR Pipeline

The model currently uses only **5 of 12** available semantic features:

| Used | Unused (computed by `../abr` pipeline) |
|------|---------------------------------------|
| motion_score | `spatial_detail_score` |
| saliency_score | `edge_density_score` |
| scene_cut_count | `texture_complexity_score` |
| object_score | `person_score` |
| quality_sensitivity | `face_score` |
| | `text_score` |
| | `compression_complexity_score` |

These are already in every `_segments.csv` output. Adding them to `SEMANTIC_FEATURE_NAMES` in `scratch/ai_llm/features.py` (and refitting the semantic scaler) is a zero-download change that expands the semantic input from 5 → 12 features. It is especially valuable for the underrepresented rungs: rung 1 and 2 are almost always selected when content is highly compressible (low `texture_complexity_score`, low `spatial_detail_score`), so those features directly address the rung-1 recall gap.

**Steps:**
1. Add the 7 feature names to `SEMANTIC_FEATURE_NAMES` in `features.py`
2. Refit scalers: `python3 scratch/ai_llm/train.py --fit-scalers-only`
3. Rebuild `all_runs.jsonl` (scaler now applied at load time, so raw JSONL is unchanged)
4. Retrain V0/V2 with the expanded semantic input; compare macro-F1

---

## Lever 5 — Add PSNR/SSIM Per Rung to Ladder Context

The current `ladder_context` stores `[size_bytes, vmaf]` per rung (6 × 2 = 12 values). The ABR pipeline also computes `psnr` and `ssim` per resolution. Adding them would make the ladder context 6 × 4 = 24 values and give the model a richer picture of quality vs. rate trade-offs.

**Caveat:** the pipeline's resolutions (360p/480p/720p/1080p) don't map 1:1 to the 6-rung ns-3 ladder. Either:
- Linearly interpolate PSNR/SSIM from the 4 ABR resolutions to the 6 ns-3 rungs, or
- Rerun the ABR pipeline with a 6-rung ladder matching `NS3_RATE_LADDER_BPS` (preferred; requires config change in `config/default.yaml`)

---

## Recommended Sequencing

| Step | Lever | Expected rows | GPU hours |
|------|-------|--------------|-----------|
| 1 | Lever 1 — 5→10 runs per video | ~48k | ~2h (CPU, ns-3) |
| 2 | Lever 4 — expand to 12 semantic features | (same rows, richer signal) | 0 |
| 3 | Lever 2 — 25→50 videos | ~100k | ~60h ABR pipeline |
| 4 | Lever 3 — 120→240s clips | ~200k | ~60h ABR re-pipeline |
| 5 | Lever 5 — PSNR/SSIM in ladder | (same rows, richer signal) | 0 |

Steps 1 and 2 can be done immediately. Step 3 requires a Pexels API key and ~60 GPU hours for the ABR pipeline batch. Steps 4–5 follow once the expanded catalog is validated.
