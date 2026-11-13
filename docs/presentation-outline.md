# Content-Aware LLM-Based ABR for Non-Terrestrial Networks
## Supervisor presentation — slide outline + talking points

> Format: presentation (bullets + visuals). Depth: detailed-technical. Scope: the **final** system —
> architecture, methodology, dataset, **benchmarking (detailed)**, and results. Reuse figures in
> `scratch/ai_llm/benchmark/paper/` (`table1_oracle`, `fig1_pareto`, `fig2_curve`).
>
> Narrative arc: *content-aware LLM ABR for NTN → how it is built and trained → how we benchmark it
> rigorously against the ABR literature → it is competitive (tied #1 on perceptual %-of-oracle, zero stalls).*

---

## SECTION A — Motivation & framing

### Slide 1 — Title
- **Content-Aware LLM-Based Adaptive Bitrate Streaming for Non-Terrestrial Networks**
- Name · supervisor · date · affiliation.
- One-line subtitle: *a small multimodal LLM that chooses video bitrate from live network + content state,
  evaluated in a full closed-loop ns-3 NTN simulator.*

### Slide 2 — Problem & why NTN is hard
- DASH video is split into **segments**; an **ABR** algorithm picks a **bitrate rung** for each segment.
- **QoE = quality − rebuffering − quality-switching.** All three matter; they trade off.
- **NTN / satellite mmWave** is the hard setting: throughput is **highly variable** and **congested**, with
  large swings → ABR is stressed exactly where it matters.
- We optimise **perceptual quality (VMAF)**, not raw bitrate — bitrate is a poor proxy for what users see.
- *Talking point:* the question is whether a learned, content-aware policy can do better than the classic
  rate/buffer heuristics in this regime.

### Slide 3 — Glossary (vocabulary up front)
- **DASH / ABR** — segmented HTTP streaming / the bitrate-selection algorithm.
- **Bitrate ladder & rung** — the fixed set of encodings (here 6: 250k, 750k, 1500k, 3000k, 6000k, 12000k);
  a "rung" = one encoding.
- **VMAF** — Netflix perceptual quality metric, 0–100 (higher = better perceived quality).
- **QoE** — quality − rebuffering penalty − switching penalty.
- **Causal vs acausal** — decide using only present/past info vs. using future knowledge.
- **Oracle** — an idealised best-possible chooser; used as a performance **ceiling**.
- **% of oracle** — fraction of the achievable QoE that an algorithm captured.
- **Regret** — per-segment QoE gap from the oracle (lower = better).
- **NTN** — non-terrestrial (satellite) network.

### Slide 4 — Approach in one line
- A **small LLM (Qwen2.5-1.5B)** that **fuses** live **network state** + per-segment **content features** +
  the **encoding ladder**, and outputs the next **rung**.
- **Trained by imitation**; **evaluated in closed-loop ns-3 NTN** against the ABR literature.

---

## SECTION B — Final architecture

### Slide 5 — System overview  *(DIAGRAM)*
- Draw the **closed loop**: `ns-3 NTN sim (mmWave PHY + DASH client)  ⇄  LLM ABR policy`.
- Each segment: sim sends **state** → policy returns **chosen rung** → sim downloads it → next state.
- *Diagram should show:* UE(s), satellite/mmWave link, DASH client, the policy as a separate block, arrows
  for state-out / decision-in.

### Slide 6 — Model architecture  *(DIAGRAM)*
- **Frozen Qwen2.5-1.5B-Instruct** backbone (language model used as a sequence reasoner).
- **LoRA** adapters (the only backbone params trained).
- **Numeric soft-tokens** — project the numeric features into the LLM's embedding space.
- **Multimodal encoder** — three branches: **network**, **content/semantic**, **ladder**.
- **Encoder-residual policy head** → softmax over the 6 rungs.
- *Show trained vs frozen:* trained = LoRA + encoder + soft-token projection + head; frozen = the backbone.
- *Talking point:* small model (1.5B) — deployable; the LLM is the reasoning core, not a chatbot.

### Slide 7 — Inputs (what the model sees at each decision)
- **Network (7, +missing-indicator masks → 9):** throughput estimate, buffer level, buffer differential,
  segment fetch time, last-segment bitrate, smoothed RTT, recent loss rate.
- **Content / semantic (per segment):** quality-sensitivity, saliency, object score, scene-cut indicator.
- **Ladder context:** the 6 rungs × **[segment size, VMAF]** for *this* segment.
- **Window = 1** (single-snapshot decision).
- *Talking point:* it is genuinely **multimodal** — network + content + the per-segment quality ladder.

### Slide 8 — Output & the closed loop
- Output: one of **6 rungs** → mapped to a bitrate → ns-3 fetches that encoding → produces the next state.
- **IPC bridge:** the ns-3 binding runs under Python 3.6; the torch model under Python 3.11 → a small
  **stdin/stdout JSON inference server** couples them. The sim **pauses** during inference (free w.r.t. QoE,
  costs only wall-clock).
- **Serving-time no-stall guard:** a buffer-dynamics-aware clamp ensures the chosen rung is downloadable
  before the buffer drains (`bitrate ≤ throughput · max(buffer, seg_dur)/seg_dur · safety`). Standard ABR
  safety rail; it rarely overrides the policy (mostly only at cold start).

---

## SECTION C — Methodology: dataset, conversion, training

### Slide 9 — Data generation  *(DIAGRAM / pipeline)*
- Run **ns-3 NTN simulations** over: **many videos** × a **link-rate sweep** {100 Gb/s, 36, 18, 9, 4.5 Mbps}.
- Each run produces **raw traces**: the playback log + per-segment metadata, including **per-rung VMAF and
  size** for every segment.
- *Talking point:* the link-rate sweep deliberately spans uncongested → severe congestion so the data covers
  the regimes ABR must handle.

### Slide 10 — Trace → training rows  *(show ONE example row)*
- Conversion produces one row per **(segment)**: `observation` (the network features), `semantic_raw`
  (content features), `ladder` (per-rung [size, VMAF]), and a **teacher label** (the target rung).
- Features are **normalised at load** with fixed scalers (train/serve parity).
- *Show a real JSON row* so the supervisor sees exactly what a training example looks like.

### Slide 11 — Training objective + the two oracles
- **Training = imitation learning**: cross-entropy to a **teacher rung** per segment.
- **Teacher (training target) = a causal, throughput-sustainable expert:** choose the highest rung the
  **current** throughput can sustain (with headroom). *Causal* = present-only, so the policy learns decisions
  it can actually reproduce online.
- **DP-oracle (evaluation ceiling, separate):** an **acausal** dynamic program that, using the full future
  throughput, maximises **VMAF-QoE** `Σ w_vmaf·sem·VMAF − w_stall·rebuffer − w_switch·|ΔVMAF|`. Used **only**
  to measure "% of oracle" — never as a training target.
- *Talking point (one line):* we imitate a **causal** expert (deployable) and measure against the **acausal**
  optimum (the best-possible ceiling). They are different by design.

### Slide 12 — Split & training config
- **Whole-video, genre-stratified 70/15/15 split** → no segment-level leakage; **14 held-out test videos**
  used only for evaluation.
- Train: LoRA + encoder + soft-token projection + head; **window = 1**; on GPU.
- *Talking point:* the held-out split is what makes the benchmark a fair generalisation test.

---

## SECTION D — Benchmarking  *(first-class, detailed)*

### Slide 13 — The headline metric: % of DP-oracle (VMAF-QoE)
- For each (video, run): replay **policy / oracle / floor** rung sequences through the player model on the
  **realised** throughput, and compute `% = (policy − floor) / (oracle − floor)` (floor = always-lowest rung).
- **Meaning:** the fraction of the *achievable perceptual QoE* an algorithm captured (100% = optimal,
  0% = no better than always-lowest).
- **Complementary metrics** (reported alongside): **regret** (per-seg oracle−policy gap, no floor
  degeneracy), **stall time**, **switch count**, **mean delivered VMAF**.
- *Talking point:* % of oracle is the efficiency headline; regret + stalls are the robustness complements.

### Slide 14 — Benchmark design & fairness
- **7 algorithms:** our **LLM-ABR** + 6 classic ns-3 DASH heuristics — **SVAA, SFTM, RAAHS, FDASH, AAASH,
  OSMP** (Vergados et al. ns-3 DASH suite).
- **5 link-rate conditions** × **14 held-out test videos** → 70 cells per algorithm.
- **Full closed-loop ns-3 NTN** with the real mmWave PHY (not an abstract trace replay).
- **Fairness:** *every* algorithm is scored by the **identical** path — same player model, same DP-oracle
  ceiling, same videos/conditions/seeds.

### Slide 15 — How the field benchmarks vs. how we do  *(positioning)*
- **Field standard for learned ABR** (Pensieve, Comyco, NetLLM, …): **trace-driven chunk-level simulation**
  on public throughput traces (FCC, Norway/HSDPA, Oboe), scored with the linear Pensieve QoE — this
  **abstracts away the PHY** and is terrestrial.
- **Ours:** a **full closed-loop ns-3 NTN** simulation — more realistic (real link/queue/PHY, satellite
  setting).
- **Consequence:** we benchmark against the **ns-3-implementable classic heuristics**, and **cite** the
  terrestrial trace-driven learned baselines rather than re-running them out of their intended setting.
- *(If room)* **Reproducibility:** fixed seeds, held-out split, scoring scripts
  (`perceptual_qoe` → `score_v2_perceptual` → `aggregate_table`) regenerate the tables/figures.

---

## SECTION E — Results

### Slide 16 — Headline table  *(use `paper/table1_oracle`)*
- **% of DP-oracle by algorithm × link rate (+ overall).**
- **Our LLM-ABR = 30.3% overall, tied #1** with SVAA; ahead of SFTM (29.3), RAAHS (20.5), FDASH (18.1),
  AAASH (11.9), OSMP (0.7).
- Note the congestion column: **at 9 Mbps our model holds 18.6% while SVAA collapses to 2.9%.**

### Slide 17 — Figures  *(use `paper/fig1_pareto` + `paper/fig2_curve`)*
- **Pareto (Fig 1):** mean VMAF vs rebuffering — our model sits on the **quality–rebuffering frontier** with
  **zero stalls**.
- **Curve (Fig 2):** % of oracle vs link rate — our model **holds quality through congestion** where the best
  heuristic drops.

### Slide 18 — Results summary
- **Competitive on perceptual QoE** (co-leader, 30.3%), **lowest overall regret**, **≈ zero stalls**,
  graded throughput-tracking.
- One-line: *matches the best classic heuristic on perceptual QoE while being the most stable.*

---

## SECTION F — Close

### Slide 19 — Related work / positioning
- **Classic bitrate ABR:** BBA, BOLA (dash.js default), MPC + the ns-3 DASH suite — adapt **bitrate** to
  network state.
- **Perceptual / quality-aware ABR:** Comyco (VMAF + imitation), Fugu/Puffer (SSIM), GreenABR (VMAF).
- **LLM-for-ABR:** NetLLM (SIGCOMM'24, Llama2-7B, terrestrial).
- **Our niche:** **NTN + LLM + content/VMAF** — uncovered by the above.

### Slide 20 — Conclusion & future work
- **Conclusion:** a small content-aware LLM-ABR is **competitive with the ABR literature in NTN** on
  perceptual QoE, with strong stability (zero stalls, lowest regret).
- **Future work:** control-aware / RL training; stronger baselines (BOLA/MPC); real NTN traces; **isolating
  and strengthening the contribution of the content channel** (ongoing).

---

## BACKUP — anticipated supervisor questions (one slide each, only if asked)

- **Is the DP-oracle a fair ceiling — it's acausal?** Yes; it's an efficiency *upper bound* (no causal policy
  can beat it). That's why we also report **regret** and **stalls**, which are causal-fair.
- **Why a causal teacher and not train on the oracle directly?** The deployed policy is causal (no future
  knowledge); imitating an acausal optimum teaches hindsight bets it cannot reproduce online, leading to
  over-asking. A causal teacher yields decisions the model can actually make → stable online behaviour.
  *(This is the key methodological insight if probed.)*
- **Is the cross-algorithm comparison fair?** Identical scoring path, same oracle, same videos/conditions/seeds.
- **Why not include Pensieve/Comyco/NetLLM in the table?** They exist only as terrestrial trace-driven code;
  we compare against ns-3-implementable classics and cite the learned ones (Slide 15).
- **Does the content/semantic channel actually help?** The architecture ingests content features and is
  competitive; cleanly *isolating* the content channel's contribution is ongoing work (future work).
- **Generalisation?** Whole-video, genre-stratified split; 14 videos held out and used only for evaluation.

---

## Asset checklist
- `scratch/ai_llm/benchmark/paper/table1_oracle.tex` / `.md` → Slide 16.
- `scratch/ai_llm/benchmark/paper/fig1_pareto.pdf` → Slide 17.
- `scratch/ai_llm/benchmark/paper/fig2_curve.pdf` → Slide 17.
- `scratch/ai_llm/benchmark/paper/inline_stats.txt` → numbers for Slides 16/18.
- **To draw:** system-loop diagram (S5), architecture diagram (S6), data-generation pipeline (S9).
