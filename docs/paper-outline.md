# Paper outline — Content-Aware Adaptive Bitrate Streaming with a Small Language Model over Satellite 5G IAB Networks

**Final title (locked):** *Content-Aware Adaptive Bitrate Streaming with a Small Language Model
over Satellite 5G IAB Networks* — a deliberate sequel to the first paper's title pattern
("… Video Streaming … over Satellite 5G IAB Networks").

Draft plan for the second paper (sequel to the arXiv testbed paper). Target: **IEEE Network (magazine),
double-column, hard cap ≤ 7 pages *including references*** — same venue type and house style as the
first paper. Magazine conventions apply: accessible/tutorial tone for a broad networking audience,
**figure- and insight-driven** (not equation-heavy), a **lean reference list** (group citations), and
a short "challenges / future directions" beat near the end. Authors: same as the first paper
(Zahid, Hossain, Hu). This is a planning doc — section blueprints, verified literature, and style
exemplars. Prose drafting comes next.

**Citation convention.** Inline keys like `[Mao'17]` resolve to the full entries in §10. Every
blueprint bullet carries a `*Cite:*` line naming exactly which references anchor that claim, so the
citation map is explicit before drafting. `*Cite:* —` = no external source (our own testbed,
measurements, architecture, or ablations). `[add: …]` = a reference used inline but not yet in §10
(collected in the §10 flags).

---

## 0. Framing decisions (locked)

- **No coined acronym; the paper is title-anchored.** In-text handle = **"our content-aware SLM
  policy"** (or just "our policy"); in tables/figures the method row is labeled **"Ours"**. Naming
  the backbone a "small language model (SLM)" in the title also quietly differentiates from the
  terrestrial "LLM-ABR" (He et al. 2024) and NetLLM without a fight over the name. Do **not** reuse
  "LLM-ABR" or "SLM-ABR".
- **No "oracle" anywhere.** The greedy causal policy is **"the teacher"** — a *training target*
  in §5 only. No evaluation ceiling, no "% of oracle" denominator.
- **Lead with absolute QoE, not a normalized fraction.** Headline number:
  **Ours 9.55 vs. best heuristic (RAAHS) 7.42 = +28.7% per-chunk VMAF-QoE.**
- **Sequel hook.** Prior work (Zahid, Hossain, Hu; arXiv 2026) established the ns-3 satellite-IAB
  MPEG-DASH testbed and showed QUIC+BBR is the best transport. This paper reuses that testbed and
  asks what a learned, content-aware policy on top can do — so the testbed is *cited*, not
  re-derived.
- **Hard length cap: ≤ 7 pages including references.** Everything (body + figures + tables +
  bibliography) fits in 7 double-column pages. This forces the tight float budget (4 floats, §4.x)
  and a compressed related-work section, and it is why the testbed is cited rather than re-described.

### One-line thesis
> Our **content-aware SLM policy** — a small (1.5B), LoRA-adapted, multimodal language-model ABR policy — delivers
> the **highest perceptual (VMAF) QoE under congestion** of any evaluated policy, with **near-zero
> stalls**, in a **full closed-loop ns-3 non-terrestrial-network** simulation.

Three pillars, all leading: **content-aware** (what it is) → **best QoE under congestion** (the
result) → **full closed-loop ns-3 NTN** (a *methodological* pillar: the policy is graded on the
network its own decisions induce over a real satellite mmWave/IAB stack — something trace-driven
ABR work structurally cannot do).

### Honesty guardrails
- "A content-aware policy achieves the best QoE" is defensible. "Content-awareness is *why* it
  wins" is **not** — the offline ablation (`v3-language-channel-result`) did not robustly isolate a
  content-channel gain. Keep content-awareness as an *architectural pillar + system-level strength*;
  do not claim a clean causal ablation.
- Confirm the held-out video count before submission (presentation-outline says 14; the shrunk
  benchmark matrix says 10 headroom-stratified). State the actual number used for the reported table.
- "OSMP" in older notes → the ns-3 client is **OSMF**. Fix in the baselines list.

---

## 1. Abstract (blueprint)
1. ABR trades quality vs. stalls vs. switching; NTN (volatile, congested, high-RTT satellite links)
   stresses this exactly where it matters.
2. Prior learned/LLM ABR is terrestrial + trace-driven, abstracting the PHY; we run a *content-aware
   small-LM policy* in a *full closed-loop* ns-3 NTN loop.
3. Our policy: frozen Qwen2.5-1.5B + LoRA, multimodal soft-token encoder (network + per-segment content
   + VMAF ladder), imitation-trained on a causal teacher.
4. Result: highest QoE under congestion among 7 policies (**+28.7% per-chunk VMAF-QoE** over the best
   heuristic), near-zero stalls.

---

## 2. Introduction (6-paragraph funnel) — now ALSO absorbs Background (§3 merged in)

> **[2026-09-07] Structural change (done in `latex-article/slm_abr.tex`).** The standalone Background
> section is gone; its still-missing content was folded into the Introduction: VMAF defined as the
> perceptual metric (P3); multi-hop / shared-beam NTN detail + why throughput-prediction breaks (P2);
> sim-to-real strengths, sweepable congestion + per-rung ground-truth VMAF (P4); the honest
> "why an LM" caveat + why-small (P5). **Related Work moved up to become §II** (see §8).

- **P1 — Problem.** Video dominates traffic; DASH/ABR picks a rung per segment; QoE = quality −
  rebuffering − switching; the three trade off.
  *Cite:* MPEG-DASH [Sodagar'11]; the QoE decomposition (quality − rebuffer − switching) [Mao'17, Yin'15].
- **P2 — Why NTN is hard & timely.** LEO/GEO satellite backhaul → high, variable RTT, congestion,
  handover-driven throughput swings.
  ABR is stressed most here.
  *Cite:* prior testbed [Zahid'26]; Starlink measurement [Michel'22, Izhikevich'24]; NTN architecture [3GPP TR 38.821]; LEO streaming [Fang'24].
- **P3 — Two gaps.** (a) *What is optimized*: prior ABR, including the learned/LLM systems
  (Pensieve, Comyco, NetLLM, He'24 "LLM-ABR"), targets *bitrate* — a poor proxy for *perceptual*
  quality (VMAF). (b) *How it is evaluated*: those systems are validated **trace-driven and
  terrestrial** — replaying recorded throughput the policy cannot affect, over an abstracted PHY —
  never in a closed loop on a satellite stack. (LLM-for-ABR to date is also inline-over-numbers.)
  *Cite:* learned/LLM ABR [Mao'17, Huang'19, Wu'24, He'24]; trace-driven datasets [Akhtar'18] (+[FCC'16, Riiser'13]); perceptual metric [Li'16, Bampis'19].
- **P4 — Why full-stack, closed-loop simulation (ns-3) — the second pillar.** ABR is a *control
  loop*: the chosen bitrate pushes bytes into the network, reshaping queueing, self-induced
  congestion, RTT (bufferbloat) and loss — and hence *future* available throughput. **Trace replay
  severs this feedback** and grades the policy on a network it cannot influence; it also cannot
  represent the mmWave NR PHY/MAC, IAB backhaul, QUIC/BBR transport, handover, or beam-capacity
  sharing that define the NTN regime. We instead evaluate in **ns-3** — a packet-level,
  discrete-event, standards-aligned full-stack simulator — so the policy is graded on the network
  *its own decisions create*, over a real satellite mmWave/IAB stack. This is the faithful middle
  ground between trace replay (an unrealistic loop) and live deployment (uncontrollable,
  unreproducible); we reuse the validated testbed from our prior work [Zahid'26].
  *Cite:* full-stack ns-3 mmWave/IAB [Mezzavilla'18, Polese'18]; the testbed [Zahid'26]; sim-to-real caution [Yan'20]; bufferbloat [Gettys'12].
- **P5 — Approach.** Our content-aware SLM policy: frozen Qwen2.5-1.5B + LoRA, multimodal (network +
  per-segment content + VMAF ladder), VMAF-aware, causal-teacher imitation — evaluated in the
  closed-loop ns-3 NTN loop against 6 classic ns-3 heuristics.
  *Cite:* backbone [Qwen'24]; adapters [Hu'22]; the 6 heuristics — see §6.1.
- **P6 — Contributions (bulleted):**
  - **C1 (method):** A **content-aware small-LM ABR policy** — multimodal soft-token fusion + frozen
    Qwen2.5-1.5B + LoRA — optimizing perceptual (VMAF) QoE.
  - **C2 (methodology / realism):** The **first learned ABR evaluated in a full closed-loop ns-3 NTN
    loop** — graded on the network its own decisions induce, over a real mmWave/IAB satellite stack —
    not trace replay, not terrestrial.
  - **C3:** A **causal-teacher imitation** recipe yielding deployable, stall-free online behavior.
  - **C4:** A head-to-head benchmark vs. 6 ns-3 heuristics × 5 congestion levels × held-out videos →
    **best absolute QoE under congestion**, plus a practical serving path (µs-scale IPC bridge,
    sim-pause inference, buffer-aware guardrail).
  *Cite (positioning):* distinguish from [Wu'24 (NetLLM), He'24]; PEFT stack [Hu'22, Dettmers'23]; soft-token/reprogramming [Li&Liang'21, Lester'21, Jin'24].

---

## 3. Background & Motivation — MERGED into §2 (Introduction) [2026-09-07]

> No standalone Background section in the paper. The 3.1--3.4 content below is now a folded-in
> checklist (see the §2 note for where each item landed); keep it for reference only. The §II slot is
> taken by Related Work instead (see §8).

- **3.1 DASH & ABR primer.** Segments, ladder/rungs, the QoE trade-off. Define VMAF vs. bitrate
  (why perceptual quality, not raw bitrate).
  *Cite:* MPEG-DASH [Sodagar'11]; linear QoE [Mao'17, Yin'15]; VMAF [Li'16, Bampis'19].
- **3.2 The NTN setting.** Satellite-IAB architecture (1-paragraph recap + cite prior paper);
  RTT/variability/congestion; why this breaks throughput-prediction ABR.
  *Cite:* [Zahid'26]; NTN [3GPP TR 38.821]; measurement [Michel'22, Izhikevich'24]; stack modules [Mezzavilla'18, Polese'18]; sat-IAB [Abdullah'23].
- **3.3 Evaluating ABR: trace replay vs. full-stack closed loop.** The detailed version of the
  intro's second pillar. Trace-driven pipelines are insufficient here for two reasons: (i) *broken
  feedback* — a replayed throughput series is independent of the policy, so the self-congestion /
  bufferbloat / loss its own bitrate choices induce never feed back into future throughput; (ii)
  *no stack* — satellite RTT, the mmWave NR PHY/MAC + scheduler, IAB multi-hop backhaul, QUIC/BBR,
  handover, and beam-capacity sharing are stack-level phenomena absent from any throughput trace.
  ns-3 is packet-level and full-stack, so the policy is graded on the network it induces. Honest
  trade-off (Puffer/Fugu, NSDI'20 — learned ABR tuned in sim can underperform in situ): a simulator
  is less real than in-situ traces, but full-stack closed-loop ns-3 is *controllable*,
  *reproducible*, *sweepable* (the 100 Gb/s → 4.5 Mbps congestion ladder), and yields **per-rung
  ground-truth VMAF** that neither opportunistic traces nor live deployment can provide. Testbed
  inherited and validated in our prior paper [Zahid'26].
  *Cite:* trace-driven canon [Mao'17, Huang'19, Wu'24] on datasets [Akhtar'18] (+[FCC'16, Riiser'13]); sim-to-real [Yan'20]; testbed [Zahid'26].
- **3.4 Why a (small) language model.** Pretrained sequence priors; multimodal fusion via soft
  tokens; PEFT makes 1.5B deployable. Honest note: an LM over bare numbers is not automatically
  better — which motivates the *content* + *ladder* modalities as the justification (this is the
  `llm-architecture-analysis` reasoning compressed to a motivation paragraph). Why *small*: latency
  and deployability on the slow satellite path.
  *Cite:* PEFT [Hu'22, Dettmers'23]; soft prompts / reprogramming [Li&Liang'21, Lester'21, Jin'24]; backbone [Qwen'24]; the LM-over-bare-numbers caveat echoes [Wu'24, He'24].

---

## 4. System Design (the content-aware SLM policy)
- **4.1 Overview + closed loop.** *(Figure 1: system loop — ns-3 NTN sim ⇄ policy — combined with
  the architecture diagram to save space.)* Each segment: sim sends state → policy returns rung →
  sim downloads → next state.
- **4.2 Inputs / modalities.** network window `[W,7]`; content/semantic `[5]` (motion, saliency,
  object, quality-sensitivity, scene-cut); ladder `[6,2] = (size_bytes, VMAF)`. Feature table.
- **4.3 Multimodal encoder.** *(reuse the SLM-labeled `model_v2_compact` diagram.)* network 1D-CNN
  + semantic MLP + ladder attention-pool → fuse `Linear(224→256)`.
  *Cite:* — (our architecture; 1D-CNN + attention pooling are standard components).
- **4.4 Soft-token injection → frozen SLM + LoRA → policy head.** `inputs_embeds` bypass; k=4 soft
  tokens; per-token LayerNorm; mean-pool over the k tokens; encoder-residual skip; 6-rung head.
  *Cite:* LoRA [Hu'22]; QLoRA [Dettmers'23]; continuous/soft prompts [Li&Liang'21, Lester'21]; numeric→token reprogramming [Jin'24]; backbone [Qwen'24].
- **4.5 Serving & the guardrail.** IPC bridge (ns-3 binding under Py3.6 ↔ torch under Py3.11 via a
  stdin/stdout JSON inference server); sim pauses during inference (free w.r.t. QoE, only wall-clock);
  buffer-aware no-stall clamp (`bitrate ≤ throughput · max(buffer, seg_dur)/seg_dur · safety`).
  This is a *systems* contribution — NTN serving realism is an edge; lean into it.
  *Cite:* buffer-based safety-rail lineage [Huang'14, Spiteri'16]; QUIC/DASH testbed [Zahid'26].

---

## 5. Methodology: data, teacher, training
- **5.1 Data generation pipeline.** *(Figure/pipeline, or prose for space.)* ns-3 runs over
  videos × link-rate sweep {100 Gb/s, 36, 18, 9, 4.5 Mbps}; per-segment per-rung VMAF + size traces.
  The sweep deliberately spans uncongested → severe congestion.
  *Cite:* testbed [Zahid'26]; VMAF ground truth [Li'16].
- **5.2 The causal teacher (key methodological insight).** The teacher picks the highest rung the
  **present** throughput can sustain (with headroom) — present-only / causal, so the policy learns
  decisions it can reproduce online. Contrast: a *future-aware* target over-asks (bets it cannot
  reproduce) → online collapse/stalls. This is the paper's methods spine (from
  `acausal-oracle-misspecified-expert` / `causal-expert-fixes-online-collapse`), told entirely in
  "teacher" language — no "oracle".
  *Cite:* imitation-of-an-instant-quality-solver precedent [Huang'19 (Comyco)]; the causal-vs-acausal argument is ours.
- **5.3 Imitation training.** Class-weighted cross-entropy to the teacher rung; QLoRA 4-bit base;
  fixed feature scalers (train/serve parity); group-aware whole-video split (held-out test videos
  never seen in training).
  *Cite:* PEFT recipe [Hu'22, Dettmers'23].

---

## 6. Evaluation
- **6.1 Setup.** **4 policies (final): Ours + RAAHS, AAASH, BOLA.** 5 link-rate conditions × **10
  held-out videos** × **3 UEs**, window **W=4**, full closed-loop ns-3 NTN, identical scoring path for
  every policy (fairness). Backbone Qwen2.5-1.5B-Instruct; ~2.19M trainable (encoder+projection+head)
  + LoRA adapters; base frozen.
  *Cite (baselines):* RAAHS [Liu'11]; AAASH [Miller'12]; BOLA [Spiteri'16]. (FDASH/SFTM/SVAA/OSMF are
  not in the final benchmark — drop from related-work heuristic list.)
- **6.2 Metrics (absolute, oracle-free).** Absolute per-chunk VMAF-QoE
  `= w_vmaf·Σ(ω·VMAF) − w_stall·Σstall − w_switch·Σ|ΔVMAF↓|` (weights w_vmaf=0.65, w_stall=65,
  w_switch=0.8, down-only switch penalty). Plus mean VMAF ↑, stall time ↓, switch count ↓.
  **On first use, state the lineage explicitly:** this is the **linear QoE template introduced by
  Pensieve [Mao'17]** (which builds on MPC's `QoE_lin` [Yin'15]), with the **bitrate-utility term
  replaced by perceptual VMAF [Li'16, Bampis'19]** and a down-only smoothness penalty. Define once,
  report directly.
  *Cite:* QoE template [Mao'17, Yin'15]; perceptual substitution [Li'16, Bampis'19].
- **6.3 Main result.** ***Table 1 = the consolidated absolute per-chunk VMAF-QoE*** (overall per arm,
  `table2_absqoe.tex`): **Ours 9.55, RAAHS 7.42, BOLA 6.21, AAASH 1.61** → **Ours +28.7% over the best
  heuristic**. This overall number is the robust headline (per-rate absolute is heavy-tailed; that
  breakdown lives in Fig 2, not the table). Report mean VMAF / stall / switches alongside in prose.
- **6.4 Best under congestion.** *(Figure 2: absolute per-chunk VMAF-QoE vs. link rate, ascending.)*
  Honest claim (Puffer register): **Ours leads overall and at 4 of 5 rates, including the most severe
  (4.5 Mbps, +138%); the one exception is 9 Mbps, where BOLA edges ahead (5.83 vs 5.33, −8.6%).** State
  the exception explicitly. Per-arm by-rate values: Ours `13.4 / 12.0 / 11.4 / 5.3 / 5.6`;
  RAAHS `11.0 / 10.1 / 8.9 / 4.9 / 2.1`; BOLA `8.3 / 8.8 / 6.3 / 5.8 / 1.8`; AAASH `3.5 / −6.6 / 6.5 /
  2.4 / 2.4` (100Gb/s → 4.5Mbps). Note the mild non-monotonicity at 9→4.5 Mbps (stall heavy-tail).
- **6.5 Quality–rebuffering Pareto.** *(Figure 3: mean VMAF vs. rebuffering.)* On the frontier,
  ~zero stalls.
- **6.6 Content-awareness (prose).** Content-awareness = the **content-derived terms in the QoE
  objective and inputs**: the VMAF quality term `w_vmaf·ω·VMAF`, the scene-cut discount (0.65), and the
  semantic weight mixture `ω` (quality-sensitivity 0.5 / saliency 0.3 / object 0.2), plus the content
  branch + ladder VMAF the encoder consumes. Frame as an architectural pillar + how it enters the
  reward; report the effect honestly (no clean isolated-ablation causal claim).
- **6.7 Systems (measured, NVIDIA L40S, batch=1 per decision).** Per-decision forward latency
  **p50 25.6 ms / p95 26.0 ms** (p99 26.7, mean 25.7, min 25.3, max 30.2 — a very tight distribution).
  Model: **1.56 B params total = 1.54 B frozen + 20.7 M trainable** (encoder + soft-token projection +
  head + LoRA). **GPU memory ~3.2 GB weights, ~3.5 GB peak reserved.** The IPC leg is ~19 µs
  (negligible), so end-to-end decision latency is GPU-forward-bound (~26 ms + small feature/IPC
  overhead). Deployable on a single commodity GPU; the sim pauses during inference so latency costs
  only wall-clock, not QoE.

---

## 7. Discussion & Limitations
Sim-only (ns-3), no real operator traces; content channel's isolated gain modest (honest); imitation
ceiling (RL is future work); low UE count. Deployment path: distill the controller.
*Cite:* sim-to-real gap [Yan'20]; controller-distillation precedent [Hinton'15].

---

## 8. Related Work — NOW §II in the paper (moved up from the tail) [2026-09-07]

> Written and rendered as `\section{Related Work}` right after the Introduction in
> `latex-article/slm_abr.tex`: four grouped-citation clusters (classic+learned ABR; perceptual /
> content-aware ABR; LLM-for-networking + PEFT basis; NTN + IAB/mmWave + our prior work) plus the
> closed-loop positioning close. Original four-bucket blueprint retained below.

Bridge of **LLM-for-networking** + **perceptual ABR** into **NTN**, evaluated in a **closed loop** —
the intersection none of the prior art occupies. (Full verified list in §10.)

**Evaluation-methodology positioning (the second pillar in related-work form):** the learned-ABR
canon (Pensieve, Comyco, NetLLM, He'24) is validated by *trace-driven chunk-level replay* on public
throughput traces (FCC, Norway/HSDPA, Oboe) — terrestrial, and with the control loop opened. Puffer
(NSDI'20) is the counterpoint that in-situ behavior diverges from sim. We occupy the middle ground
none of them do: a *full-stack, closed-loop* evaluation on a satellite NR/IAB stack, where the
policy is scored on the network its own decisions produce.

---

## 9. Conclusion & Future Work
Recap the three pillars. Future: RL / control-aware training; **intent-steerable ABR** (the
`intent-guided-abr-research-framing` doc as the natural sequel); real NTN traces; multi-UE scale.

---

## 10. Literature review — verified references, mapped to sections

> **[2026-09-07] All references LINK-VERIFIED; encoded in `latex-article/slm_refs.bib` (canonical source
> for the paper's bibliography).** Confirmed arXiv IDs / DOIs: LoRA `2106.09685` (corrected from a bad
> `...714`), QLoRA `2305.14314`, Prefix `2101.00190`, Prompt `2104.08691`, Time-LLM `2310.01728`,
> Qwen2.5 `2412.15115`, NetLLM `2402.02338` (DOI 10.1145/3651890.3672268), He'24 `2404.01617`, TCP-LLM
> `2412.18200`, Zhou survey `2405.10825`, Izhikevich `2409.09846`, Mezzavilla `1705.02882`, Polese
> `1808.00376`, Abdullah `2304.01304` (PIMRC'23), Zahid `2604.16634`; Michel'22 IMC (DOI
> 10.1145/3517745.3561416), Fang'24 MM (DOI 10.1145/3664647.3680712), Sodagar (DOI 10.1109/MMUL.2011.71).
> Classic/perceptual/heuristic refs confirmed via their proceedings.
>
> **[2026-09-07] Trimmed to 25 references for magazine leanness (was 35).** Cut 10 single-use /
> redundant keys from Related Work grouped citations (clusters kept a citation, no rewording): BBA
> `huang2014bba`, Oboe `akhtar2018oboe`, Genet `xia2022genet`, GreenABR `turkkan2022greenabr`, Ye
> `ye2023vsabr`, VMAF-archival `bampis2019vmaf`, FCC traces `fcc2016`, Starlink-IMC `michel2022starlink`,
> TCP-LLM `shrestha2024tcpllm`, LLM-telecom-survey `zhou2024survey`. Entries remain in `slm_refs.bib`
> (uncited, do not render) in case one is added back.


Venues/years verified via web search (Sept 2026). Flags noted inline.

### Classic ABR  (cite §3.1, §6.1, §8)
- Huang et al., *"A Buffer-Based Approach to Rate Adaptation: Evidence from a Large Video Streaming
  Service"*, ACM SIGCOMM 2014 (BBA).
- Yin et al., *"A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP"*,
  ACM SIGCOMM 2015 (MPC/RobustMPC).
- Spiteri et al., *"BOLA: Near-Optimal Bitrate Adaptation for Online Videos"*, IEEE INFOCOM 2016.

### Learned / RL ABR  (cite §2, §8)
- Mao et al., *"Neural Adaptive Video Streaming with Pensieve"*, ACM SIGCOMM 2017.
- Akhtar et al., *"Oboe: Auto-tuning Video ABR Algorithms to Network Conditions"*, ACM SIGCOMM 2018.
- Xia et al., *"Genet: Automatic Curriculum Generation for Learning Adaptation in Networking"*,
  ACM SIGCOMM 2022.

### Perceptual / quality-aware ABR  (cite §3.1, §8)
- Huang et al., *"Comyco: Quality-Aware Adaptive Video Streaming via Imitation Learning"*,
  ACM MM 2019.  **[closest sibling — imitation + VMAF]**
- Yan et al., *"Learning in situ: a randomized experiment in video streaming"* (Puffer/Fugu),
  USENIX NSDI 2020.  **[skeptical-eval anchor]**
- Turkkan et al., *"GreenABR: Energy-Aware Adaptive Bitrate Streaming with DRL"*, ACM MMSys 2022.
- Ye et al., *"A Visual Sensitivity Aware ABR Algorithm for DASH via DRL"*, ACM TOMM 2023.
- VMAF: Li et al., *"Toward A Practical Perceptual Video Quality Metric"* (Netflix blog, 2016) **+**
  Bampis et al., *"SpatioTemporal Feature Integration and Model Fusion for FR VQA"*, IEEE TCSVT 2019
  (archival, peer-reviewed — cite both).

### LLM / SLM for networking  (cite §2, §3.3, §8)
- Wu et al., *"NetLLM: Adapting Large Language Models for Networking"*, ACM SIGCOMM 2024.
  **[primary precedent / baseline]**
- He et al., *"Designing Network Algorithms via Large Language Models"* (a.k.a. "LLM-ABR"),
  arXiv 2024.  **[distinguish our small-LM policy from this — do not reuse the "LLM-ABR" name]**
- Shrestha et al., *"Adapting LLMs for Improving TCP Fairness over WiFi"* (TCP-LLM), arXiv 2024.
- Jia et al., *"Beyond Interpretability: … Adaptive Video Streaming through LLMs"*, arXiv 2025.
- Zhou et al., *"LLM for Telecommunications: A Comprehensive Survey"*, arXiv 2024.

### PEFT / soft-token injection  (cite §4.4)
- Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022.
- Dettmers et al., *"QLoRA: Efficient Finetuning of Quantized LLMs"*, NeurIPS 2023.
- Qwen Team, *"Qwen2.5 Technical Report"*, arXiv 2024.
- Li & Liang, *"Prefix-Tuning: Optimizing Continuous Prompts for Generation"*, ACL 2021.
- Lester et al., *"The Power of Scale for Parameter-Efficient Prompt Tuning"*, EMNLP 2021.
- Jin et al., *"Time-LLM: Time Series Forecasting by Reprogramming LLMs"*, ICLR 2024.
  **[numeric→token-space analogue for §4.4]**

### NTN / satellite video streaming  (cite §2, §3.2)
- Michel et al., *"A First Look at Starlink Performance"*, ACM IMC 2022.
- Izhikevich et al., *"A Global Perspective on … Video Streaming over Starlink"*,
  ACM SIGMETRICS 2024.  **[Netflix collab — great for motivation]**
- Fang et al., *"Robust Live Streaming over LEO Satellite Constellations: … Handover-Aware
  Adaptation"*, ACM MM 2024.  **[closest NTN-ABR prior art]**
- 3GPP TR 38.821, *"Solutions for NR to support non-terrestrial networks (NTN)"*, Rel-16, 2023.

### 5G IAB / mmWave ns-3  (cite §3.2, §4.1)
- Mezzavilla et al., *"End-to-End Simulation of 5G mmWave Networks"*, IEEE COMST 2018.
- Polese et al., *"End-to-End Simulation of Integrated Access and Backhaul at mmWaves"*,
  IEEE CAMAD 2018.
- Abdullah et al., *"Integrated Access and Backhaul via Satellites"*, IEEE PIMRC 2023.
- **Zahid, Hossain, Hu**, *"End-to-End Performance of Video Streaming with MPEG-DASH over Satellite
  5G IAB Networks"*, arXiv 2026.  **[our prior work — the testbed]**

**Pre-submission flags:** BBA needs full subtitle; He'24 is arXiv-only; FDASH venue = IEEE Systems
Journal 2016 (there is no separate ns-3 DASH module paper — FDASH/Vergados is the citable one);
map RAAHS→Liu MMSys'11, SFTM→Liu SPIC'12, SVAA→Tian & Liu CoNEXT'12, AAASH→Miller PV'12,
OSMF→Mok QDASH MMSys'12; confirm AAASH DOI; 3GPP portal 403s automated fetch (spec verified via
mirrors); decide comfort with a few very-recent (2025–26) preprints.

### Additional references (verified, now added)
- **[Sodagar'11]** I. Sodagar, *"The MPEG-DASH Standard for Multimedia Streaming Over the Internet"*,
  IEEE MultiMedia 18(4):62–67, 2011 (DOI 10.1109/MMUL.2011.71). The DASH reference (formal spec:
  ISO/IEC 23009-1).
- **[Riiser'13]** H. Riiser, P. Vigmostad, C. Griwodz, P. Halvorsen, *"Commute Path Bandwidth Traces
  from 3G Networks: Analysis and Applications"*, ACM MMSys 2013, pp. 114–118. The Oslo/HSDPA 3G traces
  the trace-driven ABR canon replays.
- **[FCC'16]** Federal Communications Commission, *"Raw Data — Measuring Broadband America (MBA)"*,
  2016. The FCC broadband throughput traces used by Pensieve and successors.
- **[Gettys'12]** J. Gettys, K. Nichols, *"Bufferbloat: Dark Buffers in the Internet"*, Communications
  of the ACM 55(1):57–65, 2012 (also ACM Queue, 2011). Anchors the closed-loop feedback / self-
  congestion claim (§2 P4, §3.3).
- **[Hinton'15]** G. Hinton, O. Vinyals, J. Dean, *"Distilling the Knowledge in a Neural Network"*,
  arXiv:1503.02531, 2015 (NeurIPS DL Workshop 2014). The future-work controller-distillation reference.

> Magazine note: IEEE Network favors a lean reference list — group these into compact citation
> clusters (e.g., "throughput traces [FCC'16, Riiser'13]") rather than one sentence per reference.

---

## 11. Style / language exemplars (what to model the prose on)

1. **Our own first paper (Zahid et al., arXiv 2026)** — #1 anchor. Same authors, testbed vocabulary,
   QoE definitions. Reuse its NTN/IAB framing for continuity; reviewers of a sequel expect it.
2. **Pensieve (Mao, SIGCOMM'17)** — template for a learned-ABR intro + contributions list; crisp
   problem→gap→approach funnel.
3. **Puffer/Fugu (Yan, NSDI'20)** — model the *evaluation voice*: measured, skeptical, honest about
   when learning does *not* win. Right tone for the honest content-awareness finding.
4. **NetLLM (SIGCOMM'24)** — how to narrate an LLM-for-networking contribution (adaptation, PEFT,
   "why an LLM here"). Closest structural sibling for §3.3/§4.
5. **Comyco (MM'19)** — imitation-learning + perceptual-QoE framing language; methodological twin.
6. **BOLA / MPC** — crisp, formal problem-formalization sentences (the QoE objective, the decision).

**Prose conventions:** present tense for contributions ("We present…"); active voice; quantified
claims in the intro (the +28.7% number, "highest QoE at 9 and 4.5 Mbps"); a one-sentence
contributions bullet list; avoid hype adjectives ("novel", "powerful") — Puffer's restraint is the
model.

---

## 12. Writing style guide — language, tone, sentence structure

Grounded in three exemplars we actually pulled and read (verbatim quotes below). House voice = **our
first paper's measured, comparative tone**, borrowing **NetLLM's assertiveness for the method/system**
and **Puffer's restraint for the honest content-awareness finding**.

**What the exemplars actually do:**
- **Our first paper [Zahid'26]** — formal, understated, *comparative not absolute*; contributions are
  implicit (shown by what was built/measured, not announced as "novel"). E.g. *"Results show that no
  single configuration is uniformly optimal across all metrics."* Reuse this register for continuity.
- **NetLLM [Wu'24]** — assertive, strong action verbs, a historical-arc intro that ends on a motivating
  question, precisely quantified results. E.g. *"Despite their promising potential, existing
  learning-based algorithms still suffer from two key limitations…"* Use this shape for §2 P3–P4 + §4.
- **Puffer/Fugu [Yan'20]** — skeptical, honest, hedged. E.g. *"We found that more-sophisticated
  algorithms do not necessarily beat simpler, older algorithms."* and *"we do not fully understand
  it."* Use this exact restraint in §6.6 (content-awareness) and §7 (limitations).

**Concrete rules:**
- **Voice / tense:** first-person plural "we"; **present** for the system and contributions ("we
  present", "the encoder maps…"); **past** for experiments and related work ("we ran", "prior work
  showed").
- **Tone:** measured and comparative. Prefer *"best QoE under congestion among the evaluated policies"*
  over *"optimal"* / *"state-of-the-art-beating"*. Claims track the evidence.
- **Sentence structure (burstiness):** deliberately vary length. Follow a long multi-clause technical
  sentence with a short blunt one — Puffer does this for emphasis. ~12–30 words, but break the rhythm.
- **Paragraph shape:** vary it — not every paragraph is topic-sentence + three supports + wrap-up.
- **Contributions:** one bullet each, strong verb + what + why. No "novel", "powerful", "cutting-edge".
- **Results:** lead with the number and a comparator (*"+28.7% per-chunk VMAF-QoE over the best
  heuristic"*; *"holds at 9 Mbps where SVAA drops to 2.9%"*). Qualify honestly (*"with one exception…"*).
- **Transitions:** use contrast/causal connectives sparingly and meaningfully ("However", "In
  particular", "As a result") — not as reflexive paragraph-openers.
- **Terminology discipline:** "the teacher" (never "oracle"); "our content-aware SLM policy" / "Ours";
  "rung"; "perceptual QoE / VMAF". Define once, then reuse the *same* term (no elegant variation).
- **Citations (IEEE style):** refer to a prior work by its citation number, never by a vague
  descriptor. Do NOT write "a large in-situ study shows..." or "a recent work proposes..."; instead
  state the finding and cite it (e.g., "gains measured in simulation do not always survive real
  deployment \cite{...}"), or name the system/authors when that aids the argument (e.g., "NetLLM
  \cite{...}"). The bracket number carries the attribution.
- **Future work as bullets; do NOT self-discredit:** present future directions as a bulleted list,
  each a positive extension or opportunity, not an admission of weakness. Reframe a limit as the
  direction that addresses it ("reinforcement learning can carry the policy beyond imitation," not
  "the policy is capped at the teacher"). Keep any genuinely necessary caveat brief and place it in
  the Introduction (e.g., the honest sim-vs-deployment hedge), not as a self-deprecating limitations
  dump in the closing section.

| Do | Don't |
|---|---|
| "The policy chooses the highest rung the present throughput sustains." | "The policy leverages a sophisticated mechanism to intelligently select bitrates." |
| "We do not isolate a robust content-channel gain; we report the null." | "Our novel content-aware approach significantly improves performance." |
| "Ours holds 18.6% at 9 Mbps; SVAA drops to 2.9%." | "Ours demonstrates superior performance across the board." |

---

## 13. Anti-AI-language pass + detector check (do before submission)

**Framing (read first).** AI-text detectors (GPTZero, Originality.ai, Turnitin) are **unreliable** —
they false-positive on formal academic prose and prove nothing on their own. The real goal is **prose
that reads as specific, human, technical writing**; passing a detector is a *side effect* of that, not
the target. Separately: **IEEE/ACM require disclosing generative-AI assistance** — follow the venue
policy honestly. This section removes machine-like *style*; it is not about hiding authorship.

**Register (match the existing text).** Formal, plain, declarative, measured claims. **No
contractions.** Define acronyms at first use. Keep spelling and terminology consistent with the current
draft (no elegant variation once a term is chosen).

**Banned words / phrases (classic LLM tells — grep and kill):** delve, leverage (as filler), realm,
tapestry, landscape, showcase, underscore(s), pivotal, "crucial role", "plays a (significant/vital)
role", "testament to", "it is worth noting", "it is important to note", "in today's world",
"ever-evolving", "seamless(ly)", "a myriad of", "navigating the complexities", "unlock", "harness the
power", "at the forefront", "In conclusion", "Overall,", "In summary,", reflexive
"Moreover/Furthermore/Additionally" as paragraph openers, "not only … but also" spam, "aims to",
"can potentially".

**Structural tells (reviewers notice these even when the words are clean):**
- **Low burstiness** — every sentence 15–22 words. Fix: add short punches; one long complex sentence
  per paragraph.
- **Rule-of-three overuse** — reflexive tricolons ("fast, robust, and scalable"). Cut to one or two,
  make them concrete.
- **Symmetric scaffolding** — "Firstly/Secondly/Thirdly", every section ending in a summary sentence,
  every paragraph the same shape. Break the symmetry.
- **Hollow topic sentences** — "Video streaming is an important area of research." Delete; open with a
  claim that carries information.
- **Over-signposting** — "In this section, we will discuss…". Just discuss it.
- **Empty hedging** — "This could potentially help in some cases." Commit or cut.
- **Em dashes (—)** — one of the strongest LLM tells; **skip them entirely** in the paper prose.
  Replace each with a comma, parentheses, or two separate sentences. This applies to the
  spaced " — " construction too. (This planning doc uses them freely; the *paper* must not.) Keep
  en dashes only for numeric ranges (e.g. 5–10°, 4–7 min).
- **Mid-sentence colons (:)** — also a strong LLM tell; **do not use a colon inside a sentence** to
  introduce a list or an elaboration. Split into two sentences, or use a comma. Colons are acceptable
  only for structural labels such as figure/table captions, not in running prose.
- **Parenthetical overuse** — do not stack explanatory information inside brackets when it can be
  written as a plain clause or sentence. Acronym definitions on first use, e.g. "adaptive bitrate
  (ABR)", are fine; a "(...)" aside that carries a real explanation should be folded into the sentence.
  Rule of thumb: at most one short parenthetical per sentence, and none that runs longer than a few words.
- **Long figure/table captions** — keep captions to a short phrase; the explanation belongs in the body
  text, not the caption. If a caption reads like a sentence explaining the figure, move that into prose.

**Positive human signals to add:**
- **Specificity** — exact numbers, condition names, algorithm/file names, one concrete example per
  claim. Machines generalize; specifics read human.
- **Judgment / voice** — occasional first-person assessment ("we were surprised that…", "we do not
  fully understand this", "with one exception") — Puffer's signature move.
- **Asymmetry** — uneven paragraph lengths; a one-sentence paragraph where it earns emphasis.
- **Citations woven into the argument**, not name-dropped in a clump.

**Procedure (checklist):**
1. Draft, then **read it aloud** — anything you'd never say, rewrite.
2. **grep the banned list** on the `.tex` (command below); replace each hit with concrete phrasing.
3. **Sentence-length variance check** — skim for runs of same-length sentences; inject burstiness.
4. **Every claim carries a number or a citation** — no free-floating adjectives.
5. **Detector smoke test** — run 1–2 detectors *only* as a nudge; a high "AI" score means "add
   specificity / vary rhythm", not "reword to fool it".
6. **Human read** — a co-author's eye beats any detector.

**grep one-liner** (run on the draft `.tex`, case-insensitive):
```bash
grep -inE 'delve|leverag|realm|tapestry|landscape|showcas|underscore|pivotal|crucial role|testament to|it is (worth|important) (noting|to note)|seamless|myriad|navigating the|in conclusion|in summary|ever-evolving|not only .* but also|aims to|can potentially|moreover|furthermore' paper.tex
# em dashes (LaTeX "---" or the literal — character), and mid-sentence colons, must be rewritten:
grep -nE '---|—' paper.tex
grep -nE '[[:alnum:])]:[[:space:]]+[a-z]' paper.tex   # mid-sentence colons (word: word); the address "Emails:" is fine
grep -noE '\([^)]{35,}\)' paper.tex                   # long explanatory parentheticals to fold into prose
grep -nE '\\caption\{[^}]{55,}\}' paper.tex           # captions over ~55 chars = probably too long
```

---

## 14. Open items before drafting prose
- [x] Metric = **absolute per-chunk VMAF-QoE only** (no % of teacher anywhere).
- [x] Benchmark is **final/frozen**.
- [x] Final baseline set = **4 arms** (Ours, RAAHS, AAASH, BOLA).
- [x] Held-out videos = **10**; UEs = **3**; window **W=4**; reward weights confirmed; backbone Qwen2.5-1.5B.
- [x] Table 1 = consolidated absolute QoE (`table2_absqoe`); Fig 2 = absolute QoE vs link rate (regenerated).
- [x] Venue = **IEEE Network magazine**, authors as first paper, **no AI disclosure** required.
- [x] **System values measured (L40S):** latency p50 25.6 / p95 26.0 ms; 1.56 B params (1.54 B frozen
      + 20.7 M trainable incl. LoRA); ~3.5 GB peak GPU memory. Benchmark: `~/bench_latency.py`.
- [x] **4 floats LOCKED:** (F1) system + architecture figure; (F2) Table 1 = consolidated absolute QoE
      (`table2_absqoe`); (F3) Fig 2 = absolute QoE vs link rate (congestion); (F4) Fig 1 = quality–
      rebuffering Pareto. (`fig3_headroom`, `fig5_rung_dist` = backup / extended version only.)
- [x] §6.6 content-awareness = the content terms in the QoE equation (mentioned; see §6.6).
- [x] Six additional references added + `[add:]` markers resolved (§10 "Additional references").
- [ ] Run the anti-AI-language pass (§13) on every drafted section before submission.
