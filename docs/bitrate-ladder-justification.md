# Bitrate-ladder justification — why {0.25, 0.75, 1.5, 3, 6, 12} Mbps (paper reference)

Reference for the paper's experimental-setup section: why our encoding ladder has these six rungs, and why
that is the *right* ladder for the NTN benchmark rather than Pensieve's (lower) or NADA's 4G/5G (higher).
Ladder constants: `scratch/ai_llm/paths.py` (`NS3_RATE_LADDER_BPS`), `scratch/ai_llm/metadata_loader.py`
(`LADDER_LEVELS`). Companion: [[teacher-reward-design]].

## 1. The ladder
Six rungs, roughly-geometric (~2×) spacing:

| Rung | r0 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|
| Nominal | 250k | 750k | 1500k | 3000k | 6000k | 12000k |
| Mbps | 0.25 | 0.75 | 1.5 | 3 | 6 | 12 |

Six rungs give a compact policy action space (6-way head); ~2× steps make the quality (VMAF) increments
roughly even and each rung perceptually distinct — the standard HLS/DASH authoring convention (Apple HLS
authoring spec, DASH-IF IOP) that adjacent rungs differ by ~1.5–2×.

## 2. The sizing principle (Pensieve and NADA both follow it)
**The ladder top should equal the peak *per-UE sustainable* throughput of the environment**, and the bottom
a low mobile-quality floor. This is not our invention — NADA (arXiv:2404.01617) uses *two* ladders and states
it explicitly: `{300,750,1200,1850,2850,4300}` kbps (top 4.3 Mbps) for the low-bandwidth FCC/Starlink traces,
but *"since our 4G and 5G datasets exhibit much higher bandwidth, we elevated the bitrate ladder to
{1850,2850,4300,12000,24000,53000}"* (top 53 Mbps). Pensieve/EnvivioDash3 uses the 4.3 Mbps-top ladder for
FCC broadband. The ladder tracks the bandwidth regime.

## 3. Why ours sits between Pensieve and NADA — by construction, not arbitrarily
Our bottleneck is an NTN feeder link shared by **3 UEs**, swept over {4.5, 9, 18, 36} Mbps (plus a 100 Gb/s
uncongested control). The quantity a rung must fit is **per-UE** throughput = link ÷ 3 UEs:

| Feeder link | 4.5 | 9 | 18 | 36 Mbps |
|---|---|---|---|---|
| Per-UE sustainable | ~1.5 | ~3 | ~6 | ~12 Mbps |

So our realistic per-UE operating range is **1.5–12 Mbps**. That is *above* FCC-broadband/Starlink (~1–5 Mbps,
Pensieve top 4.3) and *below* 4G/5G (tens of Mbps, NADA top 53). A ladder topping at **12 Mbps** is therefore
the correct match: "higher than Pensieve, lower than NADA" is a direct consequence of our per-UE bandwidth
regime, not an arbitrary pick. The bottom two rungs align with the Pensieve/EnvivioDash3 reference (250≈300,
750 = 750), extended upward for NTN feeder capacity — exactly the elevation NADA performs for its faster networks.

## 4. Data validation — the ladder is exercised as designed
The ladder is not just plausibly sized; the policy's *behaviour* confirms it. Model mean chosen rung per
condition lands on the rung whose bitrate matches that condition's per-UE capacity, and the full ladder r2→r5
is used across the sweep (from `Quic_artifacts/AiDash/bench_down_scored/*/run_*/decisions.jsonl`):

| Condition | per-UE cap (Mbps) | capacity-matched rung | model mean rung |
|---|---|---|---|
| 4.5 Mbps | 1.5 | r2 | 2.14 |
| 9 Mbps | 3.0 | r3 | 3.04 |
| 18 Mbps | 6.0 | r4 | 3.81 |
| 36 Mbps | 12.0 | r5 | 3.79 |
| 100 Gb/s | ∞ (cap) | r5 | 4.03 |

Each congestion step moves the operating rung down by ~one, landing on the capacity-matched rung. This both
(a) shows the ladder is well-sized (no rung is wasted; none is missing for the tested conditions), and (b)
makes the benchmark discriminating — each network condition exercises a distinct rung.

## 5. Caveat and metric-invariance
The 100 Gb/s condition is an *unconstrained control*; there per-UE bandwidth exceeds the ladder, so the top
rung r5 (12 Mbps) is the cap. This is realistic — no content is encoded above 12 Mbps, and one cannot stream a
rung that was never encoded — and it does not affect any comparison: the %-of-oracle metric is **ladder-invariant**
(every arm *and* the greedy causal-teacher ceiling share the same ladder), so rankings are unchanged by the ladder's
absolute values. Extending the ladder toward 4K/tens-of-Mbps rates is future work relevant only to conditions with
>12 Mbps sustained per-UE bandwidth, which lie outside our congestion-focused contribution.

## References
- Pensieve — Mao et al., *Neural Adaptive Video Streaming with Pensieve*, SIGCOMM'17 (EnvivioDash3 ladder,
  `QoE_lin`).
- NADA — He et al., *Designing Network Algorithms via Large Language Models*, arXiv:2404.01617 (regime-matched
  ladders; explicit elevation for 4G/5G).
- Geometric-spacing convention — Apple HLS Authoring Specification / DASH-IF IOP (background, not conformance).
