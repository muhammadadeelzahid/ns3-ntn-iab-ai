# Intent-Guided ABR — research framing & literature analysis

Companion to [`llm-architecture-analysis.md`](llm-architecture-analysis.md). That doc analyzed
the *inline* LLM (LLM sits in the per-decision loop, consumes numeric state). This doc analyzes
a different and stronger proposal:

```
User feedback / Expert command / language-guided intent   (LANGUAGE)
        │
        ▼
   SLM / LLM interpreter          ← slow path: per-session or on-command, NOT per-segment
        │   emits a conditioning signal (QoE-weight vector / goal embedding / constraints)
        ▼
[ + network features  + semantic features ]  →  RNN / CNN / MLP controller  →  rung
                                                 ↑ fast path: per-segment, per-UE
```

**One-line thesis:** the LLM earns its place by interpreting *language* (operator intent,
expert command, human feedback) into a conditioning signal for a fast neural ABR controller —
**not** by regressing 24 numbers. This turns the LLM from a weakly-justified inline regressor
into a *steerability* mechanism that numeric ABR structurally cannot provide.

---

## 1. Where Tier 0 / Tier 1 actually stand (grounded in the code)

| Tier | What it is in `model.py` | Status | Justification |
|------|--------------------------|--------|---------------|
| **0** | `MultimodalEncoder` (network 1D-CNN over W=8 window + semantic MLP + ladder proj → fuse) → `AbrPolicyHead` (MLP→6 logits) | **Built, trained** (0.508 majority-class run, 2026-06-17) | Strong, fast baseline. ~0.1–1 M params, sub-ms. |
| **1** | Same sandwich, `backbone_projector` replaced by frozen Qwen/Llama + LoRA | **Not built** — `backbone_projector` is a 3-layer MLP placeholder; `DEFAULT_MODEL_ID` never loaded | **Weak on input grounds.** An LLM over 24 normalized scalars has no advantage over the MLP — stated in `llm-architecture-analysis.md` §2–3 and confirmed by NetLLM/TCP-LLM never ablating against a no-LLM encoder. |

The unresolved question in the prior doc was: *if the LLM needs language to be justified, where does
the language come from?* The 5 semantic scalars are not language. **This proposal answers it: the
language is operator/human intent, and the LLM is moved out of the per-decision loop.**

This also resolves a hard constraint specific to this project: GEO RTT ≈ 600 ms and a 1.5-RTT QUIC
handshake (see QUIC handshake analysis) make a per-segment 7B forward pass infeasible anyway. The LLM
*must* live on a slow path. Intent interpretation is exactly that slow path.

---

## 2. Literature analysis — four pillars and the white space

### Pillar 1 — ABR control (the numeric baselines)
BBA buffer-based (Huang, SIGCOMM'14); MPC / RobustMPC (Yin, SIGCOMM'15); BOLA (Spiteri, INFOCOM'16);
Pensieve, RL (Mao, SIGCOMM'17); Oboe auto-tuning (Akhtar, SIGCOMM'18); Comyco imitation (Huang, MM'19);
Fugu/Puffer real-world (Yan, NSDI'20); Genet curriculum (Xia, SIGCOMM'22).
**Gap:** every one optimizes a *fixed* QoE objective baked in at training time. None is steerable by
language at run time; changing the operating point means retraining or hand-tuning weights.

### Pillar 2 — LLM-for-networking (the closest prior art)
NetLLM (Wu et al., SIGCOMM'24 — in `reference_codes/NetLLM/`): one LLM adapted to ABR / viewport /
cluster scheduling. TCP-LLM (Shrestha et al.): LLM for congestion-control / CCA selection
(see `llm-architecture-analysis.md` §11).
**Gap that matters for you:** both put the LLM *inline over numeric state* and **neither ablates
against a no-LLM encoder baseline**, so neither isolates what the LLM adds. Your Tier 0 vs Tier 1
ablation already goes beyond them; your intent reframing sidesteps their weakest point.

### Pillar 3 — language-conditioned control / language-to-reward (the ML backbone of the idea)
Survey: Luketina et al., "RL Informed by Natural Language" (IJCAI'19); grounded language learning
(Hermann et al., 2017). Most directly:
- **Language-to-Rewards** (Yu et al., CoRL'23): an LLM translates a language instruction into
  reward-function *parameters* that a low-level controller then optimizes.
- **Eureka** (Ma et al., ICLR'24): LLM writes reward code for a low-level policy.
- Robotics analogues with the *exact* slow-LLM/fast-policy decoupling: SayCan (Ahn et al., 2022),
  PaLM-E (Driess et al., 2023), RT-2 (Brohan et al., 2023).
**This is the literature that legitimizes the architecture — and no ABR paper imports it.**

### Pillar 4 — Intent-Based Networking + human-in-the-loop
"Like a net manager" = **Intent-Based Networking** (IETF RFC 9315; TMForum Autonomous Networks):
operators declare high-level intent, the system compiles it to control. LLM-driven intent translation
for 5G/6G management is active (e.g. Nokia Bell Labs' *Hermes* line, 2024; a wave of LLM-for-6G-management
surveys). Human-feedback shaping is RLHF (Christiano et al., NeurIPS'17; Ouyang et al./InstructGPT, 2022).
**Gap:** IBN/LLM-intent work targets network *configuration/orchestration*; nobody has pushed intent
into the *per-segment ABR control loop*, and nobody has done it for *NTN/satellite*, where operator
policy (scarce shared beam capacity, data caps, priority/emergency traffic) is most consequential.

### The white space
**Intersection of Pillars 3+4 applied to Pillar 1, in the NTN/satellite setting:**
*natural-language-steerable ABR for satellite video, with the LLM on the slow intent path and a fast
neural controller on the per-segment path.* This is novel, defensible, and unoccupied.

---

## 3. What makes the best paper (decision)

| Option | Verdict |
|--------|---------|
| **Tier 0 vs Tier 1 only** | **Weak / likely negative.** Your own doc predicts Tier 1 ≈ Tier 0 over numbers; NetLLM/TCP-LLM already hold this ground. Don't make it the paper. |
| **Tier 0 vs Tier 1 vs intent-model** | **Correct experimental design, wrong framing.** Run all three — but Tier 0 = fast numeric baseline, Tier 1 = *negative control*, not co-equal contributions. |
| **Intent-guided system is the paper** | **Recommended.** Contribution = steerable ABR via natural-language intent (LLM slow-path) + fast neural controller (per-segment), demonstrated in NTN. Tier 0/1 become baseline + negative control *inside* it. |

**Why the negative control is a feature, not a detour:** showing inline-LLM-over-numbers ≈ encoder
(Tier 1 ≈ Tier 0) is the experiment that *proves the LLM belongs on the language path*. It is what makes
a reviewer trust you are not LLM-hyping. Frame it as: "we first show the obvious inline use of an LLM
does not pay off, which motivates relocating the LLM to the intent path."

**Headline:** *Inline LLMs over numeric ABR state don't beat a small encoder; the LLM's real value is
interpreting operator/human intent expressed in language to steer a fast controller — a capability
numeric ABR structurally lacks. We demonstrate intent-satisfying, steerable ABR for satellite video.*

---

## 4. Contribution claims & research questions

**Claims (C):**
- **C1 (capability):** a single trained controller satisfies multiple, *compound*, and *previously
  unseen* operator intents (latency-first / quality-first / data-saver / fairness-first / "be
  conservative during the storm") with **no retraining**.
- **C2 (honest ablation):** inline LLM over numeric state (Tier 1) does **not** beat the encoder
  (Tier 0); the gain comes from the *language/intent path*, not from the LLM seeing numbers.
- **C3 (NTN relevance):** in high-RTT, capacity-shared NTN, intent steering changes the QoE
  operating point in operator-meaningful ways that a fixed-objective ABR cannot reach.
- **C4 (HITL):** expert corrections / feedback expressed in language measurably move the policy
  toward the intended operating point (an interactive, not just declarative, loop).

**Research questions (RQ):**
- **RQ1:** Does language-conditioning let one controller span the QoE trade-off frontier that
  otherwise needs N separately-trained policies?
- **RQ2:** Does the LLM beat a *non-LLM intent parser* (regex / keyword lookup → weight vector) on
  **compound and held-out** intents? *(This is the experiment that defeats "it's just a knob" — §6.)*
- **RQ3:** How much of any end-to-end gain is the LLM vs the conditioning mechanism vs the controller?
  (Tier 0, Tier 1, intent-model + ablations isolate this.)
- **RQ4:** Latency/cost: the slow-path LLM amortized over a session vs per-segment inline — quantify.

---

## 5. Proposed architecture (mapped onto the existing code)

Reuse the Tier-0 stack almost verbatim; add an intent path and a conditioning mechanism.

```
Operator/expert text ──► SLM/LLM interpreter ──► intent embedding  z_intent  (cached per session)
                                                      │
network window [B,W,9] ─► network 1D-CNN ─┐           │  FiLM / concat / cross-attn
semantic [B,5]         ─► semantic MLP    ├─► fuse ─► CONDITION on z_intent ─► policy head ─► 6 logits
ladder [B,6,2]         ─► ladder proj     ┘
```

- **Interpreter:** small instruct model (SLM, 0.5–3B; that's plenty for intent→vector) or even a
  frozen-encoder + projection. Output options, in increasing ambition:
  1. **QoE-weight vector** `(w_vmaf, w_stall, w_switch, w_rung, ...)` — interpretable, ties straight
     into your existing oracle reward (`R_t = w_vmaf·sem·VMAF − w_stall·stall − switch − w_rung·rung`).
     *Start here — it is the cleanest and most defensible.*
  2. **Dense goal embedding** `z_intent ∈ R^d` injected via **FiLM** (feature-wise modulation) on the
     fused state — more expressive, less interpretable.
  3. **Constraint set** (hard caps: max rung for prepaid, min buffer for VIP) applied as an action mask.
- **Conditioning point:** inject `z_intent` at the `fuse` step in `MultimodalEncoder.forward` (concat
  or FiLM). The 1D-CNN / semantic / ladder branches and `AbrPolicyHead` are unchanged → Tier 0 stays a
  strict sub-network (set `z_intent = 0` → exactly today's model). Clean ablation, minimal new code.
- **Controller core:** keep the CNN+MLP. The "RNN" option (GRU over the segment history) is a fair
  variant to report, but the W=8 1D-CNN already captures the temporal window; an RNN ablation is a
  nice-to-have, not the contribution.
- **Training:** condition the oracle on the intent weights. For each training trajectory, *sample an
  intent*, recompute the DP-oracle label under that intent's weights, and train the controller to
  imitate the intent-conditioned oracle. This is the key data move — it teaches the controller to
  *obey* the conditioning. (Generate paraphrases of each intent with an LLM so the interpreter sees
  language variety, not 5 fixed strings.)

---

## 6. Evaluation design (this is where the paper is won)

**Baselines:** FDASH (C++ guardrail), Tier 0 (fixed-objective encoder), Tier 1 (inline LLM over
numbers, the negative control), and per-intent *separately-trained* Tier-0 policies (the strong
"specialist" upper bound the single conditioned model is trying to match).

**Novel metric — Intent Satisfaction.** Top-1 accuracy is the wrong axis (you already hit the
majority-class trap at 0.508). Define, per intent, a target region in QoE space and measure:
- **Operating-point shift:** does "latency-first" actually lower stall/rebuffer and "quality-first"
  raise mean VMAF, relative to neutral? Report the *direction and magnitude* per intent.
- **Frontier coverage:** plot QoE trade-off (mean-VMAF vs stall vs switches). One conditioned model
  should trace a *frontier*; show it approaches the envelope of the N specialist models (C1/RQ1).
- **Constraint adherence:** for hard intents (data cap, max rung), measure violation rate.

**The "it's just a knob" killer experiment (RQ2).** Compare the LLM interpreter against a
regex/keyword→weight lookup table on:
- **compound intents** ("prioritize live sports for premium users but cap data for prepaid"),
- **held-out intents** (phrasings/goals not in training),
- **context-grounded intents** ("be conservative during the storm" → requires mapping world knowledge
  to caution on buffer).
If the LLM beats the lookup table *here* but ties it on simple seen intents, you have proven the LLM
is doing real interpretation, not table lookup. **This single result is the spine of the paper.**

**HITL experiment (C4/RQ4).** Seed a neutral policy; issue an expert correction in language
("you're switching too much"); show the operating point moves toward fewer switches. Optionally fold
corrections into preference data (RLHF-style) and show further improvement.

**NTN-specific scenarios (your domain edge).** GEO/LEO RTT regimes, beam-capacity contention across
UEs, handover events, operator policies (emergency-traffic priority, prepaid data caps). Show intent
steering matters *most* exactly where capacity is scarce and operator policy is explicit — the
satellite setting — and tie back to the 600 ms RTT / 1.5-RTT handshake making a slow-path LLM the only
viable placement.

**Scale:** 1-UE then 10+-UE (your Phase-5 plan), report inference latency p50/p95 and confirm the
intent LLM cost is amortized per session, not paid per segment.

---

## 7. Threats to validity / reviewer objections (preempt them)

| Objection | Rebuttal / experiment |
|-----------|----------------------|
| "It's just a QoE-weight knob with an LLM bolted on." | RQ2 killer experiment: LLM beats a regex/lookup parser on **compound, held-out, context-grounded** intents. A knob can't compose or generalize. |
| "Why not train N specialist policies?" | C1: one conditioned model approaches the N-specialist frontier (frontier-coverage plot) *and* handles unseen/compound intents the N specialists were never trained for. Operationally, one model << N models to deploy/update. |
| "The LLM doesn't help (Tier 1 ≈ Tier 0)." | That's **your own result**, presented deliberately (C2) to motivate moving the LLM to the language path. Turn the weakness into the argument. |
| "Synthetic intents / no real operators." | Use LLM-paraphrased intent corpora + a small human-written held-out set; frame HITL as the realistic-feedback channel. Be explicit it's a simulation study (it's ns-3). |
| "Numeric precision / hallucination if LLM emits actions." | The LLM never emits the action — it emits a conditioning vector; the fast controller (and FDASH fallback) emits the rung. Same single-forward-pass guardrail you already have. |

---

## 8. Suggested paper skeleton

1. **Intro** — ABR optimizes a fixed objective; operators have *changing, language-expressed* intent;
   satellite makes this acute (scarce shared capacity, explicit policy, high RTT).
2. **Motivation / negative result** — inline LLM over numbers ≈ encoder (Tier 1 vs Tier 0). Motivates
   relocating the LLM.
3. **Design** — slow LLM intent interpreter → conditioning → fast controller; intent-conditioned
   oracle for supervision.
4. **Implementation** — ns-3 / NTN / QUIC / DASH; the encoder + conditioning; the FDASH guardrail.
5. **Evaluation** — intent satisfaction, frontier coverage, RQ2 killer experiment, HITL, NTN scenarios,
   scale + latency.
6. **Related work** — the four pillars (§2); position as the bridge of Pillars 3+4 into ABR for NTN.
7. **Discussion / limitations** — sim-only, intent corpus realism, deployment path (distill controller).

---

## 9. References (real, grouped; verify exact venue/year before submission)

**ABR:** Huang BBA (SIGCOMM'14); Yin MPC/RobustMPC (SIGCOMM'15); Spiteri BOLA (INFOCOM'16);
Mao Pensieve (SIGCOMM'17); Akhtar Oboe (SIGCOMM'18); Huang Comyco (ACM MM'19); Yan Fugu/Puffer
(NSDI'20); Xia Genet (SIGCOMM'22).
**LLM-for-networking:** Wu NetLLM (SIGCOMM'24); Shrestha TCP-LLM (arXiv 2412.18200).
**Language-conditioned control / language-to-reward:** Luketina RL-informed-by-language survey
(IJCAI'19); Hermann grounded language learning (2017); Yu Language-to-Rewards (CoRL'23);
Ma Eureka (ICLR'24); Ahn SayCan (2022); Driess PaLM-E (2023); Brohan RT-2 (2023).
**Intent-Based Networking / HITL:** IETF RFC 9315 (IBN concepts & terminology); TMForum Autonomous
Networks; Nokia Bell Labs *Hermes* LLM-for-network-management (2024); Christiano RLHF (NeurIPS'17);
Ouyang InstructGPT/RLHF (2022).
**NTN/satellite streaming (context):** recent Starlink/LEO measurement & streaming studies (e.g. IMC'24)
— add the specific ones you cite in the NTN sections.

> Note: venue/year are from memory (knowledge cutoff Aug 2025) — re-verify each before submission, and
> replace the Pillar-4 LLM-intent and NTN citations with the exact papers you build the related-work on.
