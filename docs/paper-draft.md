# Content-Aware Adaptive Bitrate Streaming with a Small Language Model over Satellite 5G IAB Networks

Draft prose. Voice/style per `docs/paper-outline.md` (§12): measured and comparative, first-person
plural, quantified claims, no em dashes, no AI-tell vocabulary. Citation keys resolve to §10 of the
outline.

## 1. Introduction

Video is the largest share of Internet traffic, and almost all of it is delivered by adaptive bitrate
(ABR) streaming. The video is cut into short segments, each encoded at several bitrates, and for every
segment an ABR algorithm on the client picks one rung of that bitrate ladder. The choice sets the
quality of experience (QoE): a higher rung looks better, but if the network cannot deliver it in time
the playback buffer drains and the video stalls, and frequent rung changes are themselves distracting.
Quality, rebuffering, and switching pull against each other, and the algorithm has to trade them off
one segment at a time using only what it has measured so far.

This trade-off gets much harder over a satellite link. Non-terrestrial networks (NTN), where a 5G
access network is backhauled through a low- or geostationary-orbit satellite, are moving from proposal
to deployment, and they carry video like any other access network. The satellite path has long and
variable round-trip times, congestion on a shared feeder link, and throughput that can swing sharply
when the serving beam or satellite changes. Recent Starlink measurements report exactly these swings,
together with the bitrate churn and rebuffering they cause [Izhikevich'24]. Our earlier work built a
full ns-3 model of this setting and showed that even the transport choice matters, with QUIC over BBR
giving the most stable delivery [Zahid'26]. ABR runs on top of that transport, and it is stressed most
in the regime where users are least willing to tolerate a stall.

Two habits in how learned ABR is built and tested make it a poor fit for this regime. The first is
what the policy optimizes. Most ABR, including the recent learned and language-model systems, still
targets bitrate, which is a coarse proxy for what a viewer sees; two segments at the same bitrate can
differ by tens of VMAF points depending on their content [Mao'17, Wu'24]. The second is how the policy
is evaluated. The learned-ABR literature is validated by replaying recorded throughput traces on a
terrestrial abstraction of the network [Mao'17, Huang'19]. That protocol is convenient, but it hides
the satellite stack, and it breaks the feedback loop that ABR lives inside.

ABR is a control loop, not a prediction task. The rung a client picks decides how many bytes it pushes
into the network, which reshapes queueing, self-induced congestion, round-trip time, and loss, and so
changes the throughput the client will see next. A replayed trace is fixed in advance, so it cannot
react to the policy's own decisions: the policy is graded on a network it cannot affect, and the
satellite radio, the integrated access and backhaul (IAB) hops, the transport, and the handovers are
all absent. We take the other route. We evaluate every policy inside a full closed-loop ns-3 NTN
simulation, so each decision is scored on the network that decision helps create, over a real mmWave
and IAB stack. A simulator is not as faithful as a live deployment, and we do not claim it is, but it
is far closer than trace replay while staying controllable and reproducible. We reuse the validated
testbed from our prior work [Zahid'26] so the environment itself is not a new variable.

Inside that loop we place a content-aware policy built on a small language model. A frozen Qwen2.5-1.5B
model with LoRA adapters sits at the core, fed not by text but by the live state of each segment: a
short window of network measurements, per-segment content features (motion, saliency, scene cuts, and
per-rung VMAF), and the bitrate ladder itself, all projected into the model's embedding space as soft
tokens. We train it by imitation. The target is not a future-aware optimum but a causal teacher that
picks the highest rung the current throughput can sustain, so the policy learns decisions it can
reproduce when it runs online. We call the model a small language model on purpose: at 1.5 billion
parameters it decides a rung in about 26 ms on a single GPU, and only 21 million of its parameters are
trained.

This paper makes four contributions.

- We present a content-aware ABR policy that reasons over network, content, and ladder state with a
  small, mostly frozen language model, and that optimizes perceptual (VMAF) quality rather than bitrate.
- We evaluate it, and every baseline, inside a full closed-loop ns-3 NTN simulation, so the policies
  are scored on the network their own decisions produce rather than on a fixed recorded trace.
- We show that imitating a causal, throughput-sustainable teacher, rather than a future-aware one, is
  what keeps the policy stall-free once it runs online.
- Across five congestion levels and ten held-out videos, our policy delivers the best absolute
  perceptual QoE overall, 9.55 against 7.42 for the strongest heuristic, a gain of 28.7%, and it stays
  ahead through the deepest congestion. We report the one operating point where a buffer-based
  heuristic still edges it rather than leave it out.
