"""Pensieve (Mao et al., SIGCOMM 2017) RL ABR baseline, retrained on our ns-3 traces.

Network-only, canonical linear QoE reward, trained from scratch (random init) with a MINIMAL,
budget-capped synchronous A2C loop so it never out-trains our LLM policy (see the project memory
`pensieve-benchmark-fairness`). Served through the existing ns-3 inference bridge as
`--policy pensieve`.
"""
