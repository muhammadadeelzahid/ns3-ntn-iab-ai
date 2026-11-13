#!/usr/bin/env python3
"""Train Pensieve from scratch (random init) with a MINIMAL, budget-capped synchronous A2C loop.

Fairness (see project memory `pensieve-benchmark-fairness`): single synchronous agent (not A3C's 16
parallel actors) and a small default update budget, so Pensieve never out-trains our LLM policy
(~22k rows x ~15 effective epochs ~= 1e4 gradient steps). Default here = 2000 updates x 32 steps
(~64k chunk-decisions, ~2000 gradient steps) - comfortably under the LLM budget. Bump --updates
only with that fairness ceiling in mind.

Run under the torch venv:
    ~/abr-llm-venv/bin/python scratch/ai_llm/pensieve/train.py \
        --out checkpoints_pensieve/pensieve_actor.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # scratch/
sys.path.insert(0, _HERE)

from ai_llm.pensieve.env import ABREnv                 # noqa: E402
from ai_llm.pensieve.model import (                    # noqa: E402
    A_DIM, PENSIEVE_LADDER_MBPS, S_INFO, S_LEN, ActorCritic,
)
from ai_llm.pensieve.traces import build_episodes, DEFAULT_ARTIFACTS  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal synchronous A2C trainer for Pensieve.")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "checkpoints_pensieve", "pensieve_actor.pt"))
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS)
    ap.add_argument("--algos", nargs="+", default=["Fdash"],
                    help="Source arms for network traces (default Fdash: policy-independent).")
    ap.add_argument("--max-episodes", type=int, default=None, help="Cap the trace pool size.")
    ap.add_argument("--updates", type=int, default=2000, help="Gradient updates.")
    ap.add_argument("--n-steps", type=int, default=32, help="Rollout length per update.")
    ap.add_argument("--num-envs", type=int, default=16,
                    help="Parallel (vectorized) envs per step. B>1 batches the forward pass so each "
                         "gradient update sees B*n_steps samples at ~one forward's cost - the real "
                         "wall-clock lever (single-env A2C is Python-overhead bound).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy-coef", type=float, default=0.1,
                    help="Fixed entropy bonus (used only if --entropy-start is not set).")
    ap.add_argument("--entropy-start", type=float, default=None,
                    help="Anneal entropy bonus linearly from this to --entropy-end over training. "
                         "High->low keeps early exploration then sharpens. Overrides --entropy-coef.")
    ap.add_argument("--entropy-end", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=200,
                    help="Eval on the held-out set every N updates and keep the BEST checkpoint by the "
                         "offline VMAF-%oracle proxy (early-stop on the target metric, not the final iterate).")
    ap.add_argument("--eval-episodes", type=int, default=150, help="Held-out episodes for selection eval.")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="cpu is plenty (tiny net).")
    ap.add_argument("--log-every", type=int, default=100)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    episodes = build_episodes(args.artifacts_root, args.algos, args.max_episodes)
    if not episodes:
        raise SystemExit("No episodes built - check --artifacts-root / --algos.")
    B = max(1, int(args.num_envs))

    # Held-out split for checkpoint SELECTION (offline VMAF-%oracle proxy). Seeded shuffle; the eval
    # slice is excluded from the training envs so selection isn't on trained-on episodes.
    from ai_llm.pensieve.offline_eval import eval_policy  # noqa: E402
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(episodes))
    n_eval = min(int(args.eval_episodes), max(1, len(episodes) // 5))
    eval_eps = [episodes[i] for i in order[:n_eval] if episodes[i].vmafs is not None]
    train_eps = [episodes[i] for i in order[n_eval:]]
    if not train_eps:
        train_eps = episodes

    print("[pensieve-train] episodes=%d (train %d / eval %d) videos=%d updates=%d n_steps=%d num_envs=%d (~%d samples)" % (
        len(episodes), len(train_eps), len(eval_eps), len({e.video_name for e in episodes}),
        args.updates, args.n_steps, B, args.updates * args.n_steps * B), flush=True)

    # Vectorized synchronous A2C: B independent envs (distinct seeds) stepped together; the forward
    # pass batches to [B, ...] so each update sees B*n_steps samples at ~one forward's cost.
    envs = [ABREnv(train_eps, seed=args.seed + i) for i in range(B)]
    model = ActorCritic().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    states = np.stack([e.reset() for e in envs]).astype(np.float32)  # [B, 6, 8]
    ep_run = np.zeros(B, dtype=np.float64)   # running return per env
    ep_returns = []                          # completed-episode returns (for logging)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save_ckpt(update, proxy_pct):
        torch.save({
            "state_dict": model.state_dict(), "variant": "pensieve",
            "window": S_LEN, "s_info": S_INFO, "a_dim": A_DIM,
            "ladder_mbps": PENSIEVE_LADDER_MBPS, "train_updates": update,
            "n_steps": args.n_steps, "num_envs": int(args.num_envs),
            "selected_proxy_oracle_pct": proxy_pct,
        }, args.out)

    def entropy_coef(update):
        if args.entropy_start is None:
            return args.entropy_coef
        frac = (update - 1) / max(1, args.updates - 1)
        return args.entropy_start + (args.entropy_end - args.entropy_start) * frac

    best_pct, best_update = float("-inf"), 0

    for update in range(1, args.updates + 1):
        log_probs, values, rewards, entropies, masks = [], [], [], [], []
        for _ in range(args.n_steps):
            state_t = torch.from_numpy(states).to(device)          # [B, 6, 8]
            logits, value = model(state_t)                          # [B, A_DIM], [B, 1]
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()                                 # [B]
            log_probs.append(dist.log_prob(actions))                # [B]
            values.append(value.squeeze(1))                         # [B]
            entropies.append(dist.entropy())                        # [B]

            acts = actions.cpu().numpy()
            next_states = np.empty_like(states)
            r_step = np.zeros(B, dtype=np.float32)
            m_step = np.ones(B, dtype=np.float32)
            for i, e in enumerate(envs):
                ns, r, done = e.step(int(acts[i]))
                r_step[i] = r
                ep_run[i] += r
                if done:
                    ep_returns.append(ep_run[i]); ep_run[i] = 0.0
                    ns = e.reset()
                    m_step[i] = 0.0
                next_states[i] = ns
            states = next_states.astype(np.float32)
            rewards.append(torch.from_numpy(r_step).to(device))     # [B]
            masks.append(torch.from_numpy(m_step).to(device))       # [B]

        # Bootstrap value of the state after the rollout, per env.
        with torch.no_grad():
            _, next_value = model(torch.from_numpy(states).to(device))
            next_value = next_value.squeeze(1)                      # [B]

        returns, R = [], next_value
        for t in reversed(range(args.n_steps)):
            R = rewards[t] + args.gamma * R * masks[t]
            returns.insert(0, R)
        returns_t = torch.stack(returns).detach().reshape(-1)       # [n_steps*B]
        values_t = torch.stack(values).reshape(-1)
        log_probs_t = torch.stack(log_probs).reshape(-1)
        entropy_t = torch.stack(entropies).mean()

        # Normalize advantages (invariant to the large, heavy-congestion reward scale); Huber critic
        # so big negative returns don't explode gradients.
        norm_adv = (returns_t - values_t).detach()
        norm_adv = (norm_adv - norm_adv.mean()) / (norm_adv.std() + 1e-8)
        ent_coef = entropy_coef(update)
        actor_loss = -(log_probs_t * norm_adv).mean()
        critic_loss = F.smooth_l1_loss(values_t, returns_t)
        loss = actor_loss + args.value_coef * critic_loss - ent_coef * entropy_t

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if update % args.log_every == 0 or update == 1:
            recent = ep_returns[-50:] if ep_returns else [0.0]
            print("[pensieve-train] update %d/%d loss=%.3f avg_ep_return=%.2f entropy=%.3f ent_coef=%.3f" % (
                update, args.updates, float(loss.item()), float(np.mean(recent)),
                float(entropy_t.item()), ent_coef), flush=True)

        # Selection: eval on held-out episodes; keep the BEST checkpoint by the VMAF-%oracle proxy.
        if eval_eps and (update % args.eval_every == 0 or update == args.updates):
            pct = eval_policy(model, eval_eps, device=args.device, window=S_LEN)
            marker = ""
            if pct == pct and pct > best_pct:   # pct==pct guards NaN
                best_pct, best_update = pct, update
                save_ckpt(update, pct)
                marker = " <- BEST, saved"
            print("[pensieve-train] EVAL update %d: proxy_oracle=%.1f%% (best %.1f%% @%d)%s" % (
                update, pct, best_pct, best_update, marker), flush=True)

    if best_update == 0:                        # no eval improved (or no eval set): save final
        save_ckpt(args.updates, float("nan"))
        print("[pensieve-train] saved FINAL %s (no eval improvement)" % args.out, flush=True)
    else:
        print("[pensieve-train] BEST checkpoint = update %d, proxy_oracle=%.1f%% -> %s" % (
            best_update, best_pct, args.out), flush=True)


if __name__ == "__main__":
    main()
