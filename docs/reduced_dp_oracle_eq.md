> **See also:** [`oracle-rung-selection.md`](oracle-rung-selection.md) — how the reward
> becomes a chosen rung (greedy vs DP, with a worked example). This file defines the
> reward/equations; that one explains the selection mechanism.

## 1. Objective

Over all segments `t = 0 .. N-1`, the oracle maximizes the **sum of immediate rewards** on the chosen path:

```
Total_QoE = R_0 + R_1 + R_2 + ... + R_{N-1}
```

## 2. Per-segment reward equation

For segment `t`, if you choose rung `a` (index 0–5), the **immediate reward** is:

```
R_t = w_vmaf * semantic_weight_t * vmaf_t(a)
    - w_stall * rebuffer_seconds_t
    - switch_penalty_t
```
---

### 3. Semantic weight (per segment)

Before DP runs, each segment gets one scalar `semantic_weight_t` from metadata:

- **w_vmaf**: Global knob: how strongly VMAF matters vs stall/switch penalties
- **vmaf_t(a)**: How good rung `a` looks on segment `t` (from metadata)
- **semantic_weight_t**: How much **this segment’s** quality should count, from content semantics

```
semantic_weight_t = 0.5 * norm_quality_sensitivity_t
                  + 0.3 * norm_saliency_score_t
                  + 0.2 * norm_object_score_t
```

Each of `norm_*` is **min-max normalized across all segments of that video** (scaled to 0–1 within the video). Segments with higher sensitivity, saliency, or object score get a larger weight, so VMAF gains matter more there.

---

### Switch penalty (only when t > 0)

```
switch_penalty_t = w_switch * abs(vmaf_t(a) - vmaf_{t-1}(a_{t-1})) * (1 - cut_discount)

cut_discount = scene_cut_discount   if scene_cut_count_t > 0
             else 0                   otherwise
```

**Between two consecutive segments**, if the chosen quality (VMAF) **jumps a lot**, the oracle pays a **penalty**.  
`w_switch` controls **how big** that penalty is.

On a **scene cut**, the switch penalty is reduced: large VMAF jumps are allowed without as much penalty.

## 4. Player and buffer model

**During playback**:

```
rebuffer_seconds_t = max(0, download_time_t(a) - buffer_before)

buffer_after = min(max_buffer_s,
                   max(0, buffer_before - download_time_t(a)) + segment_duration_s)
```

- If download is slower than buffered time, the player stalls for the difference  
- After download, leftover buffer plus one segment of playback time forms the new buffer  
- `max_buffer_s` default is 30 s**Download time** for segment t at rung a:

```
download_time_t(a) = (8 * size_bytes_t(a)) / throughput
```

## Training Data

```json
"model_input": {
  "network_norm": [7 floats],
  "semantic_norm": [5 floats],
  "ladder_context": [[6 rungs] x [size_bytes, vmaf]]
}
```

### Network Features (7-dim contract — current)

Fixed order from `scratch/ai_llm/features.py` `NETWORK_FEATURE_NAMES`. The original 15-dim design was reduced to these 7 (see `training_data_contract` plan for the dropped fields).

| Index | Feature name |
| ----- | ------------ |
| 0 | `throughput_estimate_bps` |
| 1 | `buffer_level_s` |
| 2 | `buffer_differential_s` |
| 3 | `segment_fetch_time_s` |
| 4 | `last_segment_bitrate_bps` |
| 5 | `smoothed_rtt_ms` |
| 6 | `loss_rate_recent` |

> Dropped from the old 15-dim contract: `download_time_ratio`, `throughput_ema_bps`, `throughput_std_bps`, `prev_bitrate_bps`, `interruption_time_s`, `interruptions`, `last_decision_delay_s`, `bitrate_switch_count_recent`.
>
> Index 4 changed (Jun 2026): `current_bitrate_bps` → `last_segment_bitrate_bps`. See "Nominal rung vs actual encoded bitrate" below.

### Nominal rung vs actual encoded bitrate

The oracle and the student are two separate contracts:

- **Oracle (label generation)** consumes only **throughput** on the network side, via `download_time = (8 · size_bytes) / throughput`. RTT, loss, buffer, and bitrate are not parameters of the reward `R_t` or of the DP state — buffer is replayed internally. So of the network quantities, the label depends on throughput alone.
- **Training parameters (`model_input.network_norm`)** are what the student reads to *predict* that label.

Within the training parameters, the old `current_bitrate_bps` was **fully redundant with the rung**: it was logged as the *nominal* ladder rate (exactly `{250000, 750000, 1500000, 3000000, 6000000}`), a deterministic 1:1 map of the rung index — the same kind of restatement as the dropped `download_time_ratio`.

But ffmpeg encodes **VBR**, so a segment's *actual* bitrate diverges from its nominal rung by scene complexity. Measured over the 12,114-row corpus:

| rung | nominal (kbps) | actual encoded min–max (kbps) |
| ---- | -------------- | ----------------------------- |
| 0 | 250 | 24 – 3,595 |
| 1 | 750 | 33 – 12,329 |
| 2 | 1500 | 36 – 25,548 |
| 3 | 3000 | 52 – 59,243 |
| 4 | 6000 | 76 – 124,743 |
| 5 | 12000 | 76 – 124,743 |

The per-rung actual ranges fully overlap (you cannot recover the rung from the actual bitrate alone), and **88.5% of decisions have actual ≠ nominal, differing ~26% on average**.

**Change:** index 4 now carries **`last_segment_bitrate_bps`** — the actual encoded bitrate of the just-downloaded segment (segment `t-1` at the rung given by `oldBitRate`), sourced from metadata `actual_bitrate_kbps`; `0.0` at startup. This swaps a rung-redundant value for the real network-load signal that drove the last download's bytes and buffer drain. The rung itself remains available implicitly through the oracle label and `ladder_context` (per-rung VMAF), so no separate nominal-bitrate feature is needed. The feature is `log1p`-transformed before z-scoring (`config/feature_scaler_network.json` `log_features`), since it spans ~24 kbps – ~125 Mbps.

> **Runtime parity (live inference) — TODO:** the C++ IPC (`AbrDashCpp2PyMsg.current_bitrate_bps` in `src/ai/examples/abr-dash-ai/use-msg/abr-dash-ai-msg.h`, populated in `ai-dash-client.cc`) still emits the nominal current rate. When the online policy is wired, that field must be renamed/repurposed to provide the actual last-segment bytes so live `network_norm` matches the trained contract. Until then, the live path feeds `0.0` for index 4.

### `Semantic features'

| Index | Feature |
| ----- | ------- |
| 0 | `motion` |
| 1 | `saliency` |
| 2 | `scene_cut` |
| 3 | `object` |
| 4 | `quality_sensitivity` |

### `ladder_context` — 6×2 Matrix

One row per bitrate rung:

```text
250k, 750k, 1500k, 3000k, 6000k, 12000k
```

Each row contains:

```text
[size_bytes, vmaf]
```

> Changed (Jun 2026): dropped `actual_bitrate_kbps`. It equals `size_bytes × 8 / 1000 / segment_duration` with a **fixed 2 s** segment duration (verified: the ratio is exactly 0.5 across all 72,684 ladder entries, std 0.0), so it is perfectly collinear with `size_bytes` and carried no extra information. `size_bytes` is kept because `download_time = 8 · size_bytes / throughput` uses it directly; `vmaf` is independent (quality is not a function of size). The audit `ladder` block still records `actual_bitrate_kbps`, and `metadata_loader` still loads it (it sources `last_segment_bitrate_bps`). If variable segment durations are ever introduced, the two columns decouple and this should be revisited.




----

## Worked example (3 rungs, 5 segments)

### Greedy rule

```
At segment t:
  For each rung r in {0, 1, 2}:
      simulate download, buffer, rebuffer from current player state
      compute immediate reward R_t(r)
  Choose r* = argmax_r R_t(r)
  Update player state; continue to segment t+1
```

### Setting


| Parameter             | Value                                                                            |
| --------------------- | -------------------------------------------------------------------------------- |
| Rungs                 | 3 (low / mid / high)                                                             |
| Segments              | 5 (indices 0–4)                                                                  |
| `segment_duration_s`  | 2.0                                                                              |
| `startup_threshold_s` | 4.0                                                                              |
| `max_buffer_s`        | 10.0                                                                             |
| Reward                | `R_t = VMAF - 25 * rebuffer - switch_penalty` (no semantic weight, `w_rung = 0`) |
| `switch_penalty`      | 0 on segment 0; else `1.0 * abs(VMAF_now - VMAF_prev)`                           |


**Ladder:**


| Rung | Name | VMAF | Segment size |
| ---- | ---- | ---- | ------------ |
| 0    | low  | 30   | 100 KB       |
| 1    | mid  | 50   | 200 KB       |
| 2    | high | 70   | 400 KB       |


**Throughput per segment:**


| Segment index | 0      | 1      | 2            | 3            | 4      |
| ------------- | ------ | ------ | ------------ | ------------ | ------ |
| Throughput    | 3 Mbps | 3 Mbps | **0.5 Mbps** | **0.5 Mbps** | 2 Mbps |


**Download time:** `download_time(r, t) = (8 * SIZE[r]) / TP[t]`

**Startup** (`playing = 0`): `rebuffer = 0`; `buffer_after = buffer_before + 2.0`; playback starts when buffer ≥ 4.0 s.

**Playback** (`playing = 1`): `rebuffer = max(0, download_time - buffer_before)`; buffer drains during download then adds 2.0 s.

Each segment uses the exact `buffer_after` from the previous step as `buffer_before` (no quantization).

### Segment 0 - startup (`buffer_before = 0`, `playing = 0`)


| Rung   | Immediate `R_0`     | Picked? |
| ------ | ------------------- | ------- |
| 0 low  | `30 - 0 =` **30.0** |         |
| 1 mid  | `50 - 0 =` **50.0** |         |
| 2 high | `70 - 0 =` **70.0** | **yes** |


All rungs: startup → `buffer_after = 2.0` s, still `playing = 0`.

**Greedy choice: high.** Cumulative = **70.0**.

### Segment 1 - startup (`buffer_before = 2.0`, `playing = 0`)


| Rung   | Switch | Immediate `R_1`       | Picked? |
| ------ | ------ | --------------------- | ------- |
| 0 low  | 40     | `30 - 40 =` **−10.0** |         |
| 1 mid  | 20     | `50 - 20 =` **30.0**  |         |
| 2 high | 0      | `70 - 0 =` **70.0**   | **yes** |


`buffer_after = 4.0` s → `playing = 1` (playback starts).

**Greedy choice: high.** Cumulative = **140.0**.

### Segment 2 - slow link (`buffer_before = 4.0`, `playing = 1`, `TP = 500,000`)


| Rung   | Download | Rebuffer | Switch | Immediate `R_2`          | Picked? |
| ------ | -------- | -------- | ------ | ------------------------ | ------- |
| 0 low  | 1.6 s    | 0        | 40     | **−10.0**                |         |
| 1 mid  | 3.2 s    | 0        | 20     | **30.0**                 | **yes** |
| 2 high | 6.4 s    | 2.4 s    | 0      | `70 - 25×2.4 =` **10.0** |         |


`buffer_after = 2.8` s (next segment starts with `buffer_before = 2.8`).

**Greedy choice: mid** (avoids 2.4 s stall on high). Cumulative = **170.0**.

### Segment 3 - slow link (`buffer_before = 2.8`, `playing = 1`, `TP = 500,000`)


| Rung   | Download | Rebuffer | Switch | Immediate `R_3`            | Picked? |
| ------ | -------- | -------- | ------ | -------------------------- | ------- |
| 0 low  | 1.6 s    | 0        | 20     | **10.0**                   |         |
| 1 mid  | 3.2 s    | 0.4 s    | 0      | `50 - 25×0.4 =` **40.0**   | **yes** |
| 2 high | 6.4 s    | 3.6 s    | 20     | `70 - 90 - 20 =` **−40.0** |         |


`buffer_after = 2.0` s.

**Greedy choice: mid.** Cumulative = **210.0**.

### Segment 4 - throughput recovers (`buffer_before = 2.0`, `playing = 1`, `TP = 2,000,000`)


| Rung   | Download | Rebuffer | Switch | Immediate `R_4`      | Picked? |
| ------ | -------- | -------- | ------ | -------------------- | ------- |
| 0 low  | 0.4 s    | 0        | 20     | **10.0**             |         |
| 1 mid  | 0.8 s    | 0        | 0      | **50.0**             | **yes** |
| 2 high | 1.6 s    | 0        | 20     | `70 - 20 =` **50.0** | (tie)   |


Mid and high **tie** at immediate **50.0**; greedy picks **mid** (first max in rung order).

**Greedy choice: mid.** Cumulative = **260.0**.

### Greedy result summary

**Rung path:** `[2, 2, 1, 1, 1]` (high, high, mid, mid, mid)

**Total QoE = 260.0**


| Seg | Greedy rung | Mode         | TP   | Download (s) | Rebuf (s) | Buf before | Buf after | Immediate | Cumulative |
| --- | ----------- | ------------ | ---- | ------------ | --------- | ---------- | --------- | --------- | ---------- |
| 0   | 2 high      | startup      | 3M   | 1.07         | 0         | 0.0        | 2.0       | 70.0      | 70.0       |
| 1   | 2 high      | startup→play | 3M   | 1.07         | 0         | 2.0        | 4.0       | 70.0      | 140.0      |
| 2   | 1 mid       | playback     | 0.5M | 3.20         | 0         | 4.0        | 2.8       | 30.0      | 170.0      |
| 3   | 1 mid       | playback     | 0.5M | 3.20         | 0.4       | 2.8        | 2.0       | 40.0      | 210.0      |
| 4   | 1 mid       | playback     | 2M   | 0.80         | 0         | 2.0        | 3.2       | 50.0      | **260.0**  |



---

