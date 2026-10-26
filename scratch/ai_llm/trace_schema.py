#!/usr/bin/env python3
"""
Shared constants for trace extraction and training record contracts.
"""

# segmentId in decision logs is the segment just completed before the next request.
COMPLETED_SEGMENT_CONVENTION = "completed_before_next_request"

# Training-record schema version.
#   1 = legacy: model_input carries baked network_norm/semantic_norm (scaler applied at
#       build time).
#   2 = raw: model_input carries only raw ladder_context + missing_masks; network_norm /
#       semantic_norm are derived at load time from `observation` / `semantic_raw` via the
#       pinned scaler (see features.build_model_input_from_row).
SCHEMA_VERSION = 2

# Supervised label: bitrate class over the 6-rung ladder (bps values in paths.NS3_RATE_LADDER_BPS).
LABEL_FIELD_BPS = "chosen_bitrate_bps"
LABEL_FIELD_CLASS = "chosen_bitrate_class"

# ns-3 MPEG timing model (playback clock), not encoded byte size from metadata.
# MUST mirror src/dash/model/mpeg-header.h. 100 * 20 ms = 2.0 s = the encoded segment duration
# (abr pipeline segment_duration_s=2). (Was 33 ms => 3.3 s, mismatching the 2.0 s media.)
MPEG_FRAMES_PER_SEGMENT = 100
MPEG_FRAME_DURATION_MS = 20
MPEG_SEGMENT_DURATION_S = MPEG_FRAMES_PER_SEGMENT * MPEG_FRAME_DURATION_MS / 1000.0

# throughput_trace.jsonl (scripts/extract_throughput_traces.py) - raw keyed rows before aggregation.
THROUGHPUT_TRACE_REQUIRED_FIELDS = [
    "run_id",
    "node_id",
    "segment_id",
    "throughput_bps",
    "segment_fetch_time_s",
    "segment_bytes",
]

THROUGHPUT_TRACE_OPTIONAL_FIELDS = [
    "video_id",
    "video_name",
    "metadata_csv",
    "sim_time_s",
    "segment_duration_s",
    "segment_convention",
    "source",
    "algo_short",
    "throughput_estimate_bps",
    "chosen_bitrate_bps",
]

# aggregated_throughput.jsonl (scripts/aggregate_traces.py) - pooled per-video segment summaries.
AGGREGATED_THROUGHPUT_REQUIRED_FIELDS = [
    "video_name",
    "segment_id",
    "sample_count",
    "throughput_bps_median",
    "throughput_bps_std",
    "segment_fetch_time_s_median",
    "segment_fetch_time_s_std",
    "segment_bytes_median",
    "segment_bytes_std",
    "throughput_bps_conservative",
    "segment_fetch_time_s_conservative",
]

AGGREGATED_THROUGHPUT_OPTIONAL_FIELDS = [
    "metadata_csv",
    "algo_short",
    "segment_duration_s",
    "segment_convention",
    "run_ids",
    "node_count",
    "run_count",
]

# oracle_labels_state.jsonl (scripts/dp_oracle.py) -
# per-segment supervised labels keyed by (run_id, node_id, segment_id).
# Each row carries the optimal action under a semantic backward-DP rollout that
# replays the run/node throughput trace with a buffer/stall transition model.
ORACLE_LABELS_STATE_REQUIRED_FIELDS = [
    "run_id",
    "node_id",
    "segment_id",
    "video_name",
    "oracle_bitrate_idx",
    "oracle_bitrate_bps",
    "oracle_policy",
]

ORACLE_LABELS_STATE_OPTIONAL_FIELDS = [
    # DP diagnostics
    "immediate_reward",
    "future_value",
    "total_q_value",
    # Replay state at the decision point
    "buffer_before_s",
    "buffer_after_s",
    "predicted_download_time_s",
    "predicted_rebuffer_s",
    "playing_before",
    "playing_after",
    "prev_rung",
    # Semantic context that fed the reward
    "semantic_weight",
    "is_scene_cut",
    "scene_cut_count",
    "vmaf_chosen",
    "vmaf_prev",
    "switch_penalty",
    # Throughput used by the transition model
    "throughput_bps",
    "throughput_bps_planned",
    "throughput_source",
    # Session-level
    "session_total_qoe",
    "session_segment_count",
    # Provenance
    "segment_duration_s",
    "metadata_csv",
    "algo_short",
    "reward_config",
]
