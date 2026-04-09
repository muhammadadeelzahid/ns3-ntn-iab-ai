/*
 * QUIC BBR AI message structs for ns3-ai msg-interface (struct/vector based).
 *
 * These structs are placed in shared memory, so keep them POD-like:
 * - fixed-size fields only (no std::string, no pointers)
 * - stable types (uint32_t/uint64_t/double)
 */

#ifndef NS3AI_QUIC_BBR_AI_MSG_H
#define NS3AI_QUIC_BBR_AI_MSG_H

#include <cstdint>

/**
 * One observation item for a single QUIC connection (or subflow/path).
 *
 * Units:
 * - *_bps: bits per second
 * - *_s: seconds
 * - *_bytes: bytes
 */
struct QuicBbrAiConnObs
{
    // Identity
    uint32_t node_id{0};
    uint32_t conn_id{0}; // QuicL4Protocol socket index
    uint32_t path_id{0}; // for multipath / subflow (0 for single-path)

    // Timestamping
    double time_s{0.0};
    double dt_s{0.0}; // aggregation interval used for *_over_dt fields

    // Congestion state (core)
    uint64_t bytes_in_flight_bytes{0};
    uint64_t unacked_data_bytes{0}; // typically equal to bytes_in_flight_bytes; can be extended later

    // RTT signals
    double smoothed_rtt_s{0.0};
    double rttvar_s{0.0};
    double latest_rtt_s{0.0};
    double min_rtt_s{0.0};

    // Delivery / throughput signals
    uint64_t acked_bytes_since_last{0};
    uint64_t acked_bytes_over_dt{0};
    uint64_t delivered_bytes_over_dt{0};
    uint64_t throughput_bps_over_dt{0}; // computed from acked_bytes_over_dt and dt_s
    uint64_t delivery_rate_bps{0};       // RateSample::m_deliveryRate

    // ACK timing (aggregated)
    double ack_interarrival_last_s{0.0};
    double ack_interarrival_mean_s_over_dt{0.0};
    double ack_interarrival_min_s_over_dt{0.0};
    double ack_interarrival_max_s_over_dt{0.0};

    // Loss / reordering / dup-ack (aggregated)
    uint32_t loss_event_count_over_dt{0};
    uint64_t bytes_lost_over_dt{0};
    uint32_t packet_loss_since_last{0}; // RateSample::m_packetLoss (delta-ish signal)
    uint32_t reordering_event_count_over_dt{0}; // derived from ACK gaps/blocks
    uint32_t dup_ack_count_over_dt{0};          // derived from repeated LargestAcknowledged w/ no progress

    // BBR-specific signals (optional to fill initially)
    uint64_t btlbw_bps{0};
    double rtprop_s{0.0};
    double pacing_gain{0.0};
    double cwnd_gain{0.0};
    uint64_t pacing_rate_bps{0};
    uint64_t cwnd_bytes{0};
    uint64_t target_cwnd_bytes{0};

    // Enumerations as integers for shared memory compatibility
    // 0=STARTUP, 1=DRAIN, 2=PROBE_BW, 3=PROBE_RTT
    uint8_t bbr_mode{0};

    // Flags
    uint8_t is_app_limited{0};
};

/**
 * C++ -> Python message type.
 *
 * Recommended usage: vector-based message interface, where each element is one connection obs.
 */
struct QuicBbrAiCpp2PyMsg
{
    QuicBbrAiConnObs obs;
};

/**
 * Python -> C++ message type (actions).
 *
 * Keep this minimal initially; you can expand later.
 */
struct QuicBbrAiPy2CppMsg
{
    uint32_t node_id{0};
    uint32_t conn_id{0};
    uint32_t path_id{0};

    // Smoothing / safety (C++ uses these to apply actions gradually)
    // alpha=1.0 means "apply immediately", alpha=0.0 means "do not change".
    double alpha{1.0};

    // BBR control targets
    uint8_t has_target_pacing_gain{0};
    double target_pacing_gain{0.0};
    double max_step_pacing_gain{0.0}; // 0 => use a conservative default in C++

    uint8_t has_target_cwnd_gain{0};
    double target_cwnd_gain{0.0};
    double max_step_cwnd_gain{0.0}; // 0 => use a conservative default in C++

    // Optional caps (0 means “no cap”)
    uint8_t has_pacing_rate_cap_bps{0};
    uint64_t pacing_rate_cap_bps{0};

    uint8_t has_cwnd_cap_bytes{0};
    uint64_t cwnd_cap_bytes{0};

    // QUIC reordering sensitivity (useful in NTN; apply conservatively)
    uint8_t has_reordering_threshold{0};
    uint32_t reordering_threshold{0};
    uint32_t max_step_reordering_threshold{0}; // 0 => use a conservative default in C++

    uint8_t has_time_reordering_fraction{0};
    double time_reordering_fraction{0.0};
    double max_step_time_reordering_fraction{0.0}; // 0 => use a conservative default in C++
};

#endif // NS3AI_QUIC_BBR_AI_MSG_H

