#ifndef NS3AI_ABR_DASH_AI_MSG_H
#define NS3AI_ABR_DASH_AI_MSG_H

#include <cstdint>

/**
 * C++ -> Python message type for ABR DASH AI.
 * Contains the current network state, buffer status, and playback info.
 */
struct AbrDashCpp2PyMsg {
    // Identity
    uint32_t node_id{0};           // ns-3 node ID of this DASH client
    int32_t  segment_id{0};        // Current segment index

    // Timing
    double   sim_time_s{0.0};        // Simulator::Now() in seconds
    double   segment_fetch_time_s{0.0}; // Time to download last segment

    // Network state (from DashClient)
    double   throughput_estimate_bps{0.0}; // Windowed bitrate estimate
    double   buffer_level_s{0.0};         // Average buffer level (seconds)
    double   buffer_differential_s{0.0};  // Buffer trend (rising/falling)

    // Player state
    uint32_t current_bitrate_bps{0};    // Current segment bitrate
    uint32_t prev_bitrate_bps{0};       // Previous segment bitrate
    int32_t  player_state{0};           // 0=initial_buffering, 1=playing, 2=paused, 3=done
    double   interruption_time_s{0.0};    // Total stall time so far
    uint32_t interruptions{0};          // Total stall count
};

/**
 * Python -> C++ message type for ABR DASH AI.
 * Contains the bitrate decision from the LLM.
 */
struct AbrDashPy2CppMsg {
    uint32_t chosen_bitrate_bps{0};     // Bitrate decision from LLM
    double   delay_s{0.0};                // Optional delay before next request
};

#endif // NS3AI_ABR_DASH_AI_MSG_H
