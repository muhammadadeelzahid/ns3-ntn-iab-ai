#!/usr/bin/env python3
"""
DASH ABR orchestrator: runs ntn-iab-quic-dash-ai with ns3::AiDashClient and
exchanges AbrDash state/actions over ns3-ai shared memory (FDASH heuristic in Python).

Author: Muhammad Adeel Zahid <adeel.m.zahid@gmail.com>
        University of Manitoba

Added: Python orchestrator for ns3::AiDashClient (FDASH heuristic over AbrDash IPC).

Run from repository root: python3 scratch/run_ai_dash_heuristic.py
"""

import sys
import os
import time

# Add src/ai/python_utils and the build directory to PYTHONPATH
ns3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ns3_root, 'src', 'ai', 'python_utils'))
sys.path.insert(0, os.path.join(ns3_root, 'build', 'src', 'ai', 'examples', 'abr-dash-ai', 'use-msg'))

from ns3ai_utils import Experiment
import ns3ai_abr_dash_ai_py as py_binding

import math

def fdash_heuristic(state):
    """
    Python implementation of FdashClient::CalcNextSegment
    """
    t = 30.0  # target_dt is configured as 30.0s in ntn-iab-quic-dash-ai.cc
    currDt = state.buffer_level_s
    diff = state.buffer_differential_s
    m_bitrateEstimate = state.throughput_estimate_bps

    slow = ok = fast = falling = steady = rising = 0.0

    if currDt < 2 * t / 3:
        slow = 1.0
    elif currDt < t:
        slow = 1 - 1 / (t / 3) * (currDt - 2 * t / 3)
        ok = 1 / (t / 3) * (currDt - 2 * t / 3)
    elif currDt < 4 * t:
        ok = 1 - 1 / (3 * t) * (currDt - t)
        fast = 1 / (3 * t) * (currDt - t)
    else:
        fast = 1.0

    if diff < -2 * t / 3:
        falling = 1.0
    elif diff < 0:
        falling = 1 - 1 / (2 * t / 3) * (diff + 2 * t / 3)
        steady = 1 / (2 * t / 3) * (diff + 2 * t / 3)
    elif diff < 4 * t:
        steady = 1 - 1 / (4 * t) * diff
        rising = 1 / (4 * t) * diff
    else:
        rising = 1.0

    r1 = min(slow, falling)
    r2 = min(ok, falling)
    r3 = min(fast, falling)
    r4 = min(slow, steady)
    r5 = min(ok, steady)
    r6 = min(fast, steady)
    r7 = min(slow, rising)
    r8 = min(ok, rising)
    r9 = min(fast, rising)

    p2 = math.sqrt(r9**2)
    p1 = math.sqrt(r6**2 + r8**2)
    z = math.sqrt(r3**2 + r5**2 + r7**2)
    n1 = math.sqrt(r2**2 + r4**2)
    n2 = math.sqrt(r1**2)

    FDASH_N2_R = 0.0
    FDASH_N1_SR = 0.25
    FDASH_Z_NC = 1.0
    FDASH_P1_SI = 1.5
    FDASH_P2_I = 3.0

    sum_memberships = n2 + n1 + z + p1 + p2
    output = 0.0
    if sum_memberships > 0:
        output = (n2 * FDASH_N2_R + n1 * FDASH_N1_SR + z * FDASH_Z_NC + p1 * FDASH_P1_SI + p2 * FDASH_P2_I) / sum_memberships

    result = output * m_bitrateEstimate
    
    rates = [45000, 89000, 131000, 178000, 221000, 263000, 334000,
             396000, 522000, 595000, 791000, 1033000, 1245000, 1547000,
             2134000, 2484000, 3079000, 3527000, 3840000, 4220000, 9500000,
             15000000, 30000000, 66000000, 85000000]
    
    m_interruptionLimit = rates[-1]
    if result > m_interruptionLimit:
        result = m_interruptionLimit
        
    nextRate = rates[0]
    for rate in rates:
        if result > rate:
            nextRate = rate
            
    currRate = state.current_bitrate_bps
    if currRate == 0:
        currRate = rates[0] # handle first segment
        
    if nextRate > currRate:
        t_lim = currDt + (m_bitrateEstimate / nextRate - 1) * 2 * t
        if t_lim < t:
            nextRate = currRate
            # native recalculates t_lim but it's not used further for rate
    elif nextRate < currRate and m_interruptionLimit == rates[-1]:
        t_lim = currDt + (m_bitrateEstimate / nextRate - 1) * 2 * t
        if t_lim > t:
            t_lim = currDt + (m_bitrateEstimate / currRate - 1) * 2 * t
            if t_lim > t:
                nextRate = currRate

    delay_s = 0.0
    if nextRate == rates[-1]:
        # Constants from mpeg-header.h / fdash-client.cc
        MPEG_TIME_BETWEEN_FRAMES = 40.0 # ms
        MPEG_FRAMES_PER_SEGMENT = 50.0
        
        if m_bitrateEstimate > 0:
            time_to_download = (nextRate * MPEG_TIME_BETWEEN_FRAMES * MPEG_FRAMES_PER_SEGMENT) / 1000.0 / m_bitrateEstimate
            sleep_time = currDt - t - time_to_download
            if sleep_time > 0:
                delay_s = sleep_time

    print(f"[Python] FDASH Node {state.node_id}, Seg {state.segment_id}:")
    print(f"         Buffer: {currDt:.2f}s, Diff: {diff:.2f}s")
    print(f"         Est Tput: {m_bitrateEstimate / 1e6:.2f} Mbps")
    print(f"         Output Multiplier: {output:.4f}")
    print(f"         Continuous Result: {result / 1e6:.2f} Mbps")
    print(f"         Final Discrete Target: {nextRate / 1e6:.2f} Mbps")
    print(f"         Calculated Delay: {delay_s:.4f}s", flush=True)

    return int(nextRate), delay_s

def main():
    # Force ns3ai_utils to use python3 ./waf
    os.environ['WAF_CMD'] = 'python3 ./waf'
    ns3_root = os.path.abspath('.')
    
    exp = Experiment(
        'ntn-iab-quic-dash-ai',       # waf target name
        ns3_root,                      # path to ns-3 root
        py_binding,                    # pybind11 module
        handleFinish=True,
        useVector=False,               # struct mode for ABR
        shmSize=4096,                  # matched with C++ config
    )
    
    # Run the experiment. We pass aiEnable=True to enable the ns3-ai logic.
    # Set numUes to 1 for the baseline comparison run.
    msg = exp.run(setting={'aiEnable': 'true', 'numUes': '1', 'run': '1', 'algorithm': 'ns3::AiDashClient'}, show_output=True)
    
    print("[Python] Experiment started, waiting for ns-3...", flush=True)
    
    try:
        decisions_made = 0
        while True:
            # Block until C++ writes the state
            msg.PyRecvBegin()
            
            # Check if simulation finished
            if msg.PyGetFinished():
                print("[Python] Simulation finished!", flush=True)
                break
            
            # Get the state struct
            state = msg.GetCpp2PyStruct()
            
            # Make a decision
            chosen_bitrate, delay_s = fdash_heuristic(state)
            
            msg.PyRecvEnd()
            
            # Write the action back to C++
            msg.PySendBegin()
            action = msg.GetPy2CppStruct()
            action.chosen_bitrate_bps = chosen_bitrate
            action.delay_s = delay_s
            msg.PySendEnd()
            
            decisions_made += 1
            
            # Stop early if we just want a quick test
            # if decisions_made >= 10:
            #    print("[Python] 10 decisions made, testing complete.")
            #    break
                
    except KeyboardInterrupt:
        print("[Python] Interrupted by user.", flush=True)
    finally:
        exp.kill()
        print("[Python] Experiment killed.", flush=True)

if __name__ == '__main__':
    main()
