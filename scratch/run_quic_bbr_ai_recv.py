#!/usr/bin/env python3
# Phase 1: receive Cpp2Py observations and log fields.
# Phase 2 (partial): write dummy Py2Cpp actions per connection each tick (C++ apply step is separate).
#
# Run from repository root: python3 scratch/run_quic_bbr_ai_recv.py
# Or: cd scratch && ./run_quic_bbr_ai_recv.py

from __future__ import print_function

import argparse
import logging
import os
import shutil
import sys
import traceback

# Fixed layout: this file lives under scratch/; repo root (./waf) is the parent directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for _p in (
    os.path.join(_REPO_ROOT, 'src', 'ai', 'python_utils'),
    os.path.join(_REPO_ROOT, 'src', 'ai', 'examples', 'quic-bbr-ai', 'use-msg'),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ns3ai_quic_bbr_ai_msg_py as py_binding  # noqa: E402
from ns3ai_utils import Experiment  # noqa: E402


def fill_dummy_py2cpp_actions(py2cpp, cpp2py):
    """
    Per-connection dummy actions for Phase 2 bring-up.

    Copies identity from observations, then sets fixed transport + BBR targets
    (see quic-bbr-ai-msg.h). C++ must read GetPy2CppVector() and apply; until then
    this only populates shared memory for testing the write path.
    """
    n = len(cpp2py)
    py2cpp.resize(n)
    for i in range(n):
        obs = cpp2py[i].obs
        a = py2cpp[i]
        a.node_id = obs.node_id
        a.conn_id = obs.conn_id
        a.path_id = obs.path_id
        a.alpha = 1.0

        # Transport knobs (plan defaults)
        a.has_reordering_threshold = 1
        a.reordering_threshold = 2
        a.max_step_reordering_threshold = 1

        a.has_time_reordering_fraction = 1
        a.time_reordering_fraction = 1.125  # 9/8
        a.max_step_time_reordering_fraction = 0.05

        # BBR gain targets (C++ QuicBbr apply wiring comes later)
        a.has_target_pacing_gain = 1
        a.target_pacing_gain = 1.25
        a.max_step_pacing_gain = 0.05

        a.has_target_cwnd_gain = 1
        a.target_cwnd_gain = 2.0
        a.max_step_cwnd_gain = 0.1

        # Optional caps: leave unset (0 = no cap per msg.h)
        a.has_pacing_rate_cap_bps = 0
        a.pacing_rate_cap_bps = 0
        a.has_cwnd_cap_bytes = 0
        a.cwnd_cap_bytes = 0


def main():
    parser = argparse.ArgumentParser(
        description='QUIC BBR AI: Cpp2Py logging + dummy Py2Cpp actions per tick'
    )
    parser.add_argument('--num-ues', type=int, default=1, help='Scratch numUes')
    parser.add_argument('--ai-dt-ms', type=float, default=50.0, help='Scratch aiDtMs (>0 enables AI tick)')
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    )
    parser.add_argument('--show-waf-output', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
    )

    ns3_root = _REPO_ROOT
    if not os.path.isfile(os.path.join(ns3_root, 'waf')):
        logging.error('Expected waf at fixed repo root %s (scratch/ must stay under ns-3 root)', ns3_root)
        return 2

    if 'WAF_CMD' not in os.environ and shutil.which('python') is None:
        os.environ['WAF_CMD'] = 'python3 ./waf'

    setting = {
        'aiDtMs': str(args.ai_dt_ms),
#        'numUes': str(args.num_ues),
    }

    logging.info('NS3_ROOT=%s settings=%s', ns3_root, setting)

    exp = Experiment(
        'ntn-iab-quic-dash-ai',
        ns3_root,
        py_binding,
        handleFinish=True,
        useVector=True,
        vectorSize=32,
        shmSize=65536,
    )
    msg = exp.run(setting=setting, show_output=args.show_waf_output)

    tick = 0
    try:
        while True:
            msg.PyRecvBegin()
            if msg.PyGetFinished():
                logging.info('Simulation finished (PyGetFinished)')
                break

            try:
                cpp2py = msg.GetCpp2PyVector()
                n = len(cpp2py)
                logging.info('tick=%d active_connections=%d', tick, n)
                for i in range(n):
                    o = cpp2py[i].obs
                    logging.info(
                        '  [%d] node_id=%s conn_id=%s path_id=%s time_s=%.6f dt_s=%.6f',
                        i, o.node_id, o.conn_id, o.path_id, o.time_s, o.dt_s,
                    )
                    logging.info(
                        '      btlbw_bps=%s rtprop_s=%s pacing_gain=%s cwnd_gain=%s '
                        'pacing_rate_bps=%s cwnd_bytes=%s target_cwnd_bytes=%s bbr_mode=%s',
                        o.btlbw_bps, o.rtprop_s, o.pacing_gain, o.cwnd_gain,
                        o.pacing_rate_bps, o.cwnd_bytes, o.target_cwnd_bytes, o.bbr_mode,
                    )
                    logging.info(
                        '      smoothed_rtt_s=%s latest_rtt_s=%s min_rtt_s=%s '
                        'delivery_rate_bps=%s acked_bytes_over_dt=%s throughput_bps_over_dt=%s '
                        'is_app_limited=%s',
                        o.smoothed_rtt_s, o.latest_rtt_s, o.min_rtt_s,
                        o.delivery_rate_bps, o.acked_bytes_over_dt, o.throughput_bps_over_dt,
                        o.is_app_limited,
                    )
            except Exception:
                logging.exception('Error while reading/logging Cpp2Py vector')
                raise

            msg.PyRecvEnd()

            msg.PySendBegin()
            try:
                py2cpp = msg.GetPy2CppVector()
                fill_dummy_py2cpp_actions(py2cpp, cpp2py)
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    for j in range(len(py2cpp)):
                        d = py2cpp[j]
                        logging.debug(
                            '  Py2Cpp[%d] node=%s conn=%s path=%s '
                            'reord_th=%s/%s t_reord=%s/%s '
                            'p_gain=%s/%s cwnd_gain=%s/%s',
                            j,
                            d.node_id,
                            d.conn_id,
                            d.path_id,
                            d.has_reordering_threshold,
                            d.reordering_threshold,
                            d.has_time_reordering_fraction,
                            d.time_reordering_fraction,
                            d.has_target_pacing_gain,
                            d.target_pacing_gain,
                            d.has_target_cwnd_gain,
                            d.target_cwnd_gain,
                        )
            except Exception:
                logging.exception('Error while writing Py2Cpp dummy actions')
                raise
            msg.PySendEnd()
            tick += 1

    except KeyboardInterrupt:
        logging.info('Interrupted')
    except Exception:
        logging.error(traceback.format_exc())
        return 1
    finally:
        logging.info('Tearing down Experiment')
        del exp

    logging.info('Done (%d ticks)', tick)
    return 0


if __name__ == '__main__':
    sys.exit(main())
