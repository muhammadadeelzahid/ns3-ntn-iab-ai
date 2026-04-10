#!/usr/bin/env python3
# Phase 1: receive Cpp2Py observations, log fields; Py2Cpp handshake with no policy writes.
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


def main():
    parser = argparse.ArgumentParser(description='QUIC BBR AI: recv-only Python loop with logging')
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
