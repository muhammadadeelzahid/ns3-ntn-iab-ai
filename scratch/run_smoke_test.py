#!/usr/bin/env python3
# Smoke test: Experiment + waf launches ntn-iab-quic-dash-ai; subprocess stays up.
# Pass aiDtMs=0 to disable the ns3-ai tick (no Python recv loop needed for this smoke).
#
# Run from repository root: python3 scratch/run_smoke_test.py
#
# If ./waf fails with env: python: No such file or directory, export:
#   export WAF_CMD="python3 ./waf"

from __future__ import print_function

import os
import shutil
import sys
import time

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
    # Many HPC nodes have no `python` on PATH; waf's shebang then fails unless:
    if 'WAF_CMD' not in os.environ and shutil.which('python') is None:
        os.environ['WAF_CMD'] = 'python3 ./waf'

    ns3_root = _REPO_ROOT
    if not os.path.isfile(os.path.join(ns3_root, 'waf')):
        print('SMOKE: no waf at fixed repo root:', ns3_root, file=sys.stderr)
        return 2

    print('SMOKE: NS3_ROOT=', ns3_root)
    if os.environ.get('WAF_CMD'):
        print('SMOKE: WAF_CMD=', os.environ['WAF_CMD'])

    exp = Experiment(
        'ntn-iab-quic-dash-ai',
        ns3_root,
        py_binding,
        handleFinish=True,
        useVector=True,
        vectorSize=32,
        shmSize=65536,
    )
    setting = {'aiDtMs': '0', 'numUes': '1'}
    print('SMOKE: launching:', setting)
    exp.run(setting=setting, show_output=True)
    time.sleep(1.0)
    if exp.isalive():
        print('SMOKE: OK subprocess still running after 1s')
    else:
        print('SMOKE: FAIL subprocess exited early', file=sys.stderr)
        return 1
    exp.kill()
    print('SMOKE: done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
