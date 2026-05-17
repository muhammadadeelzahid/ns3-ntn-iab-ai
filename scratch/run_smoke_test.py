#!/usr/bin/env python3
"""
Smoke test: launches ntn-iab-quic-dash-ai via ns3ai_utils with DASH ABR IPC disabled.

Author: Muhammad Adeel Zahid <adeel.m.zahid@gmail.com>
        University of Manitoba

Added: smoke test for ntn-iab-quic-dash-ai launch (DASH ABR IPC disabled).

Run from repository root: python3 scratch/run_smoke_test.py

If ./waf fails with env: python: No such file or directory, export:
  export WAF_CMD="python3 ./waf"
"""

from __future__ import print_function

import os
import shutil
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for _p in (
    os.path.join(_REPO_ROOT, 'src', 'ai', 'python_utils'),
    os.path.join(_REPO_ROOT, 'build', 'src', 'ai', 'examples', 'abr-dash-ai', 'use-msg'),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ns3ai_abr_dash_ai_py as py_binding  # noqa: E402
from ns3ai_utils import Experiment  # noqa: E402


def main():
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
        useVector=False,
        shmSize=4096,
    )
    setting = {'aiEnable': 'false', 'numUes': '1', 'algorithm': 'ns3::FdashClient'}
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
