#!/usr/bin/env python3
"""
Verify that the AbrDash ns3-ai pybind module builds and imports correctly.

Author: Muhammad Adeel Zahid <adeel.m.zahid@gmail.com>
        University of Manitoba

Added: import check for ns3ai_abr_dash_ai_py pybind module.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('src/ai/python_utils'))
sys.path.insert(0, os.path.abspath('build/src/ai/examples/abr-dash-ai/use-msg'))

try:
    import ns3ai_abr_dash_ai_py
    print("SUCCESS: ns3ai_abr_dash_ai_py imported!")
    print(dir(ns3ai_abr_dash_ai_py))
except ImportError as e:
    print("FAILED to import: ", e)
