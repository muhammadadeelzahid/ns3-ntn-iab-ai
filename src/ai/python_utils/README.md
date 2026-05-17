# ns3ai_python_utils

Installs the `ns3ai_utils` module (`Experiment`, `run_single_simulation`) used by ns3-ai Python drivers.

## Install (recommended)

From the ns-3 repository root:

```bash
python3 -m pip install --user -e src/ai/python_utils
```

Use a venv or `pip install --user` if you cannot write to the system site-packages.

## Without pip: `PYTHONPATH`

```bash
# Replace /your/clone with the real path to this repository (do not use a literal /path/to/...).
export PYTHONPATH="/your/clone/ns3-ntn-iab/src/ai/python_utils${PYTHONPATH:+:$PYTHONPATH}"
python3 your_driver.py
```

`Experiment.kill()` tears down the simulation via `killpg` (waf runs in its own process group).

## Smoke test

After `./waf build`, from the repository root:

```bash
python3 scratch/run_smoke_test.py
```

The scratch runners use a fixed layout: repo root is the parent of `scratch/` (where `./waf` lives).

Uses `--aiEnable=false` with native `ns3::FdashClient` so no Python ABR loop is required. If `python` is not on `PATH`, the script sets `WAF_CMD=python3 ./waf` automatically.

The runner in `scratch/` adds `src/ai/python_utils` and `build/src/ai/examples/abr-dash-ai/use-msg` (pybind `.so`) to `sys.path`, so you do not need manual `PYTHONPATH` for those when using the standard layout.

## DASH ABR orchestrator

From the repository root:

```bash
python3 scratch/run_ai_dash_heuristic.py
```

It launches `ntn-iab-quic-dash-ai` with `ns3::AiDashClient`, then loops on shared-memory ABR state/actions (`AbrDashCpp2PyMsg` / `AbrDashPy2CppMsg`) until the simulation finishes.
