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

### Process-lifecycle guarantees

To avoid orphaned `python3 ./waf --run ...` and ns-3 scratch binaries lingering at 100% CPU after the orchestrator exits, the launcher combines several layers of cleanup:

1. The waf subprocess is started in its own session (`os.setsid()`), so `killpg` reliably terminates the whole subtree (waf + ns-3 binary + any helpers).
2. Linux `PR_SET_PDEATHSIG` is set on the waf child (here) and on the ns-3 binary (in `wutils.py`). If the Python orchestrator dies for any reason - including `SIGKILL`, SSH disconnect, or a hard crash - the kernel sends `SIGTERM` to waf, which propagates similarly to the ns-3 binary.
3. `atexit` and `SIGINT`/`SIGTERM`/`SIGHUP` handlers in `ns3ai_utils` call `Experiment.kill()` before the interpreter exits, then raise `KeyboardInterrupt` (for `SIGINT`) or `SystemExit` so caller `finally:` blocks still run.
4. `Experiment.kill()` escalates `SIGTERM` -> wait -> `SIGKILL` against the whole process group.
5. The DASH ABR orchestrators check `exp.proc.poll()` at the top of every loop iteration so they bail out instead of hanging in `PyRecvBegin` after an unexpected ns-3 exit.

### Manual cleanup verification

After applying changes, validate with:

```bash
# Helper: list any leftover waf/ns-3 processes
pgrep -af 'waf --run|ntn-iab-quic-dash-ai'
```

Then run each scenario and re-run the helper after the orchestrator exits - it should print nothing.

- Ctrl+C while in the main loop:
  ```bash
  python3 scratch/run_ai_dash_llm.py
  # ... wait until "[Python] LLM ABR orchestrator started." ...
  # press Ctrl+C
  ```
- Normal completion (let the simulation finish on its own).
- Hard kill of the orchestrator (PR_SET_PDEATHSIG path):
  ```bash
  python3 scratch/run_ai_dash_heuristic.py &
  ORCH_PID=$!
  sleep 30
  kill -9 $ORCH_PID
  ```
- ns-3 crash / early exit (e.g. wrong argument); the orchestrator should print "ns-3 subprocess exited (rc=...)" and exit.

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
