# Adaptive Video Streaming for Satellite backhauled 5G IAB Network

This repository builds on **[ns3-ntn-iab](https://github.com/muhammadadeelzahid/ns3-ntn-iab)**, our ns-3
testbed for **MPEG-DASH video streaming** over a **satellite-backhauled 5G Integrated Access and Backhaul
(IAB)** network. That testbed combines Non-Terrestrial Networks (NTN) and IAB on top of the ns-3 mmWave
module and runs DASH over TCP and QUIC. It is described in the following paper; please cite it if you use
this code:

> M. A. Zahid, E. Hossain, and P. Hu, "End-to-End Performance of Video Streaming With MPEG-DASH Over
> Satellite 5G IAB Networks," arXiv:2604.16634, 2026. [https://arxiv.org/abs/2604.16634](https://arxiv.org/abs/2604.16634)

On top of that testbed, this repository adds:

- **Application-layer ABR control through [ns3-ai](https://github.com/hust-diangroup/ns3-ai):** the
  `ns3::AiDashClient` DASH client and the `ntn-iab-quic-dash-ai` scenario, where an external Python process
  chooses the bitrate of every segment over shared memory.
- **An LLM-assisted ABR policy** (`scratch/ai_llm/`): a small language model (Qwen2.5-1.5B-Instruct with
  LoRA adapters) that chooses bitrates from network state and video content, with its training and
  evaluation code.
- **Baseline ABR algorithms:** Quality-MPC, RobustMPC and Pensieve served through the same bridge, and a
  native `ns3::BolaClient`.
- **QUIC and DASH fixes for long closed-loop sessions:** ACK handling and retransmission after timeouts in
  QUIC; recovery of stalled segment downloads, a reconnect watchdog and a 30 s playout-buffer cap in DASH.
- **Simulation options** for ABR experiments: a configurable feeder-link rate (`--linkRate`) and a
  beamforming-cache lifetime attribute (`ns3::MmWaveFFTCodebookBeamforming::CacheExpiry`).

Developed at the **University of Manitoba**.

## Modules and upstream code integrated

The base testbed extends the ns-3 **mmWave** stack by integrating functionality from the following upstream
modules:

- **Integrated Access and Backhaul (IAB)**: based on `ns3-mmwave-iab` from the [ns3-mmwave](https://github.com/nyuwireless-unipd/ns3-mmwave)
- **Non-Terrestrial Networks (NTN)**: from [ns-3-ntn](https://gitlab.com/mattiasandri/ns-3-ntn)
- **Multilayer base stations and Hybrid Beamforming (HBF)**: from [ns3-mmwave-hbf](https://github.com/signetlabdei/ns3-mmwave-hbf)
- **Traffic generators (NR/5G-LENA)**: from [5G-LENA](https://cttc-lena.gitlab.io/nr/html/)
- **QUIC**: the ns-3 QUIC implementation in this tree is derived from the `quic-ns-3` lineage (see `src/quic/`)
- **MPEG-DASH (HTTP Adaptive Streaming)**: the `dash` module integrated in this tree (`src/dash/`) is based on [djvergad/dash](https://github.com/djvergad/dash)

## Build

This repository uses the **waf** build system (the classic ns-3 workflow).

Configure and build:

```bash
python3 ./waf configure --enable-examples
python3 ./waf build
```

If your system already provides a `python` executable (or you installed it via something like `python-is-python3`), the following also works:

```bash
./waf configure --enable-examples
./waf build
```

## Run

Run any of the `scratch/` programs using `--run` (examples below):

```bash
python3 ./waf --run "ntn-iab"
```

MPEG-DASH video streaming experiments:

```bash
python3 ./waf --run "ntn-iab-tcp-dash"
python3 ./waf --run "ntn-iab-quic-dash"
```

You can pass program arguments after `--run`, for example:

```bash
python3 ./waf --run "ntn-iab-quic-dash --numUes=10 --numRelay=1 --run=1"
```

### DASH ABR with ns3-ai (application layer)

Python-controlled adaptive bitrate for QUIC-DASH (`ntn-iab-quic-dash-ai`, `ns3::AiDashClient`):

```bash
python3 ./waf configure --enable-examples
python3 ./waf build
python3 scratch/run_ai_dash_heuristic.py
```

Native C++ ABR clients (no shared-memory IPC), FDASH or BOLA:

```bash
python3 ./waf --run "ntn-iab-quic-dash-ai --aiEnable=false --numUes=1 --run=1 --algorithm=ns3::FdashClient"
python3 ./waf --run "ntn-iab-quic-dash-ai --aiEnable=false --numUes=3 --run=1 --algorithm=ns3::BolaClient"
```

Main options of `ntn-iab-quic-dash-ai`: `--numUes`, `--numRelay`, `--linkRate` (feeder-link rate, for
example `18Mbps`; default `100Gb/s`, uncongested), `--algorithm`, `--ccAlgorithm` (default `ns3::QuicBbr`),
`--simStopTime`, `--run` (RNG seed) and `--dashMetadataCsv` (per-segment sizes and quality of the encoded
video). The satellite pass ends at about 151 s, so keep `--simStopTime` at or below 150 s.

### LLM-assisted ABR and baselines

`scratch/ai_llm/` holds an ABR policy built on a small language model (Qwen2.5-1.5B-Instruct with LoRA
adapters), its training and evaluation code, and the baseline ABR algorithms it is compared with:

| Algorithm | How it runs |
| --- | --- |
| LLM policy | `run_ai_dash_llm.py --policy model --checkpoint <ckpt>` (GPU recommended) |
| Quality-MPC | `run_ai_dash_llm.py --policy qmpc` (RobustMPC controller with a VMAF-based QoE objective) |
| RobustMPC | `run_ai_dash_llm.py --policy mpc` |
| Pensieve | `run_ai_dash_llm.py --policy pensieve --checkpoint <ckpt>` (training code in `scratch/ai_llm/pensieve/`) |
| BOLA | native ns-3 client, `--algorithm=ns3::BolaClient` (see above) |

Two Python environments are needed. The ns3-ai binding is built for Python 3.6, so
`scratch/run_ai_dash_llm.py` runs under the system `python3` and drives ns-3; the policy runs in a separate
Python 3.11 environment (`scratch/ai_llm/inference_server.py`, started by the runner), which has PyTorch and
Transformers:

```bash
/usr/bin/python3.11 -m venv ~/abr-llm-venv && source ~/abr-llm-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r scratch/ai_llm/requirements-llm.txt
```

Build ns-3 first (`python3 ./waf build`); the runner then starts ns-3 and the policy server for one session:

```bash
python3 scratch/run_ai_dash_llm.py --policy qmpc \
  --metadata <video>_segments.csv --numUes 3 --run 1 --link-rate 18Mbps --sim-stop-time 150 \
  --server-python ~/abr-llm-venv/bin/python
```

Per-decision records are written to `Quic_artifacts/AiDash/<video>/run_<run>/decisions.jsonl`
(`--decisions-out` to change). Training: `scratch/ai_llm/train.py` (Slurm wrapper
`scratch/ai_llm/slurm/train_llm.sbatch`). Trained checkpoints and the encoded video library are not
included in the repository.

### DASH trace extraction (Slurm batch)

Slurm details, env vars, troubleshooting: **[docs/dash-trace-pipeline.md](docs/dash-trace-pipeline.md)**.

```bash
export METADATA_DIR=/project/6030214/adeel/abr/output/metadata
sbatch dash_trace_launch.slurm         # batch sim array
```

## Maintainers

- **Muhammad Adeel Zahid** (`adeel.m.zahid@gmail.com`)
- **WiCoNS Research Group** (University of Manitoba): [website](https://home.cc.umanitoba.ca/~hossaina/wicons/index.html)
- **Advanced Network and Embedded Systems Lab** (University of Manitoba): [website](https://home.cc.umanitoba.ca/~hup2/ntn/)

## License

This software is licensed under the terms of the **GNU GPLv2**, consistent with ns-3. See the [`LICENSE`](LICENSE) file for details.
