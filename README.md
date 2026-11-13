# Adaptive Video Streaming for Satellite backhauled 5G IAB Network

This repository is an **ns-3 codebase** that combines **Non-Terrestrial Networks (NTN)** and **Integrated Access and Backhaul (IAB)** on top of the ns-3 **mmWave** module, and includes end-to-end application experiments with **MPEG-DASH video streaming** over **TCP** and **QUIC**.

Developed at the **University of Manitoba**. 

## Modules and upstream code integrated

This codebase extends the ns-3 **mmWave** stack by integrating functionality from the following upstream modules:

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

Native C++ baseline (no shared-memory IPC):

```bash
python3 ./waf --run "ntn-iab-quic-dash-ai --aiEnable=false --numUes=1 --run=1 --algorithm=ns3::FdashClient"
```

### DASH trace extraction (Slurm batch)

Full pipeline (metadata → sims → training traces → oracle → `train_jsonl/`): **[docs/end-to-end-abr-pipeline.md](docs/end-to-end-abr-pipeline.md)**.

Slurm details, env vars, troubleshooting: **[docs/dash-trace-pipeline.md](docs/dash-trace-pipeline.md)**.

```bash
export METADATA_DIR=/project/6030214/adeel/abr/output/metadata
sbatch dash_trace_smoke.slurm          # single end-to-end check
sbatch dash_trace_launch.slurm         # batch sim array
# after workers:
sbatch extract_traces.slurm            # trace_decisions.jsonl
sbatch extract_throughput_traces.slurm   # throughput_trace.jsonl (oracle)
sbatch dp_oracle.slurm
```

## Maintainers

- **Muhammad Adeel Zahid** (`adeel.m.zahid@gmail.com`)
- **WiCoNS Research Group** (University of Manitoba): [website](https://home.cc.umanitoba.ca/~hossaina/wicons/index.html)
- **Advancing LEO Satellite Networks and Systems Group** (University of Manitoba): [website](https://home.cc.umanitoba.ca/~hup2/ntn/)
- **Advanced Network and Embedded Systems (ANES)** (University of Manitoba)

## License

This software is licensed under the terms of the **GNU GPLv2**, consistent with ns-3. See the [`LICENSE`](LICENSE) file for details.
