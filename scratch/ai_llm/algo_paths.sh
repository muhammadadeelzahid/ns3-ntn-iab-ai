#!/bin/bash
# Shared helpers for DASH trace Slurm scripts (source this file).

# ns-3 scratch binaries need module .so files from the build directory.
export_ns3_runtime() {
  local repo="${1:?NS3_REPO required}"
  export LD_LIBRARY_PATH="${repo}/build${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

# Strip CR from values (job_matrix.csv may have been written with CRLF).
strip_cr() {
  printf '%s' "$1" | tr -d '\r'
}

# ns3::FdashClient -> Fdash, ns3::AiDashClient -> AiDash, ns3::OsmpClient -> Osmp
algo_short_from_ns3() {
  local algo
  algo="$(strip_cr "$1")"
  local base="${algo##*::}"   # drop ns3::
  base="${base%Client}"       # drop Client suffix
  echo "${base}"
}

# Quic_artifacts or Tcp_artifacts from protocol hint
artifacts_root() {
  local protocol="${1:-QUIC}"
  if [[ "${protocol}" == "TCP" ]]; then
    echo "Tcp_artifacts"
  else
    echo "Quic_artifacts"
  fi
}

# DASH trace run directory: <repo>/<artifacts>/<algo>/<video>/run_<seed>
run_artifacts_dir() {
  local repo="${1:?repo required}"
  local artifacts="${2:?artifacts root required}"
  local algo_short="${3:?algo short name required}"
  local video_name="${4:?video name required}"
  local run_seed="${5:?run seed required}"
  echo "${repo}/${artifacts}/${algo_short}/${video_name}/run_${run_seed}"
}
