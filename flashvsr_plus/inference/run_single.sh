#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash inference/run_single.sh INPUT OUTPUT HEIGHT WIDTH [MAX_FRAMES] [auto|serial|parallel] [turbo|base|v1]" >&2
  echo "Runs a stream checkpoint from ckpts/turbo, ckpts/base, or ckpts/v1." >&2
}

if [[ ! -f inference/run_stream.py ]]; then
  echo "Run this script from the flashvsr_plus root directory." >&2
  exit 1
fi

if [[ $# -lt 4 || $# -gt 7 ]]; then
  usage
  exit 2
fi

input=$1
output=$2
height=$3
width=$4
max_frames=${5:-}
decoder_mode=${6:-auto}
mode=${7:-turbo}

if [[ ! -f "$input" ]]; then
  echo "Input video does not exist: $input" >&2
  exit 2
fi
if [[ ! "$height" =~ ^[1-9][0-9]*$ || ! "$width" =~ ^[1-9][0-9]*$ ]]; then
  echo "HEIGHT and WIDTH must be positive integers." >&2
  exit 2
fi
if (( height % 128 != 0 || width % 128 != 0 )); then
  echo "HEIGHT and WIDTH must be divisible by 128; got ${height}x${width}." >&2
  exit 2
fi
if [[ -n "$max_frames" && ! "$max_frames" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_FRAMES must be a positive integer." >&2
  exit 2
fi
if [[ "$decoder_mode" != auto && "$decoder_mode" != parallel && "$decoder_mode" != serial ]]; then
  echo "Decoder mode must be 'auto', 'serial', or 'parallel'." >&2
  exit 2
fi
if [[ "$mode" != turbo && "$mode" != base && "$mode" != v1 ]]; then
  echo "Mode must be 'turbo', 'base', or 'v1'." >&2
  exit 2
fi

args=(
  --input "$input"
  --output "$output"
  --height "$height"
  --width "$width"
  --mode "$mode"
)
if [[ -n "$max_frames" ]]; then
  args+=(--max-frames "$max_frames")
fi
if [[ "$decoder_mode" == serial ]]; then
  args+=(--serial-tiny-decoder)
elif [[ "$decoder_mode" != auto ]]; then
  args+=(--tiny-decoder-mode "$decoder_mode")
fi

python_bin=${PYTHON_BIN:-python}
exec "$python_bin" -m inference.run_stream "${args[@]}"
