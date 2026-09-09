#!/usr/bin/env bash
# Read-only diagnostics for the local feasibility gate. This script never routes audio.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest=""
model_root=""
inference_python="${OPENDUBSTREAM_INFERENCE_PYTHON:-$repo_root/.venv-inference/bin/python}"

usage() {
  cat <<'EOF'
Usage: ./scripts/diagnose-feasibility.sh [--manifest PATH] [--model-root PATH]

Run as the logged-in desktop user. Reports only; it does not install packages,
download models, move streams, or change any PipeWire/default-audio setting.

By default this uses .venv-inference, $HOME/.local/share/opendubstream/models,
and that model root's assets.sha256. Override defaults with
OPENDUBSTREAM_INFERENCE_PYTHON, OPENDUBSTREAM_MODEL_ROOT, or
OPENDUBSTREAM_MODEL_MANIFEST. Explicit command-line paths take precedence.
EOF
}

require_desktop_user() {
  if [[ "${EUID}" -eq 0 ]]; then
    printf '%s\n' 'ERROR: run this diagnostic as the logged-in desktop user, not root.' >&2
    exit 2
  fi
  if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
    printf '%s\n' 'ERROR: XDG_RUNTIME_DIR is unavailable; start this from the desktop user session.' >&2
    exit 2
  fi
}

command_report() {
  local label="$1"
  shift
  local output
  if output="$("$@" 2>&1)"; then
    printf 'PASS: %s\n%s\n' "$label" "$output"
  else
    printf 'BLOCKED: %s (exit %s)\n%s\n' "$label" "$?" "$output"
  fi
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --manifest)
      [[ "$#" -ge 2 ]] || { printf '%s\n' 'ERROR: --manifest needs a path.' >&2; exit 2; }
      manifest="$2"
      shift 2
      ;;
    --model-root)
      [[ "$#" -ge 2 ]] || { printf '%s\n' 'ERROR: --model-root needs a path.' >&2; exit 2; }
      model_root="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unsupported argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

model_root="${model_root:-${OPENDUBSTREAM_MODEL_ROOT:-$HOME/.local/share/opendubstream/models}}"
manifest="${manifest:-${OPENDUBSTREAM_MODEL_MANIFEST:-$model_root/assets.sha256}}"

require_desktop_user

printf '%s\n' '== PipeWire session access =='
printf 'XDG_RUNTIME_DIR=%s\n' "$XDG_RUNTIME_DIR"
if [[ -S "$XDG_RUNTIME_DIR/pipewire-0" ]]; then
  printf '%s\n' 'PASS: PipeWire socket exists for this user session.'
else
  printf '%s\n' 'BLOCKED: PipeWire socket is absent for this user session.'
fi
if command -v wpctl >/dev/null 2>&1; then
  command_report 'wpctl status (read-only)' wpctl status
else
  printf '%s\n' 'BLOCKED: wpctl is not installed.'
fi
if command -v pactl >/dev/null 2>&1; then
  command_report 'pactl info (read-only)' pactl info
else
  printf '%s\n' 'BLOCKED: pactl is not installed.'
fi

printf '%s\n' '== Chrome/Chromium playback streams =='
found_browser=0
for process_name in google-chrome chrome chromium chromium-browser; do
  if pgrep -a -f -- "(^|/|[[:space:]])${process_name}([[:space:]]|$)"; then
    found_browser=1
  fi
done
if [[ "$found_browser" -eq 0 ]]; then
  printf '%s\n' 'BLOCKED: no exact Chrome/Chromium browser process is running.'
fi
if command -v wpctl >/dev/null 2>&1; then
  stream_snapshot="$(wpctl status 2>&1 || true)"
  if grep -Eiq 'chrome|chromium' <<<"$stream_snapshot"; then
    printf '%s\n' 'Observed PipeWire lines matching Chrome/Chromium:'
    grep -Ei 'chrome|chromium' <<<"$stream_snapshot" || true
  else
    printf '%s\n' 'BLOCKED: no Chrome/Chromium-named PipeWire stream is visible to wpctl.'
  fi
fi
printf '%s\n' 'NOTE: this script does not select or route any stream; selection remains explicit and fail-closed.'

printf '%s\n' '== NVIDIA/CUDA availability =='
if [[ -e /dev/nvidiactl ]]; then
  printf '%s\n' 'PASS: /dev/nvidiactl exists.'
else
  printf '%s\n' 'BLOCKED: /dev/nvidiactl is absent.'
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  command_report 'nvidia-smi -L' nvidia-smi -L
else
  printf '%s\n' 'BLOCKED: nvidia-smi is not installed or not on PATH.'
fi

printf '%s\n' '== Local model, runtime, and asset-pin status =='
if [[ -x "$inference_python" ]]; then
  command_report 'local inference runtime imports' "$inference_python" -c \
    'import importlib.util; names=("torch", "ctranslate2", "faster_whisper", "transformers", "sentencepiece", "silero_vad", "kokoro_onnx", "onnxruntime"); print(" ".join(f"{name}={bool(importlib.util.find_spec(name))}" for name in names))'
else
  printf 'BLOCKED: local inference Python is unavailable: %s\n' "$inference_python"
fi
"$repo_root/scripts/verify-local-model.sh" --manifest "$manifest" --model-root "$model_root"
if [[ -x "$inference_python" ]]; then
  command_report 'Kokoro offline model/voice compatibility (read-only)' "$inference_python" -c '
from pathlib import Path
from kokoro_onnx import Kokoro
import sys

root = Path(sys.argv[1]) / "kokoro-onnx-v1"
kokoro = Kokoro(str(root / "kokoro-v1.0.onnx"), str(root / "voices-v1.0.bin"))
required = {"ef_dora", "em_alex", "em_santa"}
available = set(kokoro.get_voices())
missing = required - available
if missing:
    raise RuntimeError(f"required Kokoro voices are unavailable: {sorted(missing)}")
print("voices=" + ",".join(sorted(required)))
' "$model_root"
fi

printf '%s\n' 'Postcondition: diagnostics completed without package installation, model transfer, audio routing, or global-default changes.'
