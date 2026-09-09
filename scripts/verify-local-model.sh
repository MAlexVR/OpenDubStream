#!/usr/bin/env bash
# Verify a user-supplied local asset manifest. This script has no acquisition path.
set -euo pipefail

manifest=""
model_root=""

usage() {
  cat <<'EOF'
Usage: ./scripts/verify-local-model.sh --manifest PATH --model-root PATH

Run as the logged-in desktop user. The manifest uses sha256sum format:
<64 lowercase/uppercase hexadecimal characters><two spaces><relative asset path>

Verified relative paths must include all local runtime asset families:
`silero-vad`, `faster-whisper-distil-large-v3`, `opus-mt-en-es`, and
`kokoro-onnx-v1/kokoro-v1.0.onnx` and `kokoro-onnx-v1/voices-v1.0.bin`. The helper only reads the provided files and validates
hashes; it has no transfer logic.
EOF
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

if [[ "${EUID}" -eq 0 ]]; then
  printf '%s\n' 'ERROR: run local model verification as the logged-in desktop user, not root.' >&2
  exit 2
fi
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
  printf '%s\n' 'ERROR: XDG_RUNTIME_DIR is unavailable; start this from the desktop user session.' >&2
  exit 2
fi
if [[ -z "$manifest" || -z "$model_root" ]]; then
  printf '%s\n' 'ERROR: both --manifest and --model-root are required.' >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$manifest" || ! -r "$manifest" ]]; then
  printf 'ERROR: manifest is not a readable file: %s\n' "$manifest" >&2
  printf '%s\n' 'ERROR: required local model asset pin is absent (Silero VAD, faster-whisper distil-large-v3, OPUS-MT en→es, and Kokoro v1 model plus voices are all mandatory).' >&2
  exit 2
fi
if [[ ! -d "$model_root" || ! -r "$model_root" ]]; then
  printf 'ERROR: model root is not a readable directory: %s\n' "$model_root" >&2
  exit 2
fi

pin_count=0
silero_found=0
asr_found=0
opus_mt_found=0
kokoro_found=0
kokoro_model_found=0
kokoro_voices_found=0
line_number=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line_number=$((line_number + 1))
  [[ -z "$line" || "${line:0:1}" == '#' ]] && continue
  if [[ ! "$line" =~ ^([[:xdigit:]]{64})\ \ (.+)$ ]]; then
    printf 'ERROR: invalid manifest entry at line %s. Expected SHA-256, two spaces, then a relative path.\n' "$line_number" >&2
    exit 1
  fi
  expected="${BASH_REMATCH[1],,}"
  relative_path="${BASH_REMATCH[2]}"
  if [[ "$relative_path" == /* || "$relative_path" == '..' || "$relative_path" == ../* || "$relative_path" == */../* ]]; then
    printf 'ERROR: manifest path escapes the model root at line %s.\n' "$line_number" >&2
    exit 1
  fi
  asset="$model_root/$relative_path"
  if [[ ! -f "$asset" || ! -r "$asset" ]]; then
    printf 'ERROR: pinned asset is missing or unreadable: %s\n' "$relative_path" >&2
    exit 1
  fi
  actual="$(sha256sum -- "$asset")"
  actual="${actual%% *}"
  if [[ "$actual" != "$expected" ]]; then
    printf 'ERROR: hash mismatch for pinned asset: %s\n' "$relative_path" >&2
    exit 1
  fi
  pin_count=$((pin_count + 1))
  if [[ "$relative_path" == *opus-mt-en-es* ]]; then
    opus_mt_found=1
  fi
  if [[ "$relative_path" == *silero-vad* ]]; then
    silero_found=1
  fi
  if [[ "$relative_path" == *faster-whisper-distil-large-v3* ]]; then
    asr_found=1
  fi
  if [[ "$relative_path" == "kokoro-onnx-v1/kokoro-v1.0.onnx" ]]; then
    kokoro_model_found=1
  fi
  if [[ "$relative_path" == "kokoro-onnx-v1/voices-v1.0.bin" ]]; then
    kokoro_voices_found=1
  fi
  if [[ "$kokoro_model_found" -eq 1 && "$kokoro_voices_found" -eq 1 ]]; then
    kokoro_found=1
  fi
done < "$manifest"

if [[ "$pin_count" -eq 0 ]]; then
  printf '%s\n' 'ERROR: manifest has no asset pins.' >&2
  exit 1
fi
if [[ "$silero_found" -ne 1 || "$asr_found" -ne 1 || "$opus_mt_found" -ne 1 || "$kokoro_found" -ne 1 ]]; then
  printf '%s\n' 'ERROR: required local model asset pin is absent (Silero VAD, faster-whisper distil-large-v3, OPUS-MT en→es, and Kokoro v1 model plus voices are all mandatory).' >&2
  exit 1
fi

printf 'VERIFIED: required local model assets and %s pinned local file(s).\n' "$pin_count"
printf '%s\n' 'Postcondition: supplied local assets were only read and hash-verified.'
