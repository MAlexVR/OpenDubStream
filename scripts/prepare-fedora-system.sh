#!/usr/bin/env bash
# Explicit Fedora package preparation only. It never configures routing or GPU drivers.
set -euo pipefail

apply=0
global_defaults=0
global_confirmation=0
readonly packages=(pipewire wireplumber pipewire-pulseaudio pipewire-utils)

usage() {
  cat <<'EOF'
Usage: pkexec ./scripts/prepare-fedora-system.sh [--apply]

Root-only Fedora preparation. Dry-run is the default; --apply is required to
install the listed Fedora packages using only the official fedora and updates
repositories. Run it explicitly through pkexec from a desktop session.

This script never enables external repositories, installs NVIDIA drivers,
downloads ML models, routes audio, or changes any global default.

--enable-global-defaults --confirm-global-defaults is deliberately refused:
global-default changes are outside this script's safe scope.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --apply)
      apply=1
      shift
      ;;
    --enable-global-defaults)
      global_defaults=1
      shift
      ;;
    --confirm-global-defaults)
      global_confirmation=1
      shift
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

if [[ "${EUID}" -ne 0 ]]; then
  printf '%s\n' 'ERROR: root is required. Launch explicitly with: pkexec ./scripts/prepare-fedora-system.sh' >&2
  exit 2
fi
if [[ "$global_defaults" -eq 1 || "$global_confirmation" -eq 1 ]]; then
  printf '%s\n' 'ERROR: global-default changes are intentionally unsupported by this script.' >&2
  exit 2
fi
if [[ ! -f /etc/fedora-release ]]; then
  printf '%s\n' 'ERROR: this preparation script supports Fedora only.' >&2
  exit 2
fi
if ! command -v dnf >/dev/null 2>&1; then
  printf '%s\n' 'ERROR: dnf is required on Fedora.' >&2
  exit 2
fi

printf '%s\n' '== Supported Fedora system preparation =='
printf '%s\n' 'Packages: pipewire, wireplumber, pipewire-pulseaudio, pipewire-utils'
printf '%s\n' 'Postconditions after a successful apply:'
printf '%s\n' '  - package binaries are available from Fedora repositories only;'
printf '%s\n' '  - no audio stream/default is selected or changed;'
printf '%s\n' '  - no NVIDIA driver, ML model, or third-party repository is installed.'
printf '%s\n' 'The logged-in desktop user must then start Chrome playback and run diagnose-feasibility.sh.'
printf '%s\n' 'WARNING: if you separately install an NVIDIA driver, a reboot may be required. With Secure Boot, the driver may also require MOK enrollment and a reboot to complete it.'

if [[ "$apply" -eq 0 ]]; then
  printf '%s\n' 'DRY-RUN: no package operation will be performed. Re-run with --apply to install only the listed packages.'
  exit 0
fi

dnf install -y --best --disablerepo='*' --enablerepo=fedora --enablerepo=updates "${packages[@]}"
printf '%s\n' 'APPLIED: supported Fedora packages were installed. Re-run the user-session diagnostic before attempting the feasibility spike.'
