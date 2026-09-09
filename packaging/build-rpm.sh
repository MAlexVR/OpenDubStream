#!/usr/bin/env bash
# Build from an explicit payload allowlist, never a recursive repository archive.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -ne 1 || -z $1 ]]; then
  echo 'Usage: packaging/build-rpm.sh <owner-approved-SPDX-license>' >&2
  echo 'Publication is blocked until the owner selects a project-wide license.' >&2
  exit 2
fi
license=$1
# Reject macro or shell-like metacharacters in RPM metadata.
if [[ ! $license =~ ^[A-Za-z0-9.+-]+$ ]]; then
  echo 'Expected a single owner-approved SPDX identifier.' >&2
  exit 2
fi
if [[ ! -f "$root/LICENSE" ]]; then
  echo "Owner-approved LICENSE file is missing; no RPM will be built." >&2
  exit 2
fi
command -v rpmbuild >/dev/null
work=$(mktemp -d "${TMPDIR:-/tmp}/opendubstream-rpm.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
version='0.1.0~alpha.1'
stage="$work/opendubstream-$version"
mkdir -p "$stage" "$work/rpmbuild/SOURCES" "$root/dist"
for tree in src requirements; do
  mkdir -p "$stage/$tree"
  while IFS= read -r -d '' file; do
    install -Dm644 "$file" "$stage/${file#"$root/"}"
  done < <(find "$root/$tree" -type f \( -name '*.py' -o -name '*.txt' \) -print0)
done
for file in README.md THIRD_PARTY_NOTICES.md LICENSES/BluCast-MIT.txt \
  assets/LICENSES/CC0-1.0.txt assets/OPENDUBSTREAM-ICON-PROVENANCE.md \
  assets/opendubstream.svg assets/opendubstream.desktop \
  docs/installation.md docs/troubleshooting.md packaging/opendubstream.py \
  packaging/build-rpm.sh packaging/opendubstream.spec packaging/README.md pyproject.toml \
  scripts/provision-local-models.py scripts/local-model-sources.json scripts/verify-local-model.sh; do
  install -Dm644 "$root/$file" "$stage/$file"
done
# Preserve the verifier's executable contract used by the provisioner.
chmod 755 "$stage/scripts/verify-local-model.sh" "$stage/packaging/build-rpm.sh"
if [[ -f "$root/LICENSE" ]]; then
  install -Dm644 "$root/LICENSE" "$stage/LICENSE"
fi
# Stable order, owner and timestamps. Repeatable source payload (RPM metadata may differ).
tar --sort=name --mtime='UTC 2026-09-08' --owner=0 --group=0 --numeric-owner \
  -czf "$work/rpmbuild/SOURCES/opendubstream-$version.tar.gz" -C "$work" "opendubstream-$version"
rpmbuild -bb "$root/packaging/opendubstream.spec" \
  --define "_topdir $work/rpmbuild" --define "_tmppath $work" --define "_buildhost opendubstream-build" --define "project_license $license" --define 'dist %{nil}'
cp "$work/rpmbuild/RPMS/noarch/"*.rpm "$root/dist/"
cp "$work/rpmbuild/SOURCES/"*.tar.gz "$root/dist/"
(cd "$root/dist" && sha256sum -- *.rpm *.tar.gz > SHA256SUMS)
printf 'Artifacts: %s/dist (unsigned; verify payload before publication)\n' "$root"
