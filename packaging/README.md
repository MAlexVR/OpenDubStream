# Build the RPM

The owner selected MIT for the project. The spec requires an explicit license parameter
and the build requires the LICENSE file, preserving the release licensing gate.

On Fedora with `rpm-build` installed:

```bash
packaging/build-rpm.sh MIT
rpm -qpi dist/*.rpm
rpm -qpl dist/*.rpm
(cd dist && sha256sum -c SHA256SUMS)
```

Run checksum verification as `(cd dist && sha256sum -c SHA256SUMS)`.
The script uses an isolated temporary rpmbuild root, no downloads or host installs,
and an explicit application payload allowlist. No venv, weight, recording, journal,
benchmark dump or internal agent directory is included. The source payload has stable
ordering/ownership/timestamps; bit-identical RPM output is not promised.

The RPM is unsigned. It installs a runnable launcher and desktop/icon metadata;
`opendubstream setup --apply` explicitly provisions the per-user ML runtime. No scriptlet
runs pip, downloads weights or modifies audio at package installation/removal time.

## Release checklist

- Owner license exists and included notices match the actual source provenance.
- Check `rpm -qp --requires`, `rpm -qp --scripts` and the complete payload listing.
- Extract RPM without installation and smoke-test launcher help/dry-run/missing-runtime.
- Run the full offline suite and record skipped tests honestly.
- Validate a clean Fedora installation before claiming installation support beyond alpha.
- Validate real audio separately before claiming continuous/real-time dubbing.
- Publish as a prerelease with known limits, checksum manifest and unsigned-package notice.
