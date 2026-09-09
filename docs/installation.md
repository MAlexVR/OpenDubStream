# Install the experimental desktop runtime

The first alpha targets Fedora 44 x86_64. The package and offline tests are not a
clean-machine or real-audio certification. A functioning NVIDIA CUDA driver is required
by the current translation adapter; this installer does not install GPU drivers.

## RPM path

Review and install an available prerelease locally:

```bash
sudo dnf install ./opendubstream-0.1.0~alpha.1-1.noarch.rpm
opendubstream setup
opendubstream setup --apply
opendubstream
```

The RPM requires Python 3.12, PipeWire utilities and PulseAudio-compatible utilities.
It installs application files below `/usr/share/opendubstream` and the launcher in
`/usr/bin`. It does **not** install models or run pip in an RPM install hook. The `noarch`
label describes the Python/source payload, not tested support for every CPU architecture.

Explicit setup downloads pinned direct Python dependencies and approximately **2.55 GB
of model assets**, plus several GB of Python/CUDA wheels and temporary download storage.
Transitive dependencies are not fully locked. Review the upstream terms linked in
[third-party notices](../THIRD_PARTY_NOTICES.md) before provisioning.

Setup runs in the desktop user's session and writes below
`${XDG_DATA_HOME:-$HOME/.local/share}/opendubstream`:

| Directory | Content |
|---|---|
| `runtime/.venv-inference` | Python 3.12 environment with UI and inference dependencies |
| `runtime/scripts`, `runtime/requirements` | Small provisioning inputs |
| `models` | Explicitly downloaded model files and local integrity manifest |

`OPENDUBSTREAM_MODEL_ROOT` can point to an existing model directory. Existing models
must pass their recorded `assets.sha256` verification. Hashes detect local corruption;
they are recorded after download, not a separately trusted upstream signature.

If application-menu launch appears to do nothing before setup, run `opendubstream` in
a terminal for the actionable setup error. Uninstalling the RPM does not remove your
models, runtime or user preferences.

## Source path

```bash
git clone https://github.com/MAlexVR/OpenDubStream.git
cd OpenDubStream
python3.12 scripts/provision-local-models.py \
  --model-root "$HOME/.local/share/opendubstream/models"
# Review the dry run and upstream terms, then opt in:
python3.12 scripts/provision-local-models.py \
  --model-root "$HOME/.local/share/opendubstream/models" --apply
.venv-inference/bin/python -m pip install -r requirements/desktop-ui.txt
PYTHONPATH=src .venv-inference/bin/python -m opendubstream.ui.app
```

The source provisioner refuses to replace an existing model root. Reuse an already
provisioned environment/root rather than deleting a recovery journal or model directory
blindly. System prerequisites include `python3.12`, `pipewire-utils`, `pulseaudio-utils`,
`pipewire-pulseaudio`, `espeak-ng` and Qt's X11/XWayland libraries.
