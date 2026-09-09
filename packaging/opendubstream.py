#!/usr/bin/python3
"""RPM launcher and explicit per-user runtime provisioning (stdlib only)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenDubStream experimental desktop app")
    parser.add_argument("command", nargs="?", choices=["setup", "run"], default="run")
    parser.add_argument("--apply", action="store_true", help="allow setup downloads and user-data writes")
    args = parser.parse_args(argv)
    share = Path(os.environ.get("OPENDUBSTREAM_SHARE", "/usr/share/opendubstream"))
    data = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "opendubstream"
    runtime = data / "runtime"
    python = runtime / ".venv-inference/bin/python"
    models = Path(os.environ.get("OPENDUBSTREAM_MODEL_ROOT", str(data / "models")))
    try:
        if args.command == "setup":
            if not args.apply:
                print(f"Dry run: runtime {runtime}; models {models}.")
                print("Setup downloads pinned Python dependencies and ~2.55 GB models (plus several GB wheels).")
                print("Requires Python 3.12 and configured NVIDIA CUDA driver. Review upstream model licenses first.")
                print("Run opendubstream setup --apply explicitly to proceed. No changes made.")
                return 0
            if os.geteuid() == 0:
                raise RuntimeError("Run setup as the desktop user, not root.")
            if shutil.which("python3.12") is None:
                raise RuntimeError("Python 3.12 is required; install the Fedora python3.12 package first.")
            # The existing provisioner writes relative to its project root. Copy only its
            # small inputs into a per-user workspace; never write below /usr/share.
            for relative in ("scripts/provision-local-models.py", "scripts/local-model-sources.json",
                             "scripts/verify-local-model.sh", "requirements/local-inference.txt",
                             "requirements/desktop-ui.txt"):
                destination = runtime / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(share / relative, destination)
            if not models.exists():
                subprocess.run(["python3.12", str(runtime / "scripts/provision-local-models.py"),
                                "--model-root", str(models), "--python", "python3.12", "--apply"], check=True)
            else:
                # Reuse user assets only after verifying their recorded hashes.
                subprocess.run(["bash", str(runtime / "scripts/verify-local-model.sh"),
                                "--manifest", str(models / "assets.sha256"), "--model-root", str(models)], check=True)
                if not python.exists():
                    subprocess.run(["python3.12", "-m", "venv", str(runtime / ".venv-inference")], check=True)
                subprocess.run([str(python), "-m", "pip", "install", "-r", str(runtime / "requirements/local-inference.txt")], check=True)
            subprocess.run([str(python), "-m", "pip", "install", "-r", str(runtime / "requirements/desktop-ui.txt")], check=True)
            print("Setup complete. Run opendubstream from a desktop terminal.")
            return 0
        if args.apply:
            raise RuntimeError("--apply is only valid with setup.")
        if not python.is_file():
            raise RuntimeError("Runtime missing. In a desktop terminal run: opendubstream setup --apply")
        env = {**os.environ, "PYTHONPATH": str(share / "src"), "OPENDUBSTREAM_MODEL_ROOT": str(models)}
        os.execve(str(python), [str(python), "-m", "opendubstream.ui.app"], env)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"OpenDubStream: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
