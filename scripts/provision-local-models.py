#!/usr/bin/env python3
"""Explicit local-only model provisioning for the feasibility spike.

The normal path is a dry run. Only a user who supplies --apply can create the
dedicated .venv-inference runtime, transfer assets, write a SHA-256 manifest,
and atomically activate a previously absent model root. The repository .venv
is reserved for project tooling and tests. The script never invokes a model or
configures an inference service; consumers must load the generated local paths
with their own offline-only settings after provisioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "scripts" / "local-model-sources.json"
RUNTIME_REQUIREMENTS = REPO_ROOT / "requirements" / "local-inference.txt"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify-local-model.sh"
ASSET_MANIFEST_NAME = "assets.sha256"
ACTIVATION_METADATA_NAME = "provisioned-models.json"
SUPPORTED_PYTHON = {(3, 12), (3, 13)}


class ProvisioningError(RuntimeError):
    """The explicit provisioning request is unsafe or incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision pinned local VAD, ASR, MT, and Spanish TTS assets.",
    )
    parser.add_argument("--model-root", required=True, help="new absolute destination directory")
    parser.add_argument(
        "--python",
        default="python3.12",
        help="Python 3.12 or 3.13 executable used to bootstrap .venv-inference",
    )
    parser.add_argument("--apply", action="store_true", help="perform the explicit bootstrap and downloads")
    parser.add_argument("--internal-apply", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_apply and not args.apply:
        parser.error("--internal-apply requires --apply")
    return args


def load_sources() -> dict[str, object]:
    try:
        document = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisioningError(f"cannot read source manifest: {error}") from error
    if document.get("schema_version") != 1 or not isinstance(document.get("assets"), list):
        raise ProvisioningError("source manifest has an unsupported schema")
    return document


def validated_model_root(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ProvisioningError("--model-root must be an absolute path")
    root = candidate.resolve(strict=False)
    if root == Path("/") or root == REPO_ROOT:
        raise ProvisioningError("--model-root may not be the filesystem or repository root")
    if root.exists():
        raise ProvisioningError(f"refusing to replace an existing model root: {root}")
    return root


def require_desktop_user() -> None:
    if os.geteuid() == 0:
        raise ProvisioningError("run local model provisioning as the logged-in desktop user, not root")
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime_dir or not Path(runtime_dir).is_dir():
        raise ProvisioningError("XDG_RUNTIME_DIR is unavailable; start this from the desktop user session")


def print_dry_run(root: Path, sources: dict[str, object], python_command: str) -> None:
    estimated = int(sources["estimated_download_bytes"])
    print("DRY-RUN: no virtual environment, download, or model-root change was made.")
    print(f"Target model root: {root}")
    print(f"Runtime bootstrap: .venv-inference via {python_command}; pinned requirements: {RUNTIME_REQUIREMENTS}")
    print(f"Estimated asset download: about {estimated / 1_000_000_000:.2f} GB (plus Python wheels).")
    for asset in sources["assets"]:
        print(f"  - {asset['name']}: revision {asset['revision']} -> {asset['destination']}")
    print("Activation will occur only after every downloaded file is SHA-256 hashed and verified.")
    print(f"To apply: {Path(sys.argv[0]).resolve()} --model-root {root} --apply")


def ensure_venv(python_command: str) -> Path:
    venv = REPO_ROOT / ".venv-inference"
    interpreter = venv / "bin" / "python"
    if not interpreter.exists():
        subprocess.run([python_command, "-m", "venv", str(venv)], check=True)
    version = subprocess.check_output(
        [str(interpreter), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
    ).strip()
    major, minor = map(int, version.split("."))
    if (major, minor) not in SUPPORTED_PYTHON:
        raise ProvisioningError(
            f".venv-inference uses Python {version}; provision with Python 3.12 or 3.13 to use the pinned inference runtime"
        )
    subprocess.run(
        [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "--requirement", str(RUNTIME_REQUIREMENTS)],
        check=True,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    patch_activate_for_cuda_library_path(venv)
    return interpreter


_CUDA_LIBRARY_PATH_MARKER = "_OLD_VIRTUAL_LD_LIBRARY_PATH"
_CUDA_LIBRARY_PATH_BLOCK = """
# onnxruntime-gpu and ctranslate2 dlopen CUDA libs at runtime but do not search
# pip-installed nvidia/*/lib packages on their own; without this they silently
# fall back to CPU (onnxruntime) or fail to load cuDNN (ctranslate2). See
# benchmarks/feasibility-2026-09-03.md.
_OLD_VIRTUAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
_NVIDIA_PIP_LIBS="$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"
LD_LIBRARY_PATH="$_NVIDIA_PIP_LIBS:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH
unset _NVIDIA_PIP_LIBS
"""


def patch_activate_for_cuda_library_path(venv: Path) -> None:
    """Idempotently append the LD_LIBRARY_PATH export to a freshly created venv's activate script."""
    activate = venv / "bin" / "activate"
    if not activate.is_file():
        return
    contents = activate.read_text(encoding="utf-8")
    if _CUDA_LIBRARY_PATH_MARKER in contents:
        return
    activate.write_text(contents + _CUDA_LIBRARY_PATH_BLOCK, encoding="utf-8")


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_assets(stage: Path, sources: dict[str, object]) -> None:
    for asset in sources["assets"]:
        destination = stage / asset["destination"]
        if asset["kind"] == "url":
            download_url(asset["url"], destination)
        elif asset["kind"] == "release-files":
            for filename in asset["files"]:
                download_url(asset["release_url"] + filename, destination / filename)
        elif asset["kind"] == "huggingface":
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=asset["repo_id"],
                revision=asset["revision"],
                local_dir=str(destination),
                token=False,
            )
        else:
            raise ProvisioningError(f"unsupported source kind for {asset['name']}")


def write_asset_manifest(root: Path) -> Path:
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {ASSET_MANIFEST_NAME, ACTIVATION_METADATA_NAME}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(root).as_posix()}\n")
    if not entries:
        raise ProvisioningError("no downloadable assets were staged")
    manifest = root / ASSET_MANIFEST_NAME
    manifest.write_text("".join(entries), encoding="utf-8")
    return manifest


def verify_before_activation(stage: Path, sources: dict[str, object]) -> None:
    manifest = write_asset_manifest(stage)
    subprocess.run(
        [str(VERIFY_SCRIPT), "--manifest", str(manifest), "--model-root", str(stage)],
        check=True,
    )
    metadata = {
        "schema_version": 1,
        "source_manifest": SOURCE_MANIFEST.name,
        "assets_manifest": ASSET_MANIFEST_NAME,
        "assets": sources["assets"],
        "offline_runtime_note": "Load only these paths with local_files_only=True or equivalent offline settings.",
    }
    (stage / ACTIVATION_METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def apply(root: Path, sources: dict[str, object], python_command: str, internal: bool) -> None:
    if not internal:
        interpreter = ensure_venv(python_command)
        subprocess.run(
            [str(interpreter), str(Path(__file__).resolve()), "--model-root", str(root), "--apply", "--internal-apply"],
            check=True,
            env={**os.environ, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"},
        )
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.provisioning-", dir=root.parent))
    try:
        download_assets(stage, sources)
        verify_before_activation(stage, sources)
        os.replace(stage, root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"ACTIVATED: verified local model root: {root}")
    print(f"Offline verification command: {VERIFY_SCRIPT} --manifest {root / ASSET_MANIFEST_NAME} --model-root {root}")


def main() -> int:
    args = parse_args()
    try:
        sources = load_sources()
        root = validated_model_root(args.model_root)
        if not args.apply:
            print_dry_run(root, sources, args.python)
            return 0
        require_desktop_user()
        apply(root, sources, args.python, args.internal_apply)
        return 0
    except (ProvisioningError, subprocess.CalledProcessError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
