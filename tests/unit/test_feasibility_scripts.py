"""Behavioral tests for the explicit, no-download feasibility helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def load_provisioner_module():
    spec = importlib.util.spec_from_file_location(
        "provision_local_models",
        SCRIPTS / "provision-local-models.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(
    name: str,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPTS / name), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": os.environ["PATH"], **(environment or {})},
    )


def test_operational_scripts_are_syntax_safe_and_document_explicit_gates() -> None:
    for script in (
        "diagnose-feasibility.sh",
        "prepare-fedora-system.sh",
        "verify-local-model.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS / script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    assert "--apply" in run_script("prepare-fedora-system.sh", "--help").stdout
    assert "pkexec" in run_script("prepare-fedora-system.sh", "--help").stdout

    provisioner = subprocess.run(
        ["python3", "-m", "py_compile", str(SCRIPTS / "provision-local-models.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert provisioner.returncode == 0, provisioner.stderr


def test_diagnostic_reports_missing_browser_without_pgrep_name_length_warning(tmp_path: Path) -> None:
    """The missing-prerequisite diagnostic must not inherit real desktop assets."""
    isolated_home = tmp_path / "home"
    isolated_runtime = tmp_path / "runtime"
    isolated_home.mkdir()
    isolated_runtime.mkdir()
    result = subprocess.run(
        [str(SCRIPTS / "diagnose-feasibility.sh")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("OPENDUBSTREAM_MODEL_")},
            "HOME": str(isolated_home),
            "XDG_RUNTIME_DIR": str(isolated_runtime),
        },
    )

    assert result.returncode != 0
    assert "Chrome/Chromium playback streams" in result.stdout
    assert "Kokoro v1 model plus voices" in result.stderr
    assert "pattern that searches for process name longer" not in result.stderr


def test_diagnostic_uses_inference_venv_and_default_activated_model_root(tmp_path: Path) -> None:
    """An activated local runtime must work without repeating paths by hand."""
    isolated_repo = tmp_path / "repo"
    isolated_scripts = isolated_repo / "scripts"
    isolated_scripts.mkdir(parents=True)
    for script in ("diagnose-feasibility.sh", "verify-local-model.sh"):
        destination = isolated_scripts / script
        destination.write_text((SCRIPTS / script).read_text(encoding="utf-8"), encoding="utf-8")
        destination.chmod(0o755)

    runtime_python = isolated_repo / ".venv-inference" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${2:-}\" == *get_voices* ]]; then\n"
        "  printf '%s\\n' 'voices=ef_dora,em_alex,em_santa'\n"
        "else\n"
        "  printf '%s\\n' 'torch=True ctranslate2=True faster_whisper=True transformers=True sentencepiece=True silero_vad=True kokoro_onnx=True onnxruntime=True'\n"
        "fi\n",
        encoding="utf-8",
    )
    runtime_python.chmod(0o755)

    model_root = tmp_path / "home" / ".local" / "share" / "opendubstream" / "models"
    assets = {
        "silero-vad/model.onnx": b"vad",
        "faster-whisper-distil-large-v3/model.bin": b"asr",
        "opus-mt-en-es/pytorch_model.bin": b"translation",
        "kokoro-onnx-v1/kokoro-v1.0.onnx": b"tts-model",
        "kokoro-onnx-v1/voices-v1.0.bin": b"tts-voices",
    }
    manifest_lines = []
    for relative_path, contents in assets.items():
        asset = model_root / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(contents)
        manifest_lines.append(f"{hashlib.sha256(contents).hexdigest()}  {relative_path}\n")
    (model_root / "assets.sha256").write_text("".join(manifest_lines), encoding="utf-8")

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    result = subprocess.run(
        [str(isolated_scripts / "diagnose-feasibility.sh")],
        cwd=isolated_repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path / "home"), "XDG_RUNTIME_DIR": str(runtime_dir)},
    )

    assert result.returncode == 0, result.stderr
    assert "PASS: local inference runtime imports" in result.stdout
    assert "kokoro_onnx=True" in result.stdout
    assert "VERIFIED: required local model assets" in result.stdout
    assert "PASS: Kokoro offline model/voice compatibility (read-only)" in result.stdout
    assert "voices=ef_dora,em_alex,em_santa" in result.stdout
    assert "no user-provided manifest/model root" not in result.stdout


def test_diagnostic_honors_local_runtime_and_asset_environment_overrides(tmp_path: Path) -> None:
    """Alternative local roots remain explicit and never require a download path."""
    isolated_repo = tmp_path / "repo"
    isolated_scripts = isolated_repo / "scripts"
    isolated_scripts.mkdir(parents=True)
    for script in ("diagnose-feasibility.sh", "verify-local-model.sh"):
        destination = isolated_scripts / script
        destination.write_text((SCRIPTS / script).read_text(encoding="utf-8"), encoding="utf-8")
        destination.chmod(0o755)

    runtime_python = tmp_path / "runtime-python"
    runtime_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${2:-}\" == *get_voices* ]]; then\n"
        "  printf '%s\\n' 'voices=ef_dora,em_alex,em_santa'\n"
        "else\n"
        "  printf '%s\\n' 'torch=True ctranslate2=True faster_whisper=True transformers=True sentencepiece=True silero_vad=True kokoro_onnx=True onnxruntime=True'\n"
        "fi\n",
        encoding="utf-8",
    )
    runtime_python.chmod(0o755)

    model_root = tmp_path / "offline-assets"
    asset = model_root / "silero-vad" / "model.onnx"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"vad")
    for relative_path, contents in {
        "faster-whisper-distil-large-v3/model.bin": b"asr",
        "opus-mt-en-es/pytorch_model.bin": b"translation",
        "kokoro-onnx-v1/kokoro-v1.0.onnx": b"tts-model",
        "kokoro-onnx-v1/voices-v1.0.bin": b"tts-voices",
    }.items():
        pinned_asset = model_root / relative_path
        pinned_asset.parent.mkdir(parents=True, exist_ok=True)
        pinned_asset.write_bytes(contents)
    manifest_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(model_root)}\n"
        for path in sorted(model_root.rglob("*"))
        if path.is_file()
    ]
    manifest = tmp_path / "offline-assets.sha256"
    manifest.write_text("".join(manifest_lines), encoding="utf-8")

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    result = subprocess.run(
        [str(isolated_scripts / "diagnose-feasibility.sh")],
        cwd=isolated_repo,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "XDG_RUNTIME_DIR": str(runtime_dir),
            "OPENDUBSTREAM_INFERENCE_PYTHON": str(runtime_python),
            "OPENDUBSTREAM_MODEL_ROOT": str(model_root),
            "OPENDUBSTREAM_MODEL_MANIFEST": str(manifest),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "kokoro_onnx=True" in result.stdout
    assert "VERIFIED: required local model assets and 5 pinned local file(s)." in result.stdout
    assert "PASS: Kokoro offline model/voice compatibility (read-only)" in result.stdout


def test_local_model_provisioner_is_dry_run_by_default_and_never_creates_a_model_root(tmp_path: Path) -> None:
    model_root = tmp_path / "models"

    result = run_script(
        "provision-local-models.py",
        "--model-root",
        str(model_root),
    )

    assert result.returncode == 0, result.stderr
    assert "DRY-RUN" in result.stdout
    assert "Runtime bootstrap: .venv-inference" in result.stdout
    assert "repository .venv" not in result.stdout
    assert "--apply" in result.stdout
    assert not model_root.exists()


def test_local_model_provisioner_uses_a_dedicated_inference_venv_and_preserves_project_test_venv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provisioner = load_provisioner_module()
    project_venv = tmp_path / ".venv" / "bin" / "python"
    project_venv.parent.mkdir(parents=True)
    project_venv.write_text("Python 3.14 test environment", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        calls.append(command)

    monkeypatch.setattr(provisioner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(provisioner.subprocess, "run", fake_run)
    monkeypatch.setattr(provisioner.subprocess, "check_output", lambda *_args, **_kwargs: "3.13\n")

    interpreter = provisioner.ensure_venv("python3.13")

    inference_venv = tmp_path / ".venv-inference"
    assert interpreter == inference_venv / "bin" / "python"
    assert calls[0] == ["python3.13", "-m", "venv", str(inference_venv)]
    assert all(str(project_venv) not in command for command in calls)
    assert project_venv.read_text(encoding="utf-8") == "Python 3.14 test environment"


def test_local_model_provisioner_patches_activate_script_with_cuda_library_path(tmp_path: Path, monkeypatch) -> None:
    """onnxruntime/ctranslate2 need pip-installed nvidia/*/lib on LD_LIBRARY_PATH; see benchmarks/feasibility-2026-09-03.md."""
    provisioner = load_provisioner_module()
    venv = tmp_path / ".venv-inference"
    activate = venv / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("_OLD_VIRTUAL_PATH=\"$PATH\"\nPATH=\"$VIRTUAL_ENV/\"bin\":$PATH\"\nexport PATH\n", encoding="utf-8")

    provisioner.patch_activate_for_cuda_library_path(venv)
    once = activate.read_text(encoding="utf-8")
    provisioner.patch_activate_for_cuda_library_path(venv)  # idempotent: must not duplicate on a second provisioning run
    twice = activate.read_text(encoding="utf-8")

    assert "LD_LIBRARY_PATH" in once
    assert "nvidia" in once
    assert once == twice
    assert once.count("_OLD_VIRTUAL_LD_LIBRARY_PATH") == 1


def test_local_model_provisioner_rejects_unsafe_apply_root_before_network_or_install(tmp_path: Path) -> None:
    existing_root = tmp_path / "existing-models"
    existing_root.mkdir()
    marker = existing_root / "keep.txt"
    marker.write_text("do not replace", encoding="utf-8")

    result = run_script(
        "provision-local-models.py",
        "--apply",
        "--model-root",
        str(existing_root),
    )

    assert result.returncode != 0
    assert "refusing to replace an existing model root" in result.stderr
    assert marker.read_text(encoding="utf-8") == "do not replace"


def test_local_model_provisioner_requires_the_logged_in_desktop_session_before_apply(tmp_path: Path) -> None:
    model_root = tmp_path / "models"

    result = run_script(
        "provision-local-models.py",
        "--apply",
        "--model-root",
        str(model_root),
        environment={"XDG_RUNTIME_DIR": ""},
    )

    assert result.returncode != 0
    assert "XDG_RUNTIME_DIR is unavailable" in result.stderr
    assert not model_root.exists()


def test_source_manifest_pins_each_required_model_to_an_immutable_revision() -> None:
    sources = json.loads((SCRIPTS / "local-model-sources.json").read_text(encoding="utf-8"))

    assert {asset["name"] for asset in sources["assets"]} == {
        "silero-vad",
        "faster-whisper-distil-large-v3",
        "opus-mt-en-es",
        "kokoro-onnx-v1",
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", asset["revision"])
        for asset in sources["assets"]
        if asset["name"] != "kokoro-onnx-v1"
    )
    assert sources["estimated_download_bytes"] > 2_000_000_000

    kokoro = next(asset for asset in sources["assets"] if asset["name"] == "kokoro-onnx-v1")
    assert kokoro["kind"] == "release-files"
    assert kokoro["destination"] == "kokoro-onnx-v1"
    assert kokoro["release_url"] == (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    )
    assert kokoro["files"] == ["kokoro-v1.0.onnx", "voices-v1.0.bin"]


def test_provisioner_stages_both_official_kokoro_release_files(tmp_path: Path, monkeypatch) -> None:
    provisioner = load_provisioner_module()
    sources = json.loads((SCRIPTS / "local-model-sources.json").read_text(encoding="utf-8"))
    kokoro = next(asset for asset in sources["assets"] if asset["name"] == "kokoro-onnx-v1")
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        provisioner,
        "download_url",
        lambda url, destination: downloads.append((url, destination)),
    )

    provisioner.download_assets(tmp_path, {"assets": [kokoro]})

    assert downloads == [
        (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            tmp_path / "kokoro-onnx-v1" / "kokoro-v1.0.onnx",
        ),
        (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            tmp_path / "kokoro-onnx-v1" / "voices-v1.0.bin",
        ),
    ]


def test_local_model_verifier_accepts_complete_pinned_assets_without_download(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    assets = {
        "silero-vad/model.onnx": b"vad",
        "faster-whisper-distil-large-v3/model.bin": b"asr",
        "opus-mt-en-es/pytorch_model.bin": b"translation",
        "kokoro-onnx-v1/kokoro-v1.0.onnx": b"tts-model",
        "kokoro-onnx-v1/voices-v1.0.bin": b"tts-voices",
    }
    lines = []
    for relative_path, contents in assets.items():
        asset = model_root / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(contents)
        lines.append(f"{hashlib.sha256(contents).hexdigest()}  {relative_path}\n")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("".join(lines), encoding="utf-8")

    result = run_script(
        "verify-local-model.sh",
        "--manifest",
        str(manifest),
        "--model-root",
        str(model_root),
    )

    assert result.returncode == 0, result.stderr
    assert "VERIFIED: required local model assets" in result.stdout
    assert "download" not in result.stdout.lower()


def test_local_model_verifier_rejects_the_incompatible_spanish_kokoro_layout(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    assets = {
        "silero-vad/model.onnx": b"vad",
        "faster-whisper-distil-large-v3/model.bin": b"asr",
        "opus-mt-en-es/pytorch_model.bin": b"translation",
        "kokoro-spanish-onnx/kokoro-v1.0-es.onnx": b"old-tts",
    }
    lines = []
    for relative_path, contents in assets.items():
        asset = model_root / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(contents)
        lines.append(f"{hashlib.sha256(contents).hexdigest()}  {relative_path}\n")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("".join(lines), encoding="utf-8")

    result = run_script(
        "verify-local-model.sh",
        "--manifest",
        str(manifest),
        "--model-root",
        str(model_root),
    )

    assert result.returncode != 0
    assert "required local model asset pin is absent" in result.stderr


def test_local_model_verifier_fails_closed_for_an_unpinned_or_tampered_asset(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    asset = model_root / "opus-mt-en-es" / "weights.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"tampered")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(f"{'0' * 64}  opus-mt-en-es/weights.bin\n", encoding="utf-8")

    result = run_script(
        "verify-local-model.sh",
        "--manifest",
        str(manifest),
        "--model-root",
        str(model_root),
    )

    assert result.returncode != 0
    assert "hash mismatch" in result.stderr
