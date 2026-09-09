"""Offline package-entrypoint tests: never install dependencies or open audio."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("package_launcher", ROOT / "packaging/opendubstream.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    module = load()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("OPENDUBSTREAM_SHARE", str(ROOT))
    assert module.main(["setup"]) == 0
    assert not (tmp_path / "data").exists()
    assert "--apply" in capsys.readouterr().out


def test_missing_runtime_is_actionable(tmp_path, monkeypatch, capsys):
    module = load()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert module.main([]) == 2
    assert "opendubstream setup --apply" in capsys.readouterr().err


def test_launch_uses_installed_source_and_user_models(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("OPENDUBSTREAM_SHARE", str(ROOT))
    runtime = tmp_path / "opendubstream/runtime/.venv-inference/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    called = []
    monkeypatch.setattr(module.os, "execve", lambda *args: called.append(args))
    assert module.main([]) == 0
    assert called[0][1] == [str(runtime), "-m", "opendubstream.ui.app"]
    assert called[0][2]["PYTHONPATH"] == str(ROOT / "src")
    assert called[0][2]["OPENDUBSTREAM_MODEL_ROOT"] == str(tmp_path / "opendubstream/models")


def test_setup_apply_refuses_root(monkeypatch, capsys):
    module = load()
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    assert module.main(["setup", "--apply"]) == 2
    assert "not root" in capsys.readouterr().err


def test_setup_existing_models_verifies_before_install(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("OPENDUBSTREAM_SHARE", str(ROOT))
    monkeypatch.setattr(module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/python3.12")
    (tmp_path / "opendubstream/models").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    assert module.main(["setup", "--apply"]) == 0
    assert calls[0][0] == "bash"
    assert "--manifest" in calls[0]
    assert calls[1][1:3] == ["-m", "venv"]
    assert calls[-1][-1].endswith("requirements/desktop-ui.txt")
    assert not list(ROOT.glob("runtime"))


def test_setup_rejects_bad_existing_models(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("OPENDUBSTREAM_SHARE", str(ROOT))
    monkeypatch.setattr(module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/python3.12")
    (tmp_path / "opendubstream/models").mkdir(parents=True)
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        raise module.subprocess.CalledProcessError(2, argv)

    monkeypatch.setattr(module.subprocess, "run", fail)
    assert module.main(["setup", "--apply"]) == 2
    assert len(calls) == 1
    assert not (tmp_path / "opendubstream/runtime/.venv-inference").exists()
