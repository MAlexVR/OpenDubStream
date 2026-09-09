"""Pinned local model assets with hash validation and atomic activation."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


class AssetIntegrityError(RuntimeError):
    """A local model set cannot safely be activated."""


@dataclass(frozen=True)
class AssetPin:
    name: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class AssetManifest:
    pins: tuple[AssetPin, ...]


@dataclass(frozen=True)
class LocalAssets:
    root: Path
    manifest: AssetManifest

    def require(self, name: str) -> Path:
        for pin in self.manifest.pins:
            if pin.name == name:
                return self.root / pin.filename
        raise AssetIntegrityError(f"asset pin is absent: {name}")


class LocalAssetStore:
    """Asset store deliberately has no downloader or network dependency."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.network_calls = 0

    def require_verified(self, manifest: AssetManifest) -> LocalAssets:
        if not self.root.is_dir():
            raise AssetIntegrityError("verified asset directory is missing")
        self._verify(manifest, self.root)
        return LocalAssets(self.root, manifest)

    def stage(self, manifest: AssetManifest, source: Path) -> LocalAssets:
        """Copy a complete verified set to a sibling staging directory then promote it."""
        if self.root.exists():
            return self.require_verified(manifest)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{self.root.name}.staging-", dir=self.root.parent))
        try:
            for pin in manifest.pins:
                candidate = source / pin.filename
                if not candidate.is_file():
                    raise AssetIntegrityError(f"pinned asset is missing: {pin.name}")
                shutil.copyfile(candidate, stage / pin.filename)
            self._verify(manifest, stage)
            os.replace(stage, self.root)
            return LocalAssets(self.root, manifest)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @staticmethod
    def _verify(manifest: AssetManifest, directory: Path) -> None:
        names = [pin.name for pin in manifest.pins]
        if len(set(names)) != len(names):
            raise AssetIntegrityError("asset manifest contains duplicate names")
        for pin in manifest.pins:
            path = directory / pin.filename
            if not path.is_file():
                raise AssetIntegrityError(f"pinned asset is missing: {pin.name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != pin.sha256:
                raise AssetIntegrityError(f"asset hash mismatch: {pin.name}")
