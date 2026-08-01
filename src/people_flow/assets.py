"""Verified acquisition of small official Phase 3 assets."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from people_flow.errors import AssetDownloadError


@dataclass(frozen=True, slots=True)
class RemoteAsset:
    """Immutable metadata for an official downloadable asset."""

    name: str
    url: str
    sha256: str
    size_bytes: int


OFFICIAL_MODELS: Mapping[str, RemoteAsset] = {
    "yolo26n.pt": RemoteAsset(
        name="yolo26n.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        sha256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        size_bytes=5_544_453,
    ),
    "yolov8n.pt": RemoteAsset(
        name="yolov8n.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        sha256="f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        size_bytes=6_549_796,
    ),
}

OFFICIAL_SAMPLES: Mapping[str, RemoteAsset] = {
    "People-counting-compressed.mp4": RemoteAsset(
        name="People-counting-compressed.mp4",
        url=(
            "https://github.com/ultralytics/assets/releases/download/"
            "v0.0.0/People-counting-compressed.mp4"
        ),
        sha256="e9ebad02081416465caae6ff87a28f96c441337742266b45f29a8c8a7b80c3a5",
        size_bytes=358_219,
    )
}


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, asset: RemoteAsset) -> None:
    """Raise a clear error when size or checksum differs from official metadata."""

    actual_size = path.stat().st_size
    if actual_size != asset.size_bytes:
        raise AssetDownloadError(
            f"Asset size mismatch for {path}: expected {asset.size_bytes}, got {actual_size}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != asset.sha256:
        raise AssetDownloadError(
            f"Asset checksum mismatch for {path}: expected {asset.sha256}, got {actual_sha256}"
        )


def download_verified_asset(asset: RemoteAsset, destination: Path, *, timeout: int = 60) -> Path:
    """Download an official asset atomically and verify its size and SHA-256."""

    resolved = destination.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_file():
            raise AssetDownloadError(f"Asset destination is not a file: {resolved}")
        verify_asset(resolved, asset)
        return resolved

    resolved.parent.mkdir(parents=True, exist_ok=True)
    partial = resolved.with_name(f"{resolved.name}.part")
    request = Request(asset.url, headers={"User-Agent": "people-flow-analytics/2"})
    try:
        with urlopen(request, timeout=timeout) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify_asset(partial, asset)
        partial.replace(resolved)
    except (HTTPError, URLError, OSError, AssetDownloadError) as exc:
        partial.unlink(missing_ok=True)
        raise AssetDownloadError(
            f"Unable to download and verify {asset.name} from {asset.url}: {exc}"
        ) from exc
    return resolved


def resolve_model_path(model: str, *, weights_dir: Path) -> Path:
    """Resolve a custom model path or acquire a known official model without fallback."""

    requested = Path(model).expanduser()
    if requested.parent == Path("."):
        asset = OFFICIAL_MODELS.get(requested.name)
        if asset is not None:
            return download_verified_asset(asset, weights_dir / asset.name)
    if requested.is_file():
        return requested.resolve()
    if requested.is_absolute() or requested.parent != Path("."):
        raise AssetDownloadError(f"Requested model file does not exist: {requested.resolve()}")
    if requested.name not in OFFICIAL_MODELS:
        supported = ", ".join(sorted(OFFICIAL_MODELS))
        raise AssetDownloadError(
            f"Unknown model '{model}'. Provide an existing model path or one of: {supported}"
        )
    raise AssetDownloadError(f"Unable to resolve requested model: {model}")


def download_official_sample(*, samples_dir: Path) -> Path:
    """Download the small official people-counting sample used for local validation."""

    asset = OFFICIAL_SAMPLES["People-counting-compressed.mp4"]
    return download_verified_asset(asset, samples_dir / asset.name)
