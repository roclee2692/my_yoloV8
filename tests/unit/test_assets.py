"""Tests for verified asset handling without network access."""

from pathlib import Path

import pytest

from people_flow.assets import RemoteAsset, file_sha256, resolve_model_path, verify_asset
from people_flow.errors import AssetDownloadError


def test_verify_asset_accepts_matching_file(tmp_path: Path) -> None:
    """Official metadata should validate both byte length and SHA-256."""

    path = tmp_path / "asset.bin"
    path.write_bytes(b"people-flow")
    asset = RemoteAsset("asset.bin", "https://example.invalid/asset", file_sha256(path), 11)

    verify_asset(path, asset)


def test_verify_asset_rejects_changed_file(tmp_path: Path) -> None:
    """A corrupt local asset must not be accepted silently."""

    path = tmp_path / "asset.bin"
    path.write_bytes(b"wrong")
    asset = RemoteAsset("asset.bin", "https://example.invalid/asset", "0" * 64, 5)

    with pytest.raises(AssetDownloadError, match="checksum mismatch"):
        verify_asset(path, asset)


def test_resolve_model_path_accepts_existing_custom_model(tmp_path: Path) -> None:
    """Custom weights are allowed only when the requested file really exists."""

    model = tmp_path / "custom.pt"
    model.write_bytes(b"test")

    assert resolve_model_path(str(model), weights_dir=tmp_path / "weights") == model.resolve()


def test_resolve_model_path_rejects_unknown_bare_name(tmp_path: Path) -> None:
    """Unknown model names must not trigger a fallback download."""

    with pytest.raises(AssetDownloadError, match="Unknown model"):
        resolve_model_path("not-a-model.pt", weights_dir=tmp_path)
