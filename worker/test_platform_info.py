"""
AREA: worker · TESTS · PLATFORM-INFO
Coverage for the worker's platform/cache snapshot. The Go side reads
these fields off every Handshake, so getting them right is what makes
remote variant selection work.
"""

import platform_info


def test_arch_normalizes_x86_64_to_amd64():
    assert platform_info._normalize_arch("x86_64") == "amd64"


def test_arch_normalizes_aarch64_to_arm64():
    assert platform_info._normalize_arch("aarch64") == "arm64"


def test_arch_passes_through_known_go_names():
    assert platform_info._normalize_arch("amd64") == "amd64"
    assert platform_info._normalize_arch("arm64") == "arm64"


def test_available_backends_unknown_family_returns_empty():
    assert platform_info.available_backends("not-a-real-family") == []


def test_available_backends_includes_torch_for_moshi():
    # torch is in the worker's shared dev venv (grpcio + protobuf are
    # the only required deps), but torch isn't, so we don't require it.
    # We DO require that the function returns a list (possibly empty)
    # rather than raising for the moshi family.
    out = platform_info.available_backends("moshi")
    assert isinstance(out, list)
    # Every entry must be one of the declared moshi backends.
    assert set(out) <= {"torch", "mlx"}


def test_downloaded_repos_missing_cache_returns_empty(tmp_path, monkeypatch):
    # Point HF_HOME at an empty tmpdir so the scan finds nothing.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    assert platform_info.downloaded_repos() == []


def test_downloaded_repos_finds_complete_snapshot(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    repo = hub / "models--kyutai--moshiko-pytorch-bf16"
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "snapshots" / "abc" / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    assert platform_info.downloaded_repos() == ["kyutai/moshiko-pytorch-bf16"]


def test_downloaded_repos_skips_partial_with_incomplete_blob(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    repo = hub / "models--kyutai--moshiko-pytorch-bf16"
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "snapshots" / "abc" / "model.safetensors").write_bytes(b"x")
    (repo / "blobs").mkdir()
    (repo / "blobs" / "deadbeef.incomplete").write_bytes(b"")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    # Half-pulled repo must NOT show up — otherwise the host would
    # render Downloaded=true for something that can't actually load.
    assert platform_info.downloaded_repos() == []


def test_gather_returns_all_fields():
    info = platform_info.gather("moshi")
    assert "os" in info
    assert "arch" in info
    assert "available_backends" in info
    assert "downloaded_repos" in info
    assert "gpu_name" in info
    assert "gpu_memory_gb" in info
    assert isinstance(info["available_backends"], list)
    assert isinstance(info["downloaded_repos"], list)
    assert isinstance(info["gpu_name"], str)
    assert isinstance(info["gpu_memory_gb"], int)
