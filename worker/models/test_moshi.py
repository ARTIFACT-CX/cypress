"""
AREA: worker · MODELS · MOSHI · TESTS

Unit tests for Moshi loader logic that doesn't require real torch/HF. The
actual `load()` path lands GBs of weights and several seconds of MPS
transfer — that's integration territory; here we cover only `_detect_device`
which is pure branching.
"""

import sys
import types

import pytest

from . import moshi


def _fake_torch(mps_available: bool, cuda_available: bool) -> types.SimpleNamespace:
    # Just enough surface area for _detect_device. Building a minimal
    # SimpleNamespace beats touching the real torch module — keeps the
    # test fast and works on machines without torch installed.
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available),
        ),
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
    )


def test_detect_device_honors_explicit_override(monkeypatch):
    # CYPRESS_DEVICE short-circuits autodetection so the user can force
    # CPU when MPS is misbehaving for a given torch release.
    monkeypatch.setenv("CYPRESS_DEVICE", "cpu")
    assert moshi._detect_device() == "cpu"


def test_detect_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.delenv("CYPRESS_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True, cuda_available=False))
    assert moshi._detect_device() == "mps"


def test_detect_device_falls_back_to_cuda(monkeypatch):
    monkeypatch.delenv("CYPRESS_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=False, cuda_available=True))
    assert moshi._detect_device() == "cuda"


def test_detect_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("CYPRESS_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=False, cuda_available=False))
    assert moshi._detect_device() == "cpu"


def test_moshi_torch_registered_under_explicit_name():
    # Loose smoke check that the @register decorator wired the class in
    # under the explicit backend name. The autoselect alias `"moshi"` is
    # platform-dependent and tested separately.
    from . import REGISTRY

    assert "moshi-torch" in REGISTRY
    assert REGISTRY["moshi-torch"] is moshi.Moshi


def test_moshi_alias_resolves_to_one_of_the_backends():
    # `"moshi"` is an alias chosen at import time by __init__.py based on
    # platform / env override. Either backend is a valid resolution — we
    # just want to confirm the alias exists and points at a real class.
    from . import REGISTRY

    assert "moshi" in REGISTRY
    assert REGISTRY["moshi"] in (REGISTRY["moshi-torch"], REGISTRY["moshi-mlx"])


def test_moshi_initial_state_is_unloaded():
    # device() returns None before load() runs; relied on by the host
    # status reporter so the UI doesn't show a stale device string.
    instance = moshi.Moshi(emit=lambda _msg: None)
    assert instance.device() is None


def test_moshi_exposes_run_wav():
    # The IPC handler gates run_wav requests on hasattr(instance,
    # "run_wav"); accidentally renaming/removing the method here would
    # silently make the offline self-test reject every request as
    # "model does not support run_wav". This test pins the public name.
    assert callable(getattr(moshi.Moshi, "run_wav", None))


def test_moshi_run_wav_rejects_when_unloaded():
    # Defensive: the IPC handler gates on `instance is not None`, but a
    # half-loaded Moshi (load() failed midway) would have these unset.
    # run_wav must error clearly rather than dereferencing None into a
    # cryptic AttributeError deep in the moshi library.
    instance = moshi.Moshi(emit=lambda _msg: None)
    with pytest.raises(RuntimeError, match="not loaded"):
        instance.run_wav("in.wav", "out.wav")


def test_moshi_exposes_stream():
    # IPC handler will gate streaming requests on hasattr(instance,
    # "stream"); pin the public name so a rename doesn't silently break
    # the audio pipeline's session setup.
    assert callable(getattr(moshi.Moshi, "stream", None))


def test_moshi_stream_rejects_when_unloaded():
    # Same defensive shape as run_wav: a half-loaded instance must raise
    # clearly rather than crashing inside LMGen construction.
    instance = moshi.Moshi(emit=lambda _msg: None)
    with pytest.raises(RuntimeError, match="not loaded"):
        instance.stream()
