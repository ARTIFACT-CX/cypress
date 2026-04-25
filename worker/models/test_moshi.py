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


def test_moshi_registered_under_correct_name():
    # Loose smoke check that the @register decorator wired the class in
    # under the same name the UI sends in load_model commands.
    from . import REGISTRY

    assert "moshi" in REGISTRY
    assert REGISTRY["moshi"].name == "moshi"


def test_moshi_initial_state_is_unloaded():
    # device() returns None before load() runs; relied on by the host
    # status reporter so the UI doesn't show a stale device string.
    instance = moshi.Moshi(emit=lambda _msg: None)
    assert instance.device() is None
