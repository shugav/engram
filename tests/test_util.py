"""Tests for engram.util helpers."""

from __future__ import annotations

from engram.util import normalize_project


def test_normalize_strips_and_sanitizes():
    assert normalize_project("  My-App!  ") == "my-app"


def test_normalize_empty_uses_env(monkeypatch):
    monkeypatch.setenv("ENGRAM_PROJECT", "from-env")
    assert normalize_project("") == "from-env"
    assert normalize_project(None) == "from-env"


def test_normalize_default_when_empty_env(monkeypatch):
    monkeypatch.delenv("ENGRAM_PROJECT", raising=False)
    assert normalize_project("") == "default"
