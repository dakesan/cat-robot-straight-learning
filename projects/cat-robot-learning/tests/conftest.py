"""Shared fixtures; the tests directory is on sys.path for helper modules."""

import pytest

from catrobot import acquisition


@pytest.fixture
def fast_clock(monkeypatch):
    clock = [0.0]

    def advance():
        clock[0] += 0.01
        return clock[0]

    monkeypatch.setattr(acquisition.time, "monotonic", advance)
    monkeypatch.setattr("catrobot.quality.time.monotonic", advance)
    monkeypatch.setattr(acquisition.time, "sleep", lambda seconds: None)
