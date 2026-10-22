"""Offline stand-ins time out like real backends when their declared latency exceeds the call's timeout."""
from __future__ import annotations

import pytest

from metaharness.models.base import ModelTimeout, TraceSink
from metaharness.models.jev import Choice
from metaharness.models.mock import HeuristicSystem1, ScriptedSystem2


def test_scripted_system2_times_out_past_its_declared_latency():
    model, tr = ScriptedSystem2(replies=["up"], latency_s=2.0), TraceSink()
    with pytest.raises(ModelTimeout) as ei:
        model.run("move?", session_id="s-1", timeout_s=1.0, trace=tr)
    assert isinstance(ei.value, TimeoutError) and ei.value.session_id == "s-1"
    assert tr.records[-1]["ok"] is False and "timeout" in tr.records[-1]["error"] and model.n_calls == 0
    assert model.run("move?", timeout_s=2.0, trace=tr).text == "up" and tr.records[-1]["ok"] is True


def test_heuristic_system1_times_out_past_its_declared_latency():
    model, tr = HeuristicSystem1(latency_s=1.0), TraceSink()
    questions = {"move": Choice("q", {"up": "u", "down": "d"})}
    with pytest.raises(TimeoutError, match="timeout"):
        model.decide({}, questions, timeout_s=0.5, trace=tr)
    assert tr.records[-1]["ok"] is False and model.n_calls == 0
    assert model.decide({}, questions, timeout_s=1.0, trace=tr)["move"].value == "up"
    assert tr.records[-1]["ok"] is True
