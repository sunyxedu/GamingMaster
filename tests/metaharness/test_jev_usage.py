"""Exercise the prompt's example through the real context and HTTP adapter."""
import json
import re

from metaharness.core.contract import Budget, HarnessContext, VirtualClock
from metaharness.models.base import TraceSink
from metaharness.models.jev import JevClient, usage_guide


def test_bundled_example_uses_local_answer_contract(monkeypatch):
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"answers": {
                "condition": {"type": "choice", "choice": "unknown",
                              "probabilities": {"supported": 0.1, "contradicted": 0.1,
                                                "unknown": 0.8}, "confidence": 0.7},
                "progress": {"type": "score", "score": 1.25,
                             "probabilities": {"0": 0.0, "1": 0.75, "2": 0.25},
                             "confidence": 0.6},
                "finished": {"type": "noul", "noul": 0.25},
            }}).encode()

    def urlopen(request, timeout):
        sent.append(json.loads(request.data))
        assert 0 < timeout <= 5
        return Response()

    monkeypatch.setattr("metaharness.models.jev.urllib.request.urlopen", urlopen)
    namespace = {}
    example, = re.findall(r"```python\n(.*?)```", usage_guide(), flags=re.S)
    exec(compile(example, "jev_usage.md", "exec"), namespace)
    trace = TraceSink()
    budget = Budget(max_system1_calls=1)
    ctx = HarnessContext(
        system2=None, system1=JevClient(api_key="test-only"), trace=trace,
        clock=VirtualClock(), budget=budget, memory={}, game_kind="turn",
        act_fn=lambda action: None, observe_fn=lambda: {}, done_fn=lambda: False,
    )
    plan = {"goal": "Restore power", "preconditions": ["A power source is available."]}
    observation = {"summary": "A cable is attached, but power status is unknown."}
    try:
        result = namespace["assess_plan"](ctx, plan, observation)
    finally:
        ctx.close()
    assert len(sent) == budget.system1_calls == 1
    assert sent[0]["state"] == {"plan": plan, "observation": observation}
    assert {q["type"] for q in sent[0]["questions"].values()} == {"choice", "score", "noul"}
    assert result == {
        "condition": "unknown", "condition_probability": 0.8,
        "condition_confidence": 0.7, "progress_level": 1.25,
        "completion_probability": 0.25,
    }
    assert [e["type"] for e in trace.records] == ["model_request", "model_call"]
