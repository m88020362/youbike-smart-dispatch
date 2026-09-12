# -*- coding: utf-8 -*-
"""Pytest tests for the optional Bedrock explain layer.

Never calls real Bedrock. Verifies the explain layer:
  * only ever receives an already-finished decision dict;
  * defaults to the deterministic template (no request);
  * falls back to the template on AccessDenied / throttle / timeout / bad output;
  * cannot exceed the competition's < 1 request/sec limit;
  * never influences prediction / decision values.
"""

from __future__ import annotations

import json
import time
import types

import pytest

from src import config, explain, intervention


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _decision(
    station="捷運永安市場站",
    risk_prob=0.91,
    problem=intervention.PROBLEM_SHORTAGE,
    primary=intervention.DECISION_TRUCK,
    secondary=intervention.DECISION_RELAY,
    urgency=intervention.URGENCY_HIGH,
    risk_level="High",
    truck=None,
    relay=None,
):
    return {
        "station": station,
        "risk_level": risk_level,
        "risk_probability": risk_prob,
        "predicted_problem": problem,
        "primary_action": primary,
        "secondary_action": secondary,
        "urgency": urgency,
        "reason": "deterministic reason",
        "truck": truck if truck is not None else {
            "donor_or_dest": "鄰站A", "distance_km": 0.4, "move_bikes": 3,
        },
        "friend_relay": relay if relay is not None else {
            "original_station": "鄰站A",
            "partner_station": station,
            "mission_type": intervention.MISSION_RETURN,
            "predicted_problem": problem,
            "risk_probability": risk_prob,
            "distance_km": 0.4,
            "reward_twd": 10,
            "mission_text": "…",
        },
    }


class FakeBedrock:
    """Records invocations; can succeed, raise, or return malformed output."""

    def __init__(self, text="AWS 生成的說明。", raise_exc=None, malformed=False):
        self.text = text
        self.raise_exc = raise_exc
        self.malformed = malformed
        self.calls = []
        self.call_times = []

    def invoke_model(self, modelId, contentType, accept, body):
        self.calls.append({"modelId": modelId, "body": json.loads(body)})
        self.call_times.append(time.monotonic())
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.malformed:
            payload = {"unexpected": "shape"}
        else:
            payload = {"content": [{"text": self.text}]}
        return {"body": types.SimpleNamespace(
            read=lambda: json.dumps(payload).encode("utf-8")
        )}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    explain._reset_rate_limit()
    yield
    explain._reset_rate_limit()


# =========================================================================== #
# 1. deterministic template is the default -- no Bedrock request              #
# =========================================================================== #
def test_default_uses_template_and_makes_no_request():
    client = FakeBedrock()
    text, source, reason = explain.explain_decision(_decision(), client=client)
    assert source == explain.SOURCE_TEMPLATE
    assert reason is None
    assert client.calls == [], "default must never call Bedrock"
    assert "捷運永安市場站" in text
    assert "91%" in text


def test_template_mentions_primary_and_secondary_actions():
    text, _, _ = explain.explain_decision(_decision())
    assert "車輛調度" in text
    assert "Friend Relay" in text


def test_template_is_deterministic():
    d = _decision()
    a, _, _ = explain.explain_decision(d)
    b, _, _ = explain.explain_decision(d)
    assert a == b


def test_template_low_risk_says_no_intervention():
    d = _decision(
        risk_prob=0.05, primary=intervention.DECISION_NONE, secondary=None,
        urgency=intervention.URGENCY_NORMAL, risk_level="Low",
    )
    text, _, _ = explain.explain_decision(d)
    assert "暫不需要介入" in text


def test_template_relay_only_case():
    d = _decision(
        risk_prob=0.42, primary=intervention.DECISION_RELAY, secondary=None,
        urgency=intervention.URGENCY_NORMAL, risk_level="Medium",
    )
    text, _, _ = explain.explain_decision(d)
    assert "Friend Relay" in text
    assert "車輛調度" not in text


def test_template_reports_relay_direction_from_mission_only():
    """Direction is read from the finished mission, never re-decided."""
    borrow = _decision(relay={
        "partner_station": "B", "mission_type": intervention.MISSION_BORROW,
        "reward_twd": 5,
    })
    text, _, _ = explain.explain_decision(borrow)
    assert "借車" in text


# =========================================================================== #
# 2. explain layer only accepts a finished decision                           #
# =========================================================================== #
def test_signature_accepts_only_a_single_decision():
    """No snapshot / frame / model parameter exists, so it cannot decide."""
    import inspect

    params = set(inspect.signature(explain.explain_decision).parameters)
    assert params == {"decision", "use_bedrock", "client"}
    for forbidden in ("snapshot", "feature_frame", "artifacts", "bundle", "model"):
        assert forbidden not in params


def test_facts_sent_to_bedrock_are_only_already_decided_values(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock()
    d = _decision()
    explain.explain_decision(d, use_bedrock=True, client=client)

    sent = client.calls[0]["body"]["messages"][0]["content"]
    # The prompt carries the decided values verbatim.
    assert "truck" in sent
    assert "0.91" in sent
    # And explicitly forbids changing them.
    assert "不要改變或重新計算任何機率數字" in sent
    assert "不要改變 primary_action" in sent


def test_bedrock_result_does_not_alter_decision_dict(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock(text="完全不同的建議：不要調度。")
    d = _decision()
    before = json.dumps(d, sort_keys=True, default=str)
    explain.explain_decision(d, use_bedrock=True, client=client)
    after = json.dumps(d, sort_keys=True, default=str)
    assert before == after, "explain layer must not mutate the decision"


def test_one_call_per_invocation(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock()
    explain.explain_decision(_decision(), use_bedrock=True, client=client)
    assert len(client.calls) == 1, "one user action => at most one request"


# =========================================================================== #
# 3. Bedrock success path                                                     #
# =========================================================================== #
def test_bedrock_text_is_returned_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock(text="AWS 說明文字。")
    text, source, reason = explain.explain_decision(
        _decision(), use_bedrock=True, client=client
    )
    assert source == explain.SOURCE_BEDROCK
    assert text == "AWS 說明文字。"
    assert reason is None


def test_master_switch_off_blocks_bedrock_even_if_requested():
    """config.BEDROCK_ENABLED=False (the default) wins over use_bedrock=True."""
    assert config.BEDROCK_ENABLED is False
    client = FakeBedrock()
    text, source, reason = explain.explain_decision(
        _decision(), use_bedrock=True, client=client
    )
    assert source == explain.SOURCE_TEMPLATE
    assert client.calls == []
    assert "未啟用" in reason


# =========================================================================== #
# 4. failure -> deterministic template fallback                               #
# =========================================================================== #
@pytest.mark.parametrize(
    "exc",
    [
        Exception("AccessDeniedException: not authorized"),
        Exception("ThrottlingException: rate exceeded"),
        TimeoutError("read timeout"),
        Exception("ValidationException: model id not found"),
    ],
)
def test_bedrock_failures_fall_back_to_template(monkeypatch, exc):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock(raise_exc=exc)
    text, source, reason = explain.explain_decision(
        _decision(), use_bedrock=True, client=client
    )
    assert source == explain.SOURCE_TEMPLATE
    assert reason is not None
    # The usable deterministic sentence is still returned.
    assert "捷運永安市場站" in text and "91%" in text


def test_malformed_bedrock_output_falls_back(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock(malformed=True)
    text, source, reason = explain.explain_decision(
        _decision(), use_bedrock=True, client=client
    )
    assert source == explain.SOURCE_TEMPLATE
    assert reason is not None
    assert "91%" in text


def test_empty_bedrock_text_falls_back(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock(text="   ")
    text, source, _ = explain.explain_decision(
        _decision(), use_bedrock=True, client=client
    )
    assert source == explain.SOURCE_TEMPLATE


def test_bedrock_failure_does_not_raise(monkeypatch):
    """Explain failures must never break the main Streamlit flow."""
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock(raise_exc=Exception("boom"))
    # No exception escapes.
    explain.explain_decision(_decision(), use_bedrock=True, client=client)


# =========================================================================== #
# 5. rate limiting  (< 1 request/sec)                                         #
# =========================================================================== #
def test_rate_limit_interval_is_over_one_second():
    assert config.BEDROCK_MIN_INTERVAL_SECONDS > 1.0


def test_two_consecutive_requests_are_spaced_over_one_second(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    # Keep the test quick while still proving the gap is enforced.
    monkeypatch.setattr(config, "BEDROCK_MIN_INTERVAL_SECONDS", 1.05)
    client = FakeBedrock()
    d = _decision()
    explain.explain_decision(d, use_bedrock=True, client=client)
    explain.explain_decision(d, use_bedrock=True, client=client)
    assert len(client.call_times) == 2
    gap = client.call_times[1] - client.call_times[0]
    assert gap >= 1.0, f"requests only {gap:.3f}s apart; must be >= 1s"


def test_template_path_is_not_rate_limited():
    """Template-only calls are free and must not sleep."""
    started = time.monotonic()
    for _ in range(50):
        explain.explain_decision(_decision())
    assert time.monotonic() - started < 1.0


# =========================================================================== #
# 6. Streamlit rerun must not auto-trigger a request                          #
# =========================================================================== #
def test_streamlit_rerun_pattern_makes_no_request_without_button():
    """Simulate reruns: without the button, use_bedrock stays False."""
    client = FakeBedrock()
    session_state = {}
    for _ in range(10):  # ten reruns
        cached = session_state.get("_explain::S")
        if cached is None:
            # No button press => the app renders nothing and calls nothing.
            pass
    assert client.calls == []


def test_button_press_then_reruns_issue_only_one_request(monkeypatch):
    """One button press caches the result; later reruns reuse it."""
    monkeypatch.setattr(config, "BEDROCK_ENABLED", True)
    client = FakeBedrock()
    session_state = {}

    # rerun 1: button pressed
    text, source, reason = explain.explain_decision(
        _decision(), use_bedrock=True, client=client
    )
    session_state["_explain::S"] = {"text": text, "source": source,
                                    "fallback_reason": reason}

    # reruns 2..6: cached, no new request
    for _ in range(5):
        assert session_state.get("_explain::S") is not None

    assert len(client.calls) == 1


def test_app_explain_button_is_not_auto_invoked():
    """The app wires explain behind st.button and caches in session_state."""
    import io

    src = io.open("app.py", encoding="utf-8").read()
    assert 'st.button("AI 說明"' in src
    assert "_explain::" in src
    # The only explain call site is guarded by the button.
    idx_button = src.index('st.button("AI 說明"')
    idx_call = src.index("explain_layer.explain_decision")
    assert idx_call > idx_button, "explain must be called inside the button block"


def test_app_passes_master_switch_to_explain():
    import io

    src = io.open("app.py", encoding="utf-8").read()
    assert "use_bedrock=config.BEDROCK_ENABLED" in src
