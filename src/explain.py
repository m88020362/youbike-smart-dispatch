# -*- coding: utf-8 -*-
"""Optional Bedrock explain layer -- natural language ONLY (Task C).

Position in the pipeline (strictly last, strictly optional):

    features -> XGBoost (local or SageMaker) -> Python decision layer
      -> Friend Relay / Truck dispatch -> Streamlit UI
        -> [optional] explain_decision()   <-- this module

Hard boundaries (enforced by construction, and covered by tests)
----------------------------------------------------------------
This module receives an ALREADY-FINISHED deterministic decision dict and turns
it into a sentence. It must never:
  * engineer features or compute probabilities;
  * classify risk;
  * rank or choose a donor station;
  * compute a Friend Relay mission or reward;
  * decide or change primary_action / secondary_action;
  * process many stations in one call (one decision per call, by signature).

``explain_decision`` therefore takes only a decision dict. It reads fields and
formats text. It has no access to the model, the snapshot, or the feature frame.

Safety / cost properties
------------------------
  * A deterministic template is ALWAYS produced first and returned on any
    failure, so the demo never depends on Bedrock being reachable.
  * Bedrock is only attempted when the caller explicitly passes
    ``use_bedrock=True`` AND config.BEDROCK_ENABLED is True. Nothing here is
    called automatically on a Streamlit rerun.
  * A module-level monotonic clock guard enforces
    config.BEDROCK_MIN_INTERVAL_SECONDS (>1s) between requests, satisfying the
    competition's "< 1 request/sec" rule even if the button is spammed.
  * AccessDenied / throttling / timeout / malformed output -> template fallback,
    with the reason reported back to the caller.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Dict, Optional, Tuple

from . import config
from . import intervention

# Where the explanation text came from.
SOURCE_TEMPLATE = "template"
SOURCE_BEDROCK = "bedrock"

# Chinese labels for the deterministic decision codes.
_ACTION_ZH = {
    intervention.DECISION_TRUCK: "車輛調度",
    intervention.DECISION_RELAY: "Friend Relay 引導使用者",
    intervention.DECISION_NONE: "暫不介入",
}
_PROBLEM_ZH = {
    intervention.PROBLEM_SHORTAGE: "缺車",
    intervention.PROBLEM_FULL: "滿站",
}

# --- rate limiting --------------------------------------------------------- #
# Guards the "< 1 request/sec" competition constraint across all callers.
_rate_lock = threading.Lock()
_last_request_monotonic: Optional[float] = None


def _reset_rate_limit() -> None:
    """Test hook: forget the last-request timestamp."""
    global _last_request_monotonic
    with _rate_lock:
        _last_request_monotonic = None


def _throttle() -> None:
    """Block until at least BEDROCK_MIN_INTERVAL_SECONDS since the last request.

    Called immediately before a Bedrock request. Because the interval is > 1s,
    the sustained rate can never reach 1 request/sec.
    """
    global _last_request_monotonic
    with _rate_lock:
        now = time.monotonic()
        if _last_request_monotonic is not None:
            elapsed = now - _last_request_monotonic
            remaining = float(config.BEDROCK_MIN_INTERVAL_SECONDS) - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
        _last_request_monotonic = now


# --- deterministic template ------------------------------------------------ #


def _relay_direction_zh(decision: Dict) -> str:
    """Which way Friend Relay helps, read from the finished mission dict.

    Purely descriptive: it reports the mission_type that the deterministic layer
    already chose. It never selects a direction itself.
    """
    mission = decision.get("friend_relay")
    if not mission:
        return "引導附近使用者協助調節"
    if mission.get("mission_type") == intervention.MISSION_RETURN:
        return "引導附近使用者還車"
    return "引導附近使用者借車"


def build_template_explanation(decision: Dict) -> str:
    """The deterministic, always-available natural-language explanation.

    Built only from fields the decision layer already produced. Same input =>
    same output, no network, no LLM. This is both the default and the fallback.
    """
    station = str(decision["station"])
    pct = round(float(decision["risk_probability"]) * 100)
    problem = _PROBLEM_ZH.get(str(decision.get("predicted_problem")), "供需失衡")
    primary = str(decision.get("primary_action"))
    secondary = decision.get("secondary_action")
    urgent = decision.get("urgency") == intervention.URGENCY_HIGH

    horizon = f"{station} {int(_horizon_minutes(decision))} 分鐘後{problem}風險 {pct}%"

    if primary == intervention.DECISION_NONE:
        return f"{horizon}，目前風險偏低，暫不需要介入。"

    urgency_phrase = "，情況緊急" if urgent else ""

    if primary == intervention.DECISION_TRUCK and secondary == intervention.DECISION_RELAY:
        return (
            f"{horizon}{urgency_phrase}，建議營運端優先進行車輛調度，"
            f"並搭配 Friend Relay {_relay_direction_zh(decision)}。"
        )
    if primary == intervention.DECISION_TRUCK:
        return (
            f"{horizon}{urgency_phrase}，建議營運端進行車輛調度補足供給。"
        )
    # primary == DECISION_RELAY
    return (
        f"{horizon}{urgency_phrase}，建議優先透過 Friend Relay "
        f"{_relay_direction_zh(decision)}，以較低成本改善失衡。"
    )


def _horizon_minutes(decision: Dict) -> int:
    """Prediction horizon in minutes (from the decision, else the config value)."""
    value = decision.get("horizon_minutes")
    if value is None:
        from . import predict

        return int(predict.PREDICTION_HORIZON_MINUTES)
    return int(value)


# --- Bedrock (optional) ---------------------------------------------------- #


def _decision_facts(decision: Dict) -> Dict:
    """The read-only fact sheet handed to Bedrock.

    Only already-decided values are included. The model is asked to phrase these
    facts, never to add or revise them.
    """
    mission = decision.get("friend_relay") or {}
    truck = decision.get("truck") or {}
    return {
        "station": str(decision["station"]),
        "horizon_minutes": _horizon_minutes(decision),
        "predicted_problem": str(decision.get("predicted_problem")),
        "risk_probability": round(float(decision["risk_probability"]), 4),
        "risk_level": str(decision.get("risk_level")),
        "primary_action": str(decision.get("primary_action")),
        "secondary_action": decision.get("secondary_action"),
        "urgency": str(decision.get("urgency")),
        "relay_mission_type": mission.get("mission_type"),
        "relay_partner_station": mission.get("partner_station"),
        "relay_reward_twd": mission.get("reward_twd"),
        "truck_from_or_to": truck.get("donor_or_dest"),
        "truck_move_bikes": truck.get("move_bikes"),
    }


_PROMPT = (
    "你是一個 YouBike 營運輔助說明工具。以下 JSON 是系統『已經決定好』的結果。\n"
    "請只用繁體中文寫 1 到 2 句話，把這個結果說明給營運人員聽。\n\n"
    "嚴格規則：\n"
    "1. 不要改變或重新計算任何機率數字。\n"
    "2. 不要改變 primary_action / secondary_action。\n"
    "3. 不要自己挑選其他站點。\n"
    "4. 不要加入 JSON 沒有的資訊或建議。\n"
    "5. 只輸出說明文字本身，不要有前言、標題或 JSON。\n\n"
    "結果 JSON：\n"
)


def _invoke_bedrock(facts: Dict, client=None) -> str:
    """One Bedrock request. Raises on any problem; caller falls back."""
    if client is None:
        import boto3
        from botocore.config import Config as BotoConfig

        client = boto3.client(
            "bedrock-runtime",
            region_name=config.BEDROCK_REGION,
            config=BotoConfig(
                read_timeout=config.BEDROCK_TIMEOUT_SECONDS,
                connect_timeout=config.BEDROCK_TIMEOUT_SECONDS,
                retries={"max_attempts": 1},
            ),
        )

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(config.BEDROCK_MAX_TOKENS),
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": _PROMPT + json.dumps(facts, ensure_ascii=False),
                }
            ],
        }
    )

    _throttle()  # guarantees < 1 request/sec
    response = client.invoke_model(
        modelId=config.BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    raw = response["body"].read()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    parsed = json.loads(raw)
    text = parsed["content"][0]["text"].strip()
    if not text:
        raise ValueError("Bedrock returned empty text")
    return text


def explain_decision(
    decision: Dict,
    use_bedrock: bool = False,
    client=None,
) -> Tuple[str, str, Optional[str]]:
    """Explain ONE finished decision in natural language.

    Args:
        decision: A completed decision dict from
            intervention.decide_intervention(). Must already contain the chosen
            actions; this function never decides anything.
        use_bedrock: Opt in to a single Bedrock request. Ignored unless
            config.BEDROCK_ENABLED is also True. Default False, so no Streamlit
            rerun can trigger a request by accident.
        client: Optional pre-built bedrock-runtime client (used by tests).

    Returns:
        (text, source, fallback_reason) where source is "bedrock" or "template".
        fallback_reason is None on success, otherwise a short description of why
        the template was used.

    Raises:
        KeyError: If the decision dict lacks "station" or "risk_probability".
    """
    # Deterministic result first: it is the answer unless Bedrock succeeds.
    template = build_template_explanation(decision)

    if not use_bedrock:
        return template, SOURCE_TEMPLATE, None
    if not config.BEDROCK_ENABLED:
        return template, SOURCE_TEMPLATE, "Bedrock 未啟用（config.BEDROCK_ENABLED=False）"

    try:
        text = _invoke_bedrock(_decision_facts(decision), client=client)
    except Exception as exc:  # AccessDenied / throttle / timeout / bad output
        return template, SOURCE_TEMPLATE, f"Bedrock 呼叫失敗，已使用內建說明：{exc}"

    return text, SOURCE_BEDROCK, None
