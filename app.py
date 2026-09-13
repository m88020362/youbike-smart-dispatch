# -*- coding: utf-8 -*-
"""Streamlit end-to-end demo for the YouBike predictive dispatch MVP (Task 6, R8).

One app, TWO audiences, one shared 30-minute risk engine (design §10, R8):

    使用者模式  — 一般民眾。以「我要借車 / 還車」的意圖出發，選一個原定站點，
                 看目前狀態與 30 分鐘後風險；只有在真的能順路幫上忙時，才顯示
                 一張 Friend Relay 2.0 推薦卡（小幅繞行 + 獎勵）。沒有可行的推薦
                 就什麼都不顯示（不打擾使用者）。使用者模式**不**出現派車調度。
    營運中心    — 營運人員 / 評審。同一套預測，加上需要營運介入的「派車調度」建議。

共用概念：同一套 30 分鐘風險預測 → 使用者端 Friend Relay 2.0（用原本就會發生的
借還車行程，小幅繞行 + 獎勵做預防式微調度）；營運端 派車調度（處理需營運介入的
失衡）。

它只串接既有的公開 API — 不重新訓練模型、不修改原始 CSV、不改 Tasks 1-5 的邏輯。
繁重的工作（清理資料、建立特徵、載入模型、對某個時間點預測整網快照）都被封裝成
可 import、可快取的小函式，讓 UI 之外的邏輯能在不啟動 Streamlit 的情況下被測試。

重用的 pipeline（design §3）：
    data_loader.load_clean() → features.build_targets() → features.build_features()
      → predict.load_artifacts() / predict.predict_station()（每站一個時間點）
        → intervention.recommend()（營運端派車調度）
        → intervention.recommend_borrow_relay / recommend_return_relay
          （使用者意圖 Friend Relay 2.0 任務）

模型檔缺失時（predict.ModelArtifactsMissingError）會顯示「請先訓練
（python -m src.train）」而非崩潰。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src import (
    config,
    data_loader,
    features,
    intervention,
    predict,
    sagemaker_predict,
)
from src import explain as explain_layer

# The observation-timestamp label format shown in the selector.
_TS_FMT = "%Y-%m-%d %H:%M"

# User-facing intent labels (natural Chinese only, no bilingual clutter).
INTENT_BORROW_LABEL = "借車"
INTENT_RETURN_LABEL = "還車"

# Developer-only raw §9 prediction/recommendation inspector. Off by default so
# the normal demo stays clean; the underlying data (view["_prediction_raw"] /
# ["_recommendation_raw"]) is still built and available for debugging.
SHOW_DEV_VIEW = False

# Developer-facing labels for the prediction backend selector. The backend only
# changes WHERE shortage_prob / full_prob come from; all downstream risk levels,
# intervention rules, Friend Relay missions and UI behaviour are unchanged.
BACKEND_LABEL_LOCAL = "本機模型"
BACKEND_LABEL_SAGEMAKER = "AWS SageMaker"
BACKEND_LABELS = {
    config.BACKEND_LOCAL: BACKEND_LABEL_LOCAL,
    config.BACKEND_SAGEMAKER: BACKEND_LABEL_SAGEMAKER,
}


def backend_label(backend: str) -> str:
    """Human-readable label for a prediction backend code."""
    return BACKEND_LABELS.get(str(backend), str(backend))

# Natural-language risk-level labels for the risk badge.
_RISK_ZH = {
    predict.RISK_LOW: "低風險",
    predict.RISK_MEDIUM: "中風險",
    predict.RISK_HIGH: "高風險",
}


# --------------------------------------------------------------------------- #
# Importable, testable helpers (no Streamlit imports here so they can run      #
# head-less for smoke tests).                                                  #
# --------------------------------------------------------------------------- #


def build_feature_bundle(csv_path: Optional[str] = None) -> features.FeatureBundle:
    """Load + clean the CSV and build the leakage-free feature bundle.

    Thin wrapper over the Task 1/2 pipeline. Kept separate so callers (and the
    Streamlit cache) can reuse a single bundle across the whole session.
    """
    df = data_loader.load_clean(csv_path)
    targets = features.build_targets(df)
    return features.build_features(targets)


def _bundle_frame(bundle: features.FeatureBundle) -> pd.DataFrame:
    """Assemble a single working DataFrame from a FeatureBundle.

    Columns: every feature column in X, plus the aligned ``timestamp`` and a
    positional ``_row`` index so we can hand the exact X row back to
    predict_station. The station name is recovered from station_id via the
    bundle's deterministic station_encoding.
    """
    frame = bundle.X.copy()
    frame[config.COL_TIMESTAMP] = bundle.timestamp.values
    frame["_row"] = range(len(frame))

    # Invert the deterministic station encoding: station_id -> station name.
    id_to_station = {v: k for k, v in bundle.station_encoding.items()}
    frame["_station_name"] = frame[config.COL_STATION_ID].map(id_to_station)
    return frame


def available_timestamps(bundle: features.FeatureBundle) -> List[pd.Timestamp]:
    """Sorted unique observation timestamps present in the feature bundle."""
    ts = pd.to_datetime(pd.Series(bundle.timestamp)).dropna().unique()
    return sorted(pd.Timestamp(t) for t in ts)


def build_snapshot(
    bundle: features.FeatureBundle,
    observation_ts: pd.Timestamp,
    artifacts: Dict,
    backend: str = config.BACKEND_LOCAL,
    sm_client=None,
) -> List[Dict]:
    """Predict a whole-network §9 snapshot for one observation timestamp.

    For each station, the most recent feature row at or before ``observation_ts``
    is used (so every station contributes at most one prediction, reflecting its
    latest known state up to the chosen moment). Each row is run through
    predict.predict_station() to produce a design §9 prediction dict; the list of
    those dicts is exactly the snapshot shape intervention.* consumes.

    Args:
        bundle: FeatureBundle from build_feature_bundle().
        observation_ts: The demo "now" moment to snapshot.
        artifacts: Pre-loaded predict.load_artifacts() dict (models + meta).
        backend: config.BACKEND_LOCAL (in-process models, default) or
            config.BACKEND_SAGEMAKER (call the deployed real-time endpoints).
            Only the probability SOURCE changes; the resulting §9 dicts are
            identical in shape and semantics either way.
        sm_client: Optional sagemaker-runtime client reused across stations when
            backend is SageMaker. Created once on demand when None.

    Returns:
        A list of §9 prediction dicts (one per station that has data at/before
        the timestamp), sorted by station name.
    """
    frame = _bundle_frame(bundle)
    observation_ts = pd.Timestamp(observation_ts)

    eligible = frame[frame[config.COL_TIMESTAMP] <= observation_ts]
    if eligible.empty:
        return []

    # Latest row per station up to the chosen timestamp.
    latest = (
        eligible.sort_values(config.COL_TIMESTAMP)
        .groupby(config.COL_STATION_ID, sort=False)
        .tail(1)
    )

    use_sagemaker = backend == config.BACKEND_SAGEMAKER

    # Collect the aligned per-station inputs once; both backends consume these.
    row_indices: List[int] = []
    metas: List[Dict] = []
    timestamps: List[pd.Timestamp] = []
    for _, row in latest.iterrows():
        row_indices.append(int(row["_row"]))
        metas.append(
            {
                "station": row["_station_name"],
                "lat": float(row[config.COL_LAT]),
                "lon": float(row[config.COL_LON]),
                "current_bikes": int(row[config.COL_AVAILABLE_BIKES]),
                "current_docks": int(row[config.COL_AVAILABLE_DOCKS]),
                "total_docks": int(row[config.COL_TOTAL_DOCKS]),
            }
        )
        timestamps.append(row[config.COL_TIMESTAMP])

    if use_sagemaker:
        # BATCH: exactly two invoke_endpoint calls for the whole snapshot
        # (one shortage, one full) instead of 2 per station.
        if sm_client is None:
            sm_client = sagemaker_predict.get_runtime_client()
        feature_frame = bundle.X.iloc[row_indices].reset_index(drop=True)
        snapshot = sagemaker_predict.build_snapshot_batch(
            feature_frame,
            metas,
            timestamps,
            feature_meta=artifacts["feature_meta"],
            client=sm_client,
        )
    else:
        # LOCAL: unchanged stable path, one in-process prediction per station.
        snapshot = [
            predict.predict_station(
                bundle.X.iloc[idx], meta, ts, artifacts=artifacts
            )
            for idx, meta, ts in zip(row_indices, metas, timestamps)
        ]

    snapshot.sort(key=lambda p: str(p["station"]))
    return snapshot


def snapshot_by_station(snapshot: List[Dict]) -> Dict[str, Dict]:
    """Index a snapshot by station name for O(1) lookup in the UI."""
    return {str(p["station"]): p for p in snapshot}


def relay_for_intent(
    intent: str,
    origin_station: str,
    snapshot: List[Dict],
) -> Optional[Dict]:
    """Dispatch a user intent ("borrow"/"return") to the matching relay function.

    * "borrow" → intervention.recommend_borrow_relay(origin, snapshot)
    * "return" → intervention.recommend_return_relay(origin, snapshot)

    Returns the mission dict, or None when no suitable nearby relay exists. Any
    other intent raises ValueError so a wiring mistake fails loudly rather than
    silently doing nothing.
    """
    if intent == intervention.MISSION_BORROW:
        return intervention.recommend_borrow_relay(origin_station, snapshot)
    if intent == intervention.MISSION_RETURN:
        return intervention.recommend_return_relay(origin_station, snapshot)
    raise ValueError(
        f"Unknown intent {intent!r}; expected "
        f"{intervention.MISSION_BORROW!r} or {intervention.MISSION_RETURN!r}."
    )


# A station is considered "currently normal-looking" when it still has a
# comfortable buffer of bikes/docks right now, so a High 30-min forecast tells a
# genuine "predict ahead" story rather than restating an already-empty station.
_DEMO_NORMAL_MIN = 5


def _demo_has_action(pred: Dict, snapshot: List[Dict]) -> bool:
    """True when a station has SOMETHING to show: a truck option or a relay.

    Reuses the existing intervention APIs only -- no fabricated data. A demo is
    "actionable" if the operator recommend() yields a truck option, OR a Friend
    Relay mission exists for the station in either intent direction.
    """
    rec = intervention.recommend(pred, snapshot)
    if rec["options"]["A_truck"] is not None:
        return True
    station = str(pred["station"])
    borrow = intervention.recommend_borrow_relay(station, snapshot)
    if borrow is not None:
        return True
    ret = intervention.recommend_return_relay(station, snapshot)
    return ret is not None


def _demo_currently_non_extreme_future_high(pred: Dict) -> bool:
    """True when a station looks reasonably normal NOW but is High risk in 30 min.

    "Looks fine now, will run empty":  current_bikes >= 5 and shortage High.
    "Looks fine now, will fill up":    current_docks >= 5 and full risk High.
    Uses only real observed counts + real model probabilities.
    """
    bikes = int(pred["current_bikes"])
    docks = int(pred["current_docks"])
    shortage = float(pred["shortage_prob"])
    full = float(pred["full_prob"])
    will_run_empty = bikes >= _DEMO_NORMAL_MIN and shortage >= config.RISK_HIGH_MIN
    will_fill_up = docks >= _DEMO_NORMAL_MIN and full >= config.RISK_HIGH_MIN
    return will_run_empty or will_fill_up


def find_demo_scenario(
    bundle: features.FeatureBundle,
    artifacts: Dict,
    timestamps: List[pd.Timestamp],
    max_timestamps: int = 24,
) -> Optional[Tuple[pd.Timestamp, str]]:
    """Scan a few timestamps for the most compelling demo (timestamp, station).

    Supports the 60-second demo story (R8.7). The PREFERRED case tells the
    "predict ahead" story best: a station that still looks reasonably normal
    RIGHT NOW (a comfortable buffer of bikes/docks) but is forecast High risk in
    30 minutes AND has an actionable intervention (a truck option or a Friend
    Relay mission). We prefer these over already-extreme stations (e.g. current
    bikes = 1, shortage = 96%), which merely restate the obvious.

    Selection:
      * Preferred pass: return the first station meeting
        "currently non-extreme + future High + actionable". Among candidates at
        the same timestamp we prefer the one that looks MOST clearly normal now
        (highest current bikes/docks buffer) for a stronger story.
      * Fallback: if no such case exists in the scanned range, fall back to the
        original behavior -- the first High-risk station with any actionable
        intervention -- so the demo button always works.

    Kept bounded (``max_timestamps``) so it stays fast enough to run once at
    startup. It always loads a REAL case from the data; it never fabricates
    probabilities or stations.

    Returns (observation_ts, station_name), or None if nothing actionable is
    found across the scanned timestamps.
    """
    fallback: Optional[Tuple[pd.Timestamp, str]] = None

    for ts in timestamps[:max_timestamps]:
        snapshot = build_snapshot(bundle, ts, artifacts)
        if not snapshot:
            continue

        # Preferred candidates at this timestamp, best-looking-now first.
        preferred: List[Tuple[int, str]] = []
        for pred in snapshot:
            if str(pred["risk_level"]) != predict.RISK_HIGH:
                continue
            if not _demo_has_action(pred, snapshot):
                continue
            if fallback is None:
                fallback = (pd.Timestamp(ts), str(pred["station"]))
            if _demo_currently_non_extreme_future_high(pred):
                # Buffer = how normal the current state looks now.
                buffer_now = max(
                    int(pred["current_bikes"]), int(pred["current_docks"])
                )
                preferred.append((buffer_now, str(pred["station"])))

        if preferred:
            # Most clearly "normal now" (largest buffer) first.
            preferred.sort(key=lambda x: -x[0])
            return pd.Timestamp(ts), preferred[0][1]

    return fallback


# --------------------------------------------------------------------------- #
# View-model builders — pure functions that turn backend dicts into the exact  #
# natural-language, audience-appropriate content each tab renders. Keeping     #
# these Streamlit-free lets the whole "what shows on screen" contract be       #
# unit-tested head-less (see tests/validate_task6.py).                         #
# --------------------------------------------------------------------------- #


def _meters_from_km(distance_km: float) -> int:
    """Human-facing metres, rounded to 10 m, derived from a km distance.

    Consistent with intervention._distance_meters so the card text and the
    number we display agree (0.36 km → 360 公尺, 0.51 km → 510 公尺).
    """
    return int(round(float(distance_km) * 1000.0 / 10.0) * 10)


def _distance_phrase(distance_km: float) -> str:
    """Natural distance phrase: metres under 1 km, else one-decimal km."""
    if float(distance_km) < 1.0:
        return f"{_meters_from_km(distance_km)} 公尺"
    return f"{float(distance_km):.1f} 公里"


def risk_level_zh(risk_level: str) -> str:
    """Map a risk-level code to its natural-language label (低/中/高風險)."""
    return _RISK_ZH.get(str(risk_level), str(risk_level))


def build_status_view(pred: Dict) -> Dict:
    """The compact 'current status' the UI shows: bikes / return spaces.

    Returns a small dict of already-formatted, user-friendly fields. total_docks
    is included but treated as optional detail by the renderers.
    """
    return {
        "station": str(pred["station"]),
        "current_bikes": int(pred["current_bikes"]),
        "current_docks": int(pred["current_docks"]),
        "total_docks": int(pred["total_docks"]),
    }


def build_prediction_view(pred: Dict) -> Dict:
    """The 30-minute prediction the UI shows: shortage % / full % / level."""
    return {
        "shortage_pct": round(float(pred["shortage_prob"]) * 100),
        "full_pct": round(float(pred["full_prob"]) * 100),
        "risk_level": str(pred["risk_level"]),
        "risk_level_zh": risk_level_zh(pred["risk_level"]),
        "expected_risk_time": str(pred["expected_risk_time"]),
    }


def build_user_relay_card(intent: str, mission: Optional[Dict]) -> Optional[Dict]:
    """Turn a Friend Relay mission dict into a user-facing card, or None.

    This is the SINGLE source of truth for the user-mode recommendation. When
    ``mission`` is None it returns None, and the renderer must then emit NOTHING
    (no placeholder, no "no adjustment needed" message). When a mission exists it
    produces only natural-language, non-technical fields:

        headline   e.g. "順路多走 260 公尺，獲得 10 元"
                        "多騎 300 公尺，獲得 10 元"
        suggestion e.g. "推薦改從「三信集英路口」借車"
                        "推薦改到「文化三路一段○○站」還車"
        reason     two natural sentences about the predicted risk + why helping
                   works — using the station name, action and the relevant risk.

    No partner_station/mission_type/predicted_problem/risk_probability/
    distance_km/reward_twd/original_station keys are exposed here; they are
    rendered as sentences instead.
    """
    if mission is None:
        return None

    partner = str(mission["partner_station"])
    reward = int(mission["reward_twd"])
    dist = _distance_phrase(mission["distance_km"])
    risk_pct = round(float(mission["risk_probability"]) * 100)

    if intent == intervention.MISSION_BORROW:
        # Borrow redirection: partner is forecast FULL; borrowing there helps.
        headline = f"順路多走 {dist}，獲得 {reward} 元"
        suggestion = f"推薦改從「{partner}」借車"
        reason = (
            f"系統預測該站 30 分鐘後滿站風險較高（約 {risk_pct}%）。"
            f"從該站借走一台車，可以提前降低滿站風險。"
        )
    else:
        # Return redirection: partner is forecast SHORTAGE; returning there helps.
        headline = f"多騎 {dist}，獲得 {reward} 元"
        suggestion = f"推薦改到「{partner}」還車"
        reason = (
            f"系統預測該站 30 分鐘後缺車風險較高（約 {risk_pct}%）。"
            f"將車還到該站，可以提前降低缺車風險。"
        )

    return {
        "intent": intent,
        "headline": headline,
        "suggestion": suggestion,
        "reason": reason,
    }


def build_user_view(intent: str, origin_station: str, snapshot: List[Dict]) -> Dict:
    """The whole user-mode view-model for (intent, origin_station).

    Assembles current status + 30-min prediction for the origin station, plus
    the Friend Relay card (or None). The key contract for the None case:
    ``relay_card`` is None ⇒ the user-mode renderer shows NO relay section.

    Never touches / emits Truck Rebalancing (A_truck) — that stays operator-only.
    """
    by_station = snapshot_by_station(snapshot)
    pred = by_station[str(origin_station)]
    mission = relay_for_intent(intent, str(origin_station), snapshot)
    return {
        "intent": intent,
        "origin_station": str(origin_station),
        "status": build_status_view(pred),
        "prediction": build_prediction_view(pred),
        "relay_card": build_user_relay_card(intent, mission),
    }


def build_operator_truck_view(rec: Dict) -> Optional[Dict]:
    """Natural-language 派車調度 (Truck Rebalancing) line, or None.

    Reads intervention.recommend()'s A_truck. Returns None when no truck action
    applies (control-room 'no dispatch needed' is acceptable in operator mode,
    unlike user mode). Direction (調入 vs 移往) follows which imbalance dominates.
    """
    a_truck = rec["options"].get("A_truck")
    if a_truck is None:
        return None

    shortage = float(rec["shortage_prob"])
    full = float(rec["full_prob"])
    shortage_dominant = shortage >= full
    other = str(a_truck["donor_or_dest"])
    move = int(a_truck["move_bikes"])
    dist = f"{float(a_truck['distance_km']):.1f} km"

    if shortage_dominant:
        risk_line = f"30 分鐘後缺車風險：{round(shortage * 100)}%"
        action_line = f"從「{other}」調入 {move} 台"
    else:
        risk_line = f"30 分鐘後滿站風險：{round(full * 100)}%"
        action_line = f"將 {move} 台移往「{other}」"

    summary = f"{risk_line}｜建議派車調度｜{action_line}｜距離：{dist}"
    return {
        "summary": summary,
        "risk_line": risk_line,
        "action_line": action_line,
        "distance": dist,
    }


def build_operator_view(station: str, snapshot: List[Dict]) -> Dict:
    """The whole operator-mode view-model for one station.

    Includes current status, 30-min prediction, risk level, and the Truck
    Rebalancing recommendation (or None). This is the ONLY place A_truck is read.
    """
    by_station = snapshot_by_station(snapshot)
    pred = by_station[str(station)]
    rec = intervention.recommend(pred, snapshot)
    return {
        "station": str(station),
        "status": build_status_view(pred),
        "prediction": build_prediction_view(pred),
        "truck": build_operator_truck_view(rec),
        "_prediction_raw": pred,
        "_recommendation_raw": rec,
    }


# --------------------------------------------------------------------------- #
# Streamlit UI (imported lazily so the helpers above stay importable head-less)#
# --------------------------------------------------------------------------- #


def _risk_badge(risk_level: str) -> str:
    """Coloured badge string for a risk level, natural Chinese label."""
    dot = {
        predict.RISK_LOW: "🟢",
        predict.RISK_MEDIUM: "🟡",
        predict.RISK_HIGH: "🔴",
    }.get(str(risk_level), "⚪")
    return f"{dot} {risk_level_zh(risk_level)}"


def _render_status_and_prediction(st, view: Dict) -> None:
    """Shared compact renderer: 目前狀態 + 30 分鐘後預測 (used by both tabs)."""
    status = view["status"]
    prediction = view["prediction"]

    st.markdown("**目前狀態**")
    c = st.columns(2)
    c[0].metric("可借車輛", f"{status['current_bikes']} 台")
    c[1].metric("可還空位", f"{status['current_docks']} 格")

    st.markdown("**30 分鐘後預測**")
    p = st.columns(3)
    p[0].metric("缺車風險", f"{prediction['shortage_pct']}%")
    p[1].metric("滿站風險", f"{prediction['full_pct']}%")
    p[2].metric("風險等級", _risk_badge(prediction["risk_level"]))


def _render_user_relay_card(st, card: Optional[Dict]) -> bool:
    """Render the ONE Friend Relay recommendation card, or nothing.

    Returns True if a card was rendered, False if nothing was emitted (None
    mission). When None, this MUST stay completely silent — no placeholder.
    """
    if card is None:
        return False

    with st.container(border=True):
        st.markdown(f"### {card['headline']}")
        st.markdown(f"**{card['suggestion']}**")
        st.write(card["reason"])
        b = st.columns(2)
        if b[0].button("接受推薦", type="primary", key="relay_accept"):
            st.session_state["relay_choice"] = "accepted"
        if b[1].button("維持原定站點", key="relay_keep"):
            st.session_state["relay_choice"] = "kept"

    choice = st.session_state.get("relay_choice")
    if choice == "accepted":
        st.success("已接受推薦，祝你順路愉快，謝謝你幫忙讓車站更順暢！")
    elif choice == "kept":
        st.info("沒問題，將依你原定的站點進行。")
    return True


def _render_user_mode(st, snapshot: List[Dict], station_names: List[str]) -> None:
    """使用者模式 — borrow/return intent + single original-station selector."""
    st.subheader("我要")
    intent_label = st.radio(
        "我要",
        options=[INTENT_BORROW_LABEL, INTENT_RETURN_LABEL],
        horizontal=True,
        label_visibility="collapsed",
        key="user_intent",
    )
    intent = (
        intervention.MISSION_BORROW
        if intent_label == INTENT_BORROW_LABEL
        else intervention.MISSION_RETURN
    )

    # Reset the accept/keep flag whenever the intent changes, so a stale
    # confirmation from a previous mission doesn't linger.
    if st.session_state.get("_last_intent") != intent:
        st.session_state.pop("relay_choice", None)
        st.session_state["_last_intent"] = intent

    st.subheader("原定站點")
    preset = st.session_state.get("preset_station")
    idx = station_names.index(preset) if preset in station_names else 0
    origin_station = st.selectbox(
        "我原本打算前往的 YouBike 站點",
        options=station_names,
        index=idx,
        key="user_origin",
    )

    view = build_user_view(intent, origin_station, snapshot)
    _render_status_and_prediction(st, view)

    # Friend Relay 2.0 — silent unless there is an actionable mission.
    _render_user_relay_card(st, view["relay_card"])


def _render_operator_mode(
    st,
    snapshot: List[Dict],
    station_names: List[str],
    ts_choice: pd.Timestamp,
    backend: str = config.BACKEND_LOCAL,
) -> None:
    """營運中心 — predictive risk + Truck Rebalancing (no user relay controls)."""
    st.caption(
        f"觀測時間：{pd.Timestamp(ts_choice).strftime(_TS_FMT)}"
        f"　·　預測來源：{backend_label(backend)}"
    )

    preset = st.session_state.get("preset_station")
    idx = station_names.index(preset) if preset in station_names else 0
    station = st.selectbox("站點", options=station_names, index=idx, key="op_station")

    view = build_operator_view(station, snapshot)
    status = view["status"]
    prediction = view["prediction"]

    m = st.columns(4)
    m[0].metric("目前可借車輛", f"{status['current_bikes']} 台")
    m[1].metric("目前可還空位", f"{status['current_docks']} 格")
    m[2].metric("30 分鐘後缺車風險", f"{prediction['shortage_pct']}%")
    m[3].metric("30 分鐘後滿站風險", f"{prediction['full_pct']}%")
    st.metric("風險等級", _risk_badge(prediction["risk_level"]))

    st.markdown("**派車調度建議**")
    truck = view["truck"]
    if truck is not None:
        st.warning(truck["summary"])
    else:
        st.caption("此站目前不需要派車調度。")

    # 系統建議 — decision layer on top of Task 5. Operator has no user intent,
    # so Friend Relay is not fabricated (user_context=None).
    decision = intervention.decide_intervention(
        view["_prediction_raw"], snapshot, user_context=None
    )
    action_labels = {
        intervention.DECISION_TRUCK: "派車調度",
        intervention.DECISION_RELAY: "Friend Relay 2.0",
        intervention.DECISION_NONE: "暫不介入",
    }
    st.markdown("**系統建議**")
    primary_label = action_labels.get(decision["primary_action"], "暫不介入")
    primary_line = f"主要措施：{primary_label}"
    if decision["urgency"] == intervention.URGENCY_HIGH:
        st.error(primary_line)
    else:
        st.write(primary_line)
    if decision["secondary_action"] is not None:
        secondary_label = action_labels.get(decision["secondary_action"], "")
        st.write(f"輔助措施：{secondary_label}")
    st.write(f"原因：{decision['reason']}")

    # ---- Optional AI 說明 (explain layer) ------------------------------- #
    # Button-triggered ONLY. The deterministic decision above is already
    # complete; this merely rephrases it. Result is cached in session_state so a
    # Streamlit rerun never issues another request.
    explain_key = f"_explain::{station}"
    if st.button("AI 說明", key=f"explain_btn_{station}"):
        text, source, fallback_reason = explain_layer.explain_decision(
            decision, use_bedrock=config.BEDROCK_ENABLED
        )
        st.session_state[explain_key] = {
            "text": text,
            "source": source,
            "fallback_reason": fallback_reason,
        }

    cached = st.session_state.get(explain_key)
    if cached is not None:
        st.info(cached["text"])
        label = (
            "Amazon Bedrock"
            if cached["source"] == explain_layer.SOURCE_BEDROCK
            else "內建說明模板（未呼叫 Bedrock）"
        )
        st.caption(f"說明來源：{label}")
        if cached["fallback_reason"]:
            st.caption(cached["fallback_reason"])

    if SHOW_DEV_VIEW:
        with st.expander("開發者檢視（原始預測 / 建議資料 §9）"):
            st.write(f"Prediction backend: {backend_label(backend)}")
            st.json(
                {
                    "prediction": view["_prediction_raw"],
                    "recommendation": view["_recommendation_raw"],
                }
            )


def main_v0() -> None:
    """STABLE V0 Streamlit UI (XGBoost shortage/full). Kept intact as fallback."""
    import streamlit as st

    st.set_page_config(page_title="YouBike 預測式調度 Demo", page_icon="🚲", layout="wide")
    st.title("🚲 YouBike 預測式供需調度系統")
    st.caption("預測 30 分鐘後站點供需，提前透過使用者引導與營運派車降低失衡。")

    # ---- Load artifacts + data (cached) ---------------------------------- #
    @st.cache_resource(show_spinner="載入訓練好的模型 ...")
    def _load_artifacts():
        return predict.load_artifacts()

    @st.cache_resource(show_spinner="讀取資料並建立特徵 ...")
    def _load_bundle():
        return build_feature_bundle()

    try:
        artifacts = _load_artifacts()
    except predict.ModelArtifactsMissingError as exc:
        st.error(
            "尚未找到訓練好的模型。請先執行訓練："
            "\n\n```\npython -m src.train\n```\n\n"
            f"詳細：{exc}"
        )
        st.stop()
        return

    bundle = _load_bundle()
    if bundle.X.empty:
        st.error("資料集沒有可用的 30 分鐘目標樣本，無法產生預測。")
        st.stop()
        return

    timestamps = available_timestamps(bundle)

    @st.cache_data(show_spinner="建立整網預測快照 ...")
    def _snapshot_cached(ts_iso: str, backend: str) -> List[Dict]:
        return build_snapshot(
            bundle, pd.Timestamp(ts_iso), artifacts, backend=backend
        )

    @st.cache_data(show_spinner="尋找示範情境 ...")
    def _demo_scenario():
        scenario = find_demo_scenario(bundle, artifacts, timestamps)
        return None if scenario is None else (scenario[0].isoformat(), scenario[1])

    # ---- Sidebar: developer prediction-backend switch -------------------- #
    # Default stays 本機模型 so the stable v0 demo is never broken by AWS state.
    st.sidebar.header("開發者選項")
    backend_choice_label = st.sidebar.radio(
        "預測來源",
        options=[BACKEND_LABEL_LOCAL, BACKEND_LABEL_SAGEMAKER],
        index=0,
        key="prediction_backend",
        help="本機模型直接載入 models/ 推論；AWS SageMaker 呼叫已部署的 endpoint。"
             "兩者的特徵、風險門檻與介入規則完全相同。",
    )
    backend = (
        config.BACKEND_SAGEMAKER
        if backend_choice_label == BACKEND_LABEL_SAGEMAKER
        else config.BACKEND_LOCAL
    )

    # Probe the endpoints before using them, and fall back rather than crash.
    if backend == config.BACKEND_SAGEMAKER:
        ok, message = sagemaker_predict.check_endpoints()
        if ok:
            st.sidebar.success("SageMaker Endpoint 就緒")
        else:
            st.sidebar.warning(
                f"無法使用 SageMaker，已自動切回本機模型。\n\n{message}"
            )
            backend = config.BACKEND_LOCAL

    st.sidebar.caption(f"目前預測來源：{backend_label(backend)}")

    # ---- Sidebar: observation time + high-risk demo scenario ------------- #
    st.sidebar.header("示範控制")
    scenario = _demo_scenario()
    default_ts_idx = 0
    if scenario is not None and st.sidebar.button("🎬 載入高風險示範"):
        preset_ts = pd.Timestamp(scenario[0])
        if preset_ts in timestamps:
            default_ts_idx = timestamps.index(preset_ts)
        st.session_state["preset_station"] = scenario[1]
        st.session_state["preset_ts_idx"] = default_ts_idx

    if scenario is not None:
        st.sidebar.caption(
            f"示範情境：時間 {pd.Timestamp(scenario[0]).strftime(_TS_FMT)}、"
            f"站點「{scenario[1]}」現在看起來還正常，但 30 分鐘後會出問題，"
            f"且有可行的調整。"
        )
    else:
        st.sidebar.caption("（在掃描範圍內未找到特別強的高風險情境，仍可自由選擇。）")

    default_ts_idx = st.session_state.get("preset_ts_idx", default_ts_idx)
    ts_choice = st.sidebar.selectbox(
        "觀測時間（示範的「現在」）",
        options=timestamps,
        index=default_ts_idx,
        format_func=lambda t: pd.Timestamp(t).strftime(_TS_FMT),
    )

    try:
        snapshot = _snapshot_cached(pd.Timestamp(ts_choice).isoformat(), backend)
    except sagemaker_predict.EndpointUnavailableError as exc:
        # An endpoint died mid-session: warn, fall back to local, keep running.
        st.warning(
            f"SageMaker 推論失敗，已改用本機模型繼續示範。\n\n詳細：{exc}"
        )
        backend = config.BACKEND_LOCAL
        snapshot = _snapshot_cached(
            pd.Timestamp(ts_choice).isoformat(), backend
        )

    if not snapshot:
        st.warning("此時間點沒有可用的站點資料，請選擇另一個時間。")
        st.stop()
        return

    station_names = sorted(snapshot_by_station(snapshot).keys())

    # ---- Two top-level tabs: separate the two audiences ------------------ #
    user_tab, operator_tab = st.tabs(["使用者模式", "營運中心"])

    with user_tab:
        _render_user_mode(st, snapshot, station_names)

    with operator_tab:
        _render_operator_mode(
            st, snapshot, station_names, pd.Timestamp(ts_choice), backend
        )


def main() -> None:
    """Entry point: V1 (AWS SageMaker multimodel) with explicit V0 fallback.

    The V1 UI is the demo path. If the V1 endpoint is unavailable the user is
    told explicitly and the stable V0 UI is rendered instead -- never a silent
    fallback and never a crash.
    """
    import streamlit as st

    st.set_page_config(page_title="YouBike 預測式供需調度系統",
                       page_icon="🚲", layout="wide")

    from src import v1_app, v1_predict

    try:
        v1_app.render(st)
        return
    except v1_predict.V1EndpointError as exc:
        st.warning(
            "V1 SageMaker Endpoint unavailable，已切換至 Stable V0。\n\n"
            f"詳細：{exc}")
    except FileNotFoundError as exc:
        st.warning(
            "V1 Demo 情境資料尚未建立，已切換至 Stable V0。\n\n"
            f"詳細：{exc}")

    # ---- explicit fallback to the stable V0 UI --------------------------
    st.info("以下為 Stable V0（本機 XGBoost）介面。")
    main_v0()


if __name__ == "__main__":
    main()
