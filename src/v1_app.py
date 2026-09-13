# -*- coding: utf-8 -*-
"""V1 Streamlit UI: user mode + operator centre, backed by the SageMaker V1 endpoint.

All rendering lives here so app.py stays a thin entry point and the V0 stable UI
functions remain untouched for fallback.

Data: a real historical demo snapshot (the V1 models need rainfall + is_peak,
which the V0 live snapshot does not carry). The UI always states the scenario
timestamp and never claims live data.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import v1_decision as vd
from . import v1_predict as vp
from . import v1_ui as ui

RUNTIME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "v1_training", "runtime")
SNAPSHOTS = {"weekday": "demo_weekday_snapshot.parquet",
             "weekend": "demo_weekend_snapshot.parquet"}

# Text-input presets. Kept because users type landmarks, not station names.
# Coverage note: the dataset is 新北市, so a 台北市 landmark resolves to the
# nearest covered cluster rather than being faked.
PRESETS = {
    "SOGO 忠孝館": (25.041, 121.5435),
    "三井 Outlet 林口": (25.07158, 121.36629),
    "捷運江子翠站": (25.03043, 121.47162),
    "板橋車站": (25.0143, 121.4636),
}

TITLE = "YouBike 預測式供需調度系統"
SUBTITLE = ("運用 AI 預測未來 30 分鐘至 1 小時的站點供需變化，"
            "透過使用者協作與營運調度降低供需失衡。")


# --------------------------------------------------------------------------- #
# data loading (cached)
# --------------------------------------------------------------------------- #
def load_snapshot(day_type: str) -> pd.DataFrame:
    p = os.path.join(RUNTIME, SNAPSHOTS[day_type])
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"demo snapshot missing: {p}. Build it with "
            f"v1_training/build_demo_snapshot.py")
    return pd.read_parquet(p)


def snapshot_meta() -> Dict:
    p = os.path.join(RUNTIME, "demo_snapshot_meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def demo_cases() -> Dict:
    p = os.path.join(RUNTIME, "demo_cases.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _feature_rows(df: pd.DataFrame, day_type: str) -> List[Dict]:
    feats = vp.FEATURE_ORDER[day_type]
    return [{c: float(r[c]) for c in feats} for _, r in df.iterrows()]


def predict_snapshot(day_type: str) -> List[Dict]:
    """Score the WHOLE snapshot with exactly two endpoint calls."""
    df = load_snapshot(day_type)
    return vp.predict_batch(_feature_rows(df, day_type), day_type)


def resolve_destination(text: str, df: pd.DataFrame) -> Tuple[str, float, float, str]:
    """Deterministic local resolution: preset, then station-name match.

    No external geocoding, so there is no network single point of failure.
    Returns (label, lat, lon, how).
    """
    q = (text or "").strip()
    if not q:
        name = "三井 Outlet 林口"
        lat, lon = PRESETS[name]
        return name, lat, lon, "default"
    for name, (lat, lon) in PRESETS.items():
        if q == name or q.replace(" ", "") == name.replace(" ", ""):
            return name, lat, lon, "preset"
    hit = df[df["station"].astype(str).str.contains(q, case=False, na=False)]
    if not hit.empty:
        r = hit.iloc[0]
        return f"{r['station']}（站點）", float(r["lat"]), float(r["lon"]), "station_match"
    for name, (lat, lon) in PRESETS.items():
        if q in name or name in q:
            return name, lat, lon, "preset_partial"
    name = "三井 Outlet 林口"
    lat, lon = PRESETS[name]
    return name, lat, lon, "fallback_no_match"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _risk_badge(level: str) -> str:
    return f"{ui.RISK_EMOJI.get(level,'')} {level}"


def render_user_mode(st, day_type: str, snapshot: pd.DataFrame,
                     predictions: List[Dict], scenario_ts: str) -> None:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("我要")
        mode_label = st.radio(
            "我要", options=["我要借車", "我要還車"], index=1,
            horizontal=True, label_visibility="collapsed", key="v1_mode")
        mode = ui.MODE_BORROW if mode_label == "我要借車" else ui.MODE_RETURN

        st.subheader("我的目的地")
        dest_text = st.text_input(
            "我的目的地", value="", key="v1_dest",
            placeholder="輸入地址或地標，例如 SOGO 忠孝館",
            label_visibility="collapsed")
        radius = st.select_slider(
            "搜尋範圍", options=ui.RADIUS_CHOICES, value=ui.DEFAULT_RADIUS_M,
            format_func=lambda v: f"{v} m" if v < 1000 else f"{v/1000:g} km",
            key="v1_radius")

        label, dlat, dlon, how = resolve_destination(dest_text, snapshot)
        st.caption(f"目的地：{label}")
        if how in ("fallback_no_match", "default"):
            st.caption("（找不到相符地標，已使用資料涵蓋範圍內的預設地點）")

    recs = snapshot.to_dict(orient="records")
    for r, p in zip(recs, predictions):
        r["_pred"] = p
    near = ui.nearby_stations(recs, dlat, dlon, radius)

    with right:
        if near:
            map_df = pd.DataFrame(
                [{"lat": dlat, "lon": dlon}]
                + [{"lat": n["lat"], "lon": n["lon"]} for n in near[:30]])
            st.map(map_df, size=30, zoom=14)
        else:
            st.info("此範圍內沒有站點，請放大搜尋範圍。")

    st.divider()
    if not near:
        return

    rows = ui.build_user_rows(near, [n["_pred"] for n in near], mode,
                              max_rows=ui.MAX_USER_STATIONS)
    meaning = rows[0]["risk_meaning"] if rows else ""
    st.markdown(f"**附近站點**　·　風險代表「{meaning}」")

    table = []
    for r in rows:
        table.append({
            "站點名稱": r["station"] + ("（最近）" if r["is_baseline"] else ""),
            "距離": f"{r['distance_m']:.0f}m",
            "可借/可還": ui.availability_text(r, mode),
            "30分鐘風險": _risk_badge(r["risk_30m"]),
            "60分鐘風險": _risk_badge(r["risk_60m"]),
            "獎勵金": ui.reward_text(r),
        })
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.caption("回饋金依額外繞行距離計算，Demo 示意。")
    st.caption(f"Demo 情境時間：{scenario_ts}")


def render_operator_mode(st, day_type: str, snapshot: pd.DataFrame,
                         predictions: List[Dict], scenario_ts: str) -> None:
    recs = snapshot.to_dict(orient="records")
    statics = []
    for r in recs:
        statics.append({k: float(r[k]) for k in ui_static_keys() if k in r})
    rows = ui.build_operator_rows(recs, predictions, statics)
    ranked = ui.rank_operator_rows(rows)
    kpi = ui.operator_kpis(rows)

    c = st.columns(4)
    c[0].metric("高風險站點數", kpi["high_risk_stations"])
    c[1].metric("持續空車風險站", kpi["persistent_empty"])
    c[2].metric("持續滿車風險站", kpi["persistent_full"])
    c[3].metric("目前監控站點", kpi["monitored_stations"])

    st.divider()
    f = st.columns([1, 1, 1.4])
    districts = ["全部"] + sorted({r["district"] for r in ranked if r["district"]})
    pick_district = f[0].selectbox("行政區", districts, key="v1_op_district")
    pick_risk = f[1].selectbox("風險程度", ["全部", "高", "中", "低"], key="v1_op_risk")
    query = f[2].text_input("站點搜尋", "", key="v1_op_q",
                            placeholder="輸入站點名稱關鍵字")

    view = ranked
    if pick_district != "全部":
        view = [r for r in view if r["district"] == pick_district]
    if pick_risk != "全部":
        view = [r for r in view
                if pick_risk in (r["full_risk_30m"], r["full_risk_60m"],
                                 r["empty_risk_30m"], r["empty_risk_60m"])]
    if query.strip():
        q = query.strip()
        view = [r for r in view if q in str(r["station"])]

    st.markdown(f"**站點風險列表**（依風險排序，共 {len(view)} 站）")
    table = [{
        "站點名稱": r["station"],
        "站點地址": r["address"],
        "30分鐘滿車風險": _risk_badge(r["full_risk_30m"]),
        "1小時滿車風險": _risk_badge(r["full_risk_60m"]),
        "30分鐘空車風險": _risk_badge(r["empty_risk_30m"]),
        "1小時空車風險": _risk_badge(r["empty_risk_60m"]),
    } for r in view[:300]]
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

    persistent = [r for r in view if vd.is_persistent(r["risk_state"])][:8]
    if persistent:
        with st.expander(f"持續性風險站點說明（{len(persistent)} 站）"):
            for r in persistent:
                st.write(f"**{r['station']}** — {r['risk_state_zh']}")
                st.caption("30 分鐘與 60 分鐘後皆預測處於該風險區間，"
                           "因此判定具有持續性風險。"
                           + ("　理由：" + "；".join(r["reasons"]) if r["reasons"] else ""))
    st.caption(f"Demo 情境時間：{scenario_ts}")


def ui_static_keys() -> List[str]:
    return ["nearest_junior_high_distance", "nearest_university_distance",
            "nearest_mrt_distance", "nearest_bus_distance"]


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def render(st) -> None:
    """Render the whole V1 app. Raises EndpointUnavailableError for fallback."""
    st.title(f"🚲 {TITLE}")
    st.caption(SUBTITLE)

    meta = snapshot_meta()
    cases = demo_cases()

    @st.cache_data(show_spinner="載入 Demo 情境資料 ...")
    def _snap(day_type: str):
        return load_snapshot(day_type)

    @st.cache_data(show_spinner="呼叫 AWS SageMaker V1 進行預測 ...")
    def _preds(day_type: str, n_rows: int):
        # n_rows is part of the cache key so a snapshot change invalidates it.
        return predict_snapshot(day_type)

    ok, msg = vp.check_endpoint()
    if not ok:
        raise vp.EndpointUnavailableError(msg)

    with st.sidebar:
        st.header("Demo 控制")
        day_label = st.radio("情境日型", ["平日（尖峰）", "假日"], index=0,
                             key="v1_day")
        day_type = "weekday" if day_label.startswith("平日") else "weekend"
        with st.expander("開發者選項"):
            st.write("預測來源：**AWS SageMaker V1**")
            st.write(f"Endpoint：`{vp.ENDPOINT_NAME}`")
            st.write(f"Region：{vp.REGION}")
            st.caption(f"day_type_source = {vp.DAY_TYPE_SOURCE}")
            st.caption("國定假日尚未做 runtime 判斷（已知限制）。")
            for k in ("DEMO_CASE_USER_RETURN", "DEMO_CASE_USER_BORROW"):
                if cases.get(k):
                    cc = cases[k]
                    st.caption(f"{k}: {cc['destination']} → "
                               f"{cc['reward_station']} +${cc['reward_twd']}")

    snapshot = _snap(day_type)
    scenario_ts = meta.get(day_type, {}).get("timestamp", "（未知）")
    predictions = _preds(day_type, len(snapshot))

    st.success(f"預測來源：AWS SageMaker V1　·　Demo 情境時間：{scenario_ts}"
               f"　·　監控站點 {len(snapshot)} 站")

    user_tab, op_tab = st.tabs(["使用者模式", "營運中心"])
    with user_tab:
        render_user_mode(st, day_type, snapshot, predictions, scenario_ts)
    with op_tab:
        render_operator_mode(st, day_type, snapshot, predictions, scenario_ts)
