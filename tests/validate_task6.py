# -*- coding: utf-8 -*-
"""Validation script for Task 6 (R8) — the redesigned Streamlit UX.

Run from the project root:
    python -m tests.validate_task6

Read-only, model-free. It exercises the importable view-model builders and the
render helpers in app.py against the Task 6 UX contract, using tiny hand-built
§9 snapshots (identical shape to predict.predict_station output, same style as
tests/validate_task5.py) so it needs no trained models and never launches a
Streamlit server.

Checks (mapped to the task's A–H validation list):
  A. User borrow: status + prediction derive correctly; a mission → card data is
     built; None mission → NO Friend Relay section/message is emitted.
  B. User return: same via recommend_return_relay.
  C. Mission direction: borrow → nearby future-FULL station; return → nearby
     future-SHORTAGE station.
  D. User mode never renders Truck Rebalancing (render helper emits no A_truck /
     派車 content, and build_user_view carries no truck field).
  E. Operator mode renders Truck Rebalancing when applicable; the operator view
     is the only one carrying truck data; no borrow/return relay controls.
  F. Text: no redundant CN-EN duplicate labels; no developer keys exposed to
     normal users (scan of user-mode strings in app.py).
  G. Character issue "?寮公園": documented in the report (raw source already
     contains a literal '?'); asserted the pipeline does NOT further mangle a
     station name it is given.
  H. Regression is run separately (python -m tests.validate_task5).
"""

from __future__ import annotations

import re
import sys

import app
from src import config, intervention


class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, name, condition, detail=""):
        self.checks += 1
        status = "PASS" if condition else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line)
        if not condition:
            self.failures.append(name)

    def done(self):
        print("-" * 60)
        if self.failures:
            print(f"RESULT: {len(self.failures)}/{self.checks} checks FAILED: "
                  f"{self.failures}")
            return 1
        print(f"RESULT: all {self.checks} checks PASSED")
        return 0


def _mk(station, lat, lon, bikes, docks, sp, fp, risk):
    return {
        "station": station,
        "lat": lat,
        "lon": lon,
        "current_bikes": bikes,
        "current_docks": docks,
        "total_docks": bikes + docks,
        "shortage_prob": sp,
        "full_prob": fp,
        "risk_level": risk,
        "expected_risk_time": "2026-03-30T08:30:00",
    }


class FakeCol:
    """Records metric()/write()/markdown() calls made on a column."""
    def __init__(self, sink):
        self._sink = sink

    def metric(self, label, value, *a, **k):
        self._sink.append(("metric", str(label), str(value)))

    def button(self, label, *a, **k):
        self._sink.append(("button", str(label)))
        return False

    def write(self, *a, **k):
        self._sink.append(("write", " ".join(str(x) for x in a)))

    def markdown(self, *a, **k):
        self._sink.append(("markdown", " ".join(str(x) for x in a)))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSt:
    """A minimal Streamlit stand-in that records everything written.

    Only the calls the render helpers use are implemented. Every visible string
    is captured in ``self.texts`` so tests can assert what did / did not render.
    """
    def __init__(self):
        self.texts = []
        self.session_state = {}
        self.calls = []

    # --- text-emitting widgets ---
    def _record(self, kind, *a):
        self.calls.append(kind)
        for x in a:
            if isinstance(x, str):
                self.texts.append(x)

    def markdown(self, *a, **k):
        self._record("markdown", *a)

    def write(self, *a, **k):
        self._record("write", *a)

    def caption(self, *a, **k):
        self._record("caption", *a)

    def subheader(self, *a, **k):
        self._record("subheader", *a)

    def success(self, *a, **k):
        self._record("success", *a)

    def info(self, *a, **k):
        self._record("info", *a)

    def warning(self, *a, **k):
        self._record("warning", *a)

    def error(self, *a, **k):
        self._record("error", *a)

    def json(self, *a, **k):
        self._record("json")

    def metric(self, label, value, *a, **k):
        self.calls.append("metric")
        self.texts.append(str(label))
        self.texts.append(str(value))

    def button(self, label, *a, **k):
        self.calls.append("button")
        self.texts.append(str(label))
        return False

    # --- layout ---
    def columns(self, spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [FakeCol(_ColSink(self)) for _ in range(n)]

    def container(self, *a, **k):
        return _CtxCatcher(self)

    def expander(self, *a, **k):
        self._record("expander", *a)
        return _CtxCatcher(self)

    def radio(self, label, options, *a, **k):
        self.calls.append("radio")
        return options[0]

    def selectbox(self, label, options, *a, **k):
        self.calls.append("selectbox")
        return options[0]


class _ColSink(list):
    """A list whose appends also flow into the parent FakeSt text log."""
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

    def append(self, item):
        super().append(item)
        # item like ("metric", label, value) or ("write", text)
        for x in item[1:]:
            self._parent.texts.append(str(x))
        self._parent.calls.append(item[0])


class _CtxCatcher:
    def __init__(self, parent):
        self._parent = parent

    def __enter__(self):
        return self._parent

    def __exit__(self, *a):
        return False


def main() -> int:
    c = Checker()

    # =====================================================================
    # A. User BORROW flow
    # =====================================================================
    # User plans to borrow at U. N_full is nearby (~0.35 km) and forecast FULL
    # (Medium/High) -> borrow-relay should redirect there.
    borrow_snap = [
        _mk("U", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("三信集英路口", 25.0030, 121.5010, 20, 1, 0.05, 0.72, "High"),  # ~0.35km
    ]
    uview = app.build_user_view(intervention.MISSION_BORROW, "U", borrow_snap)

    c.check("A status derives current bikes/docks",
            uview["status"]["current_bikes"] == 6
            and uview["status"]["current_docks"] == 6)
    c.check("A prediction derives shortage/full pct + level",
            uview["prediction"]["shortage_pct"] == 10
            and uview["prediction"]["full_pct"] == 10
            and uview["prediction"]["risk_level_zh"] == "低風險")
    c.check("A borrow mission → relay card built (not None)",
            uview["relay_card"] is not None)
    card = uview["relay_card"]
    c.check("A borrow card names the partner station in the suggestion",
            card is not None and "三信集英路口" in card["suggestion"], card["suggestion"] if card else None)
    c.check("A borrow card mentions 借車 + 滿站風險 (natural sentences)",
            card is not None and "借車" in card["suggestion"] and "滿站風險" in card["reason"])
    c.check("A borrow card headline uses 公尺 + 元 (meters under 1km, reward)",
            card is not None and "公尺" in card["headline"] and "元" in card["headline"],
            card["headline"] if card else None)

    # A (None case): borrow origin with only a calm neighbor -> no mission ->
    # the render helper MUST emit nothing.
    calm_snap = [
        _mk("U2", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_calm", 25.0030, 121.5010, 8, 8, 0.10, 0.10, "Low"),
    ]
    uview_none = app.build_user_view(intervention.MISSION_BORROW, "U2", calm_snap)
    c.check("A None mission → relay_card is None", uview_none["relay_card"] is None)

    fst = FakeSt()
    rendered = app._render_user_relay_card(fst, uview_none["relay_card"])
    c.check("A None mission → render helper returns False (nothing rendered)",
            rendered is False)
    c.check("A None mission → NO text emitted at all (silent)",
            len(fst.texts) == 0 and len(fst.calls) == 0,
            f"texts={fst.texts}")

    # =====================================================================
    # B. User RETURN flow
    # =====================================================================
    return_snap = [
        _mk("R", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("文化三路一段站", 25.0030, 121.5010, 1, 20, 0.70, 0.05, "High"),  # ~0.35km
    ]
    rview = app.build_user_view(intervention.MISSION_RETURN, "R", return_snap)
    c.check("B return mission → relay card built", rview["relay_card"] is not None)
    rcard = rview["relay_card"]
    c.check("B return card names partner + 還車 + 缺車風險",
            rcard is not None and "文化三路一段站" in rcard["suggestion"]
            and "還車" in rcard["suggestion"] and "缺車風險" in rcard["reason"])
    c.check("B return card headline uses 多騎 + 公尺 + 元",
            rcard is not None and "多騎" in rcard["headline"]
            and "公尺" in rcard["headline"])

    # =====================================================================
    # C. Mission direction
    # =====================================================================
    borrow_mission = intervention.recommend_borrow_relay("U", borrow_snap)
    return_mission = intervention.recommend_return_relay("R", return_snap)
    c.check("C borrow → nearby FUTURE-FULL partner",
            borrow_mission is not None
            and borrow_mission["predicted_problem"] == "full"
            and borrow_mission["partner_station"] == "三信集英路口")
    c.check("C return → nearby FUTURE-SHORTAGE partner",
            return_mission is not None
            and return_mission["predicted_problem"] == "shortage"
            and return_mission["partner_station"] == "文化三路一段站")

    # =====================================================================
    # D. User mode NEVER renders Truck Rebalancing
    # =====================================================================
    # A high-risk station with an actionable truck option; render the whole user
    # tab and assert no 派車/truck content leaks through.
    truck_snap = [
        _mk("T_short", 25.0000, 121.5000, 1, 20, 0.85, 0.05, "High"),
        _mk("T_donor", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low"),
    ]
    # Sanity: the operator recommendation for this station DOES have a truck.
    op_probe = app.build_operator_view("T_short", truck_snap)
    c.check("D (setup) operator truck exists for the probe station",
            op_probe["truck"] is not None)

    # build_user_view must carry no truck field at all.
    uview_probe = app.build_user_view(intervention.MISSION_BORROW, "T_short", truck_snap)
    c.check("D user view-model has no 'truck' key", "truck" not in uview_probe)

    fst2 = FakeSt()
    fst2.session_state["preset_station"] = None
    app._render_user_mode(fst2, truck_snap, ["T_short", "T_donor"])
    joined = "\n".join(fst2.texts)
    forbidden_user = ["派車", "調度", "Truck", "A_truck", "donor", "move_bikes",
                      "Option", "shortage_prob", "full_prob", "partner_station"]
    leaked = [w for w in forbidden_user if w in joined]
    c.check("D user mode renders no operator/dev terms", not leaked, str(leaked))

    # =====================================================================
    # E. Operator mode renders Truck Rebalancing; carries truck data
    # =====================================================================
    fst3 = FakeSt()
    app._render_operator_mode(fst3, truck_snap, ["T_short", "T_donor"], __import__("pandas").Timestamp("2026-03-30T08:00:00"))
    op_joined = "\n".join(fst3.texts)
    c.check("E operator mode shows 派車調度", "派車調度" in op_joined)
    c.check("E operator truck summary uses 調入 (shortage-dominant direction)",
            op_probe["truck"] is not None and "調入" in op_probe["truck"]["summary"],
            op_probe["truck"]["summary"] if op_probe["truck"] else None)
    # Full-dominant direction check.
    full_snap = [
        _mk("F_full", 25.0100, 121.5100, 22, 0, 0.05, 0.88, "High"),
        _mk("F_dest", 25.0120, 121.5110, 2, 19, 0.05, 0.05, "Low"),
    ]
    op_full = app.build_operator_view("F_full", full_snap)
    c.check("E operator full-dominant uses 移往",
            op_full["truck"] is not None and "移往" in op_full["truck"]["summary"],
            op_full["truck"]["summary"] if op_full["truck"] else None)
    # No user borrow/return relay controls in operator flow: radio only appears
    # in user mode. The operator render used no radio call.
    c.check("E operator mode has no borrow/return intent radio",
            "radio" not in fst3.calls)

    # Operator "no dispatch" acceptable message when no truck.
    calm_op = app.build_operator_view("N_calm", calm_snap)
    c.check("E operator with no truck → truck is None (subtle msg allowed)",
            calm_op["truck"] is None)

    # =====================================================================
    # F. Text cleanup — scan app.py user-mode strings
    # =====================================================================
    src = open("app.py", encoding="utf-8").read()
    # Redundant bilingual labels that must be gone from UI text.
    banned_labels = [
        "目前狀態 Current status", "站點選擇 Station selector",
        "可借車數 current bikes", "可還位數 return spaces",
        "總車柱數 total docks", "缺車機率 shortage prob",
        "滿站機率 full prob", "風險分級 risk level",
        "介入建議 Intervention recommendation", "營運端派車 Truck Rebalancing",
        "原始站點 original station", "夥伴站 partner", "預測問題 problem",
        "風險機率 risk prob", "繞行距離 detour", "獎勵 reward",
        "原始站 origin", "Station selector", "Current status",
    ]
    present = [b for b in banned_labels if b in src]
    c.check("F no redundant bilingual UI labels remain", not present, str(present))

    # =====================================================================
    # G. Character handling — pipeline does not further mangle a given name
    # =====================================================================
    weird_snap = [
        _mk("?寮公園", 25.0000, 121.5000, 1, 20, 0.85, 0.05, "High"),
        _mk("鄰站", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low"),
    ]
    g_view = app.build_operator_view("?寮公園", weird_snap)
    c.check("G station name passes through view-model unchanged",
            g_view["station"] == "?寮公園")

    # =====================================================================
    # Distance-phrase formatting (meters under 1km, one-decimal km otherwise)
    # =====================================================================
    c.check("dist 0.36km → 360 公尺", app._distance_phrase(0.36) == "360 公尺")
    c.check("dist 0.51km → 510 公尺", app._distance_phrase(0.51) == "510 公尺")
    c.check("dist 1.2km → 1.2 公里", app._distance_phrase(1.2) == "1.2 公里")

    # =====================================================================
    # Subtitle text: exactly the simplified one-liner, no extra clutter.
    # =====================================================================
    expected_subtitle = "預測 30 分鐘後站點供需，提前透過使用者引導與營運派車降低失衡。"
    c.check("subtitle is exactly the simplified sentence",
            expected_subtitle in src, expected_subtitle)
    c.check("old long shared-concept subtitle removed",
            "用原本就會發生的借還車行程，小幅繞行 + 獎勵做預防式微調度" not in src)

    # =====================================================================
    # Dev view is hidden by default in the normal demo.
    # =====================================================================
    c.check("SHOW_DEV_VIEW flag exists and is False by default",
            getattr(app, "SHOW_DEV_VIEW", None) is False)
    # Rendering the operator mode must NOT emit the developer expander while the
    # flag is off.
    fst_dev = FakeSt()
    app._render_operator_mode(
        fst_dev, truck_snap, ["T_short", "T_donor"],
        __import__("pandas").Timestamp("2026-03-30T08:00:00"),
    )
    dev_joined = "\n".join(fst_dev.texts)
    c.check("dev view (§9 raw) hidden in normal demo",
            "開發者檢視" not in dev_joined and "expander" not in fst_dev.calls,
            dev_joined)

    # =====================================================================
    # find_demo_scenario: prefers "currently non-extreme + future High +
    # actionable" over an already-extreme High-risk station; falls back
    # gracefully when no preferred case exists. Model-free via a controllable
    # stub bundle + monkeypatched build_snapshot.
    # =====================================================================
    class _StubBundle:
        """Minimal stand-in; find_demo_scenario only forwards it to build_snapshot."""

    def _run_selector(snapshots_by_ts, timestamps):
        """Run find_demo_scenario with build_snapshot stubbed to return canned
        snapshots keyed by timestamp. Restores the original afterwards."""
        import pandas as _pd
        original = app.build_snapshot

        def _fake_build_snapshot(bundle, ts, artifacts):
            return snapshots_by_ts.get(_pd.Timestamp(ts), [])

        app.build_snapshot = _fake_build_snapshot
        try:
            return app.find_demo_scenario(_StubBundle(), {}, timestamps)
        finally:
            app.build_snapshot = original

    import pandas as _pd

    ts0 = _pd.Timestamp("2026-03-30T08:00:00")

    # Snapshot with BOTH an already-extreme High station (bikes=1, shortage=0.96)
    # and a "normal now, future High" shortage station (bikes>=5, shortage High)
    # that also has a nearby donor (so it is actionable). The preferred one wins.
    pref_donor = _mk("PREF_donor", 25.0030, 121.5010, 20, 3, 0.05, 0.10, "Low")
    pref_normal = _mk("PREF_normal", 25.0000, 121.5000, 7, 6, 0.80, 0.05, "High")
    extreme = _mk("EXTREME", 25.0500, 121.5500, 1, 20, 0.96, 0.05, "High")
    extreme_donor = _mk("EXTREME_donor", 25.0530, 121.5510, 18, 3, 0.05, 0.10, "Low")
    mixed_snap = [extreme, extreme_donor, pref_normal, pref_donor]
    result_pref = _run_selector({ts0: mixed_snap}, [ts0])
    c.check("selector PREFERS currently-normal + future-High station",
            result_pref is not None and result_pref[1] == "PREF_normal",
            str(result_pref))

    # Fallback: only an already-extreme High-risk actionable station exists ->
    # selector still returns it (button keeps working).
    only_extreme = [extreme, extreme_donor]
    result_fb = _run_selector({ts0: only_extreme}, [ts0])
    c.check("selector FALLS BACK to actionable High-risk station when no preferred",
            result_fb is not None and result_fb[1] == "EXTREME",
            str(result_fb))

    # No actionable station at all -> None.
    calm_only = [
        _mk("C1", 25.0000, 121.5000, 8, 8, 0.10, 0.10, "Low"),
        _mk("C2", 25.0030, 121.5010, 8, 8, 0.10, 0.10, "Low"),
    ]
    result_none = _run_selector({ts0: calm_only}, [ts0])
    c.check("selector returns None when nothing is actionable",
            result_none is None, str(result_none))

    return c.done()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
