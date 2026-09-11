# Tasks — YouBike 預測式供需調度系統 (V1 MVP)

優先級定義：
- **P0** = 必須完成才能 Demo（構成完整 vertical slice）
- **P1** = MVP 跑通後才做
- **P2** = 有額外時間才做

> 執行順序：先完成 Task 1 → 6 形成完整 vertical slice，再考慮 P1 / P2。

---

- [x] 1. Data loading + cleaning + 30-minute target 【P0】
  - **Goal**: 從單一 cp950 CSV 產生乾淨、排序、含 30 分鐘 shortage/full target 的資料表。
  - **Files**: `src/config.py`, `src/data_loader.py`, `src/features.py`（`build_targets`）
  - **Implementation**:
    - `config.py`：集中 encoding、DEFAULT_CSV、REQUIRED_COLUMNS、TARGET_MIN/MAX_MINUTES、SHORTAGE/FULL_THRESHOLD、路徑等。
    - `data_loader.load_clean()`：cp950 讀取 → required column 檢查 → rename 為內部英文欄名 → `timestamp` 轉 datetime、數值欄 coerce → drop 無效列 → 依 (station, timestamp) 排序。
    - `features.build_targets()`：groupby station 排序 → `shift(-1)` 取 future → 計算 `delta_min` → 僅保留 25–35 分 pair → 計算 `shortage_target`、`full_target`。
  - **Test / validation**: cp950 讀取成功且無亂碼；required columns 存在；timestamp 正確；排序正確；約 60 分 gap 不被當成 target；25–35 分 pair 正確；label 值正確。
  - **Acceptance**: 對應 R1、R2 全部 criteria。輸出的 DataFrame 含有效 target 樣本數 > 0。
  - **Dependencies**: 無（起點）。不修改原始 CSV。

- [x] 2. Minimal feature engineering 【P0】
  - **Goal**: 建立容易解釋的時間 / 站點 / 庫存 / lag 特徵矩陣，無時間洩漏。
  - **Files**: `src/features.py`（`build_features`）
  - **Implementation**: 時間特徵（hour/weekday/is_weekend）；站點特徵（station id、district encode、total_docks、lon、lat）；庫存特徵（available_bikes/docks、bike_ratio、dock_ratio）；lag（同站 shift(1)：prev_inventory、bike_change、dock_change）。輸出 X + y_shortage + y_full + timestamp。
  - **Test / validation**: 特徵無使用未來資訊；ratio 分母為 0 時安全處理；lag 對每站首列 NaN 有處理。
  - **Acceptance**: 對應 R3 criteria 1–5、7。
  - **Dependencies**: Task 1。

- [x] 3. Baseline + XGBoost 【P0】
  - **Goal**: 以時間切分訓練 baseline 與 XGBoost，分別預測 shortage / full，輸出可理解指標。
  - **Files**: `src/train.py`, 產物於 `models/`, `outputs/`
  - **Implementation**: 依 timestamp 前 TRAIN_FRACTION 為 train、其餘 test（禁止 random split）；簡單 baseline + 兩個 `XGBClassifier`；`predict_proba` 取正類；處理類別不平衡（scale_pos_weight）；輸出 precision/recall/F1/ROC-AUC/confusion matrix，存模型、feature_meta、metrics.json、圖。
  - **Test / validation**: 模型能 train 完成；機率落在 [0,1]；train/test 邊界依時間；metrics 檔產生。
  - **Acceptance**: 對應 R4 全部 criteria。
  - **Dependencies**: Task 2。

- [x] 4. Prediction output 【P0】
  - **Goal**: 對單站當前狀態輸出 shortage/full 機率與風險分級。
  - **Files**: `src/predict.py`
  - **Implementation**: 載入 model + feature_meta → 重建特徵 → `predict_proba` → shortage_prob/full_prob → 依 config 門檻做 Low/Medium/High 分級（取兩者較高者）→ 回傳結構化 dict（含 expected_risk_time = 當前 + 30min）。
  - **Test / validation**: 機率在合理範圍；risk level 分級正確；模型缺失時明確提示。
  - **Acceptance**: 對應 R5 全部 criteria。
  - **Dependencies**: Task 3。

- [x] 5. Intervention heuristic + Friend Relay 2.0 【P0】
  - **Goal**: 對高風險站產出 rule-based recommendation（Option A/B/C）與 predictive incentive mission card。
  - **Files**: `src/intervention.py`
  - **Implementation**: Haversine（純 numpy）；Option A（Truck Rebalancing，營運端）維持不變：shortage 高 → 找 donor station、full 高 → 找 alternative destination。Friend Relay 2.0 改為**使用者意圖**為中心的兩個明確函式：`recommend_borrow_relay(origin, snapshot)`（借車轉向：附近 Medium/High 滿站站點，排除高缺車站）與 `recommend_return_relay(origin, snapshot)`（還車轉向：附近 Medium/High 缺車站點，排除高滿站站點）；各回傳 mission dict（original_station/partner_station/mission_type/predicted_problem/risk_probability/distance_km/reward_twd/mission_text）或 None。recommend() 的 B_relay 由此使用者意圖邏輯供給（目標站為 beneficiary），保留 §9 既有 keys 並加上新欄位。reward tier 5/10/15 heuristic（依 risk/severity/distance），Low → Option C。
  - **Test / validation**: 借車 mission 選附近未來滿站站、不選高缺車站；還車 mission 選附近未來缺車站、不選高滿站站；partner 距離 <= MAX_NEIGHBOR_KM；reward 只在 {5,10,15}；無合適鄰站時回傳 None。既有 Option A/C 檢查維持通過。
  - **Acceptance**: 對應 R6、R7 全部 criteria。
  - **Dependencies**: Task 4。

- [x] 6. Streamlit end-to-end demo 【P0】
  - **Goal**: 一頁 Streamlit 展示完整 pipeline，支援 60 秒 demo story。
  - **Files**: `app.py`, `README.md`, `requirements.txt`
  - **Implementation**: 以兩個 top-level tab 分開兩種對象，共用同一套 30 分鐘風險預測（自然繁體中文 UI，不放中英對照冗字、不暴露開發用詞）：
    - **使用者模式**：「我要（借車 / 還車）」意圖 + **單一**「原定站點」選擇器 → 顯示目前狀態（可借車輛 / 可還空位）與 30 分鐘後預測（缺車 / 滿站風險 %、風險等級）。Friend Relay 2.0 借車呼叫 `intervention.recommend_borrow_relay(origin, snapshot)`、還車呼叫 `intervention.recommend_return_relay(origin, snapshot)`；**僅在有可行任務時**顯示一張自然語句推薦卡（站名 + 動作 + 繞行公尺 + 獎勵 + 相關風險）與示範按鈕（接受推薦 / 維持原定站點，不涉帳號 / 付款）；任務為 `None` 時**完全不顯示**任何 Friend Relay 區塊或提示。此模式**不**顯示派車調度。
    - **營運中心**：觀測時間、站點選擇、目前可借 / 可還、30 分鐘後缺車 / 滿站風險、風險等級，以及**派車調度**建議（`intervention.recommend()` 的 `A_truck`：自然語句方向 / 台數 / 距離；無需派車時中性訊息）；可保留收合的開發者檢視（§9 raw json）。此模式**不**混入使用者借還車控制項。
    - 呈現邏輯抽為 Streamlit-free view-model builder（`build_user_view` / `build_operator_view` 等）以利 head-less 測試；以 `st.cache_data`/`st.cache_resource` 一次載入模型與建立整網快照；「載入高風險示範」以 `find_demo_scenario` 從真實資料挑真實可行動情境。README 說明如何啟動（使用者自行啟動，不由本 spec 執行）。
  - **Test / validation**: helper 與 view-model builder 可 head-less import 測試（`tests/validate_task6.py`）；使用者借車 / 還車正確導出狀態與預測、有任務時建卡、`None` 時渲染路徑不輸出任何內容；任務方向正確（借車 → 附近未來滿站站、還車 → 附近未來缺車站）；使用者模式從不渲染派車調度、營運中心才渲染派車且無借還車控制項；無冗餘中英對照標籤 / 開發用詞；`?寮公園` 字元問題已釐清（原始來源即為字面 `?`）；regression `python -m tests.validate_task5` 仍通過；app 能啟動（smoke，手動）。
  - **Acceptance**: 對應 R8 criteria 1–5、7。
  - **Dependencies**: Task 5。

---

## P1（vertical slice 跑通後）

- [x] 7. 關鍵邏輯自動化測試補強 【P1】
  - **Goal**: 用 pytest 固化最重要邏輯，防止 regression。
  - **Files**: `tests/test_data_loader.py`, `tests/test_features.py`, `tests/test_intervention.py`
  - **Implementation**: 涵蓋 §12 測試策略：cp950 讀取、target gap 排除、label 正確、機率範圍、intervention 產出。
  - **Acceptance**: 核心 case 通過；不建立大型 framework。
  - **Dependencies**: Task 1–5。

- [ ] 8. Streamlit 簡單地圖 + 資料擴充 【P1 / P2】
  - **Goal**: 提升 demo 說服力與資料量。
  - **Files**: `app.py`, `src/data_loader.py`, `src/config.py`
  - **Implementation**:
    - 【P1】UI 加入簡單 map 顯示 station location（對應 R8 criteria 6，非阻塞）。
    - 【P2】擴充載入更多／連續多份 CSV（pipeline 跑通後才做，仍避免一次載入全六個月）。
  - **Acceptance**: 地圖顯示站點；擴充資料後 pipeline 仍成功。
  - **Dependencies**: Task 6。

---

## Future Work（不進本次 P0/P1/P2，僅記錄）

擴充至全六個月、counterfactual historical replay、估算 avoided lost demand、dynamic incentive optimization、weather/MRT/POI features、station neighborhood / graph features、GNN、fairness / service equity、AWS SageMaker、Amazon Bedrock、Agent-based decision explanation。
