# Requirements — YouBike 預測式供需調度系統 (V1 MVP)

## 專案背景

2026 新北市 AI 智慧城市黑客松「YouBike 公共自行車營運調度」題目。

現有 YouBike App 已能告訴使用者「現在」某站可借幾台、可還幾格。因此本專案**不**解決「使用者不知道目前站點狀態」的問題。

真正要解決的問題是：

> **現在的狀態，不等於使用者 15–30 分鐘後抵達時的狀態。**

範例：某站現在還有 8 台車看似正常，但模型預測 30 分鐘後有 85% 機率接近缺車。系統應在真正缺車**之前**預警並提供 intervention。

核心概念：**Predict → Prevent → Rebalance**

產品定位（暫定）：YouBike 預測式供需調度系統 / Agent。第一版**不**實作複雜 AI Agent，先把完整 decision pipeline 跑通。

## 已確認資料特性 (Confirmed，不需再次 probe)

以 `新北AWS黑克松競賽0329-31.csv 的副本.csv`（2026-03-29 ~ 2026-03-31）為代表：

| 項目 | 值 |
|---|---|
| encoding | cp950 |
| columns | 日期, 城市, 行政區, 場站名稱, 總車柱數, 可借車數, 可還位數, 經度, 緯度 |
| row count | 219,076 |
| station count | 1,532 |
| distinct timestamp count | 567 |
| min timestamp | 2026-03-29 00:00:51 |
| max timestamp | 2026-03-31 23:30:42 |
| per-station interval | min ≈ 29.1 分、median ≈ 29.983 分、most common = 30 分 |
| 已知 gap | 部分資料存在約 60 分鐘 gap |

資料本質是**站點庫存 snapshot**，不是 individual trip / OD transaction data。

## MVP 第一原則

先跑通完整 vertical slice，再提升模型與功能。第一個 milestone：

```
Raw CSV → clean data → features → 30-minute target → baseline → XGBoost
        → prediction → intervention recommendation → Streamlit screen
```

在這條 pipeline 完整成功前，不加入其他功能、不過度重構、不加不必要 abstraction。第一版只用一份日期連續的較小 CSV：`新北AWS黑克松競賽0329-31.csv 的副本.csv`。

---

## 功能需求 (EARS 格式)

### R1 — 資料載入與清理 (P0)

**User Story:** 作為系統開發者，我要能穩定讀取原始 cp950 CSV 並轉成乾淨的資料表，以便後續建模。

#### Acceptance Criteria
1. WHEN 系統讀取指定 CSV THEN 系統 SHALL 以 `cp950` encoding 正確解碼，不得出現亂碼。
2. IF CSV 缺少任一 required column（日期, 城市, 行政區, 場站名稱, 總車柱數, 可借車數, 可還位數, 經度, 緯度）THEN 系統 SHALL 拋出明確錯誤。
3. WHEN 解析「日期」欄位 THEN 系統 SHALL 轉為 datetime 型別。
4. WHEN 遇到數值欄位無法解析或缺漏 THEN 系統 SHALL 以明確策略處理（drop 或標記），不得靜默產生 NaN 汙染後續步驟。
5. WHEN 清理完成 THEN 系統 SHALL 依 (場站, timestamp) 排序輸出。
6. 系統 SHALL NOT 修改、移動或重新命名任何原始 CSV。

### R2 — 30 分鐘 Target 建立 (P0)

**User Story:** 作為模型使用者，我要一個正確、無時間洩漏的 30 分鐘後風險標籤。

#### Acceptance Criteria
1. WHEN 為每個 station 建立 future state THEN 系統 SHALL 依 timestamp 排序並使用「下一筆 observation」作為未來狀態。
2. WHEN 計算 `future_timestamp - current_timestamp` THEN 系統 SHALL 只接受 interval 約 **25–35 分鐘** 的 pair 作為 30-minute target。
3. IF 相鄰 observation 間隔約 60 分鐘（gap）THEN 系統 SHALL NOT 將其視為 30-minute target（該筆不產生 label）。
4. WHEN 建立 `shortage_target` THEN 系統 SHALL 定義為 `future 可借車數 <= SHORTAGE_THRESHOLD`（預設 2）。
5. WHEN 建立 `full_target` THEN 系統 SHALL 定義為 `future 可還位數 <= FULL_THRESHOLD`（預設 2）。
6. 所有 threshold 與 interval 邊界 SHALL 集中於 config / constants，不得散落硬編碼於各程式。

### R3 — 特徵工程 (P0)

**User Story:** 作為建模者，我要一組容易解釋、容易完成的特徵。

#### Acceptance Criteria
1. 系統 SHALL 產生時間特徵：hour、weekday、is_weekend。
2. 系統 SHALL 產生站點特徵：station identity、行政區(district)、總車柱數、經度、緯度。
3. 系統 SHALL 產生目前庫存特徵：current available bikes、current available return spaces、bike availability ratio、return-space ratio。
4. 系統 SHALL 產生時序特徵：previous inventory、recent bike change、recent return-space change、簡單 lag features。
5. IF 加入更長時窗（5/15/30/60 分鐘概念）歷史特徵 THEN 系統 SHALL 依實際 snapshot interval 正確處理，SHALL NOT 假設資料每 5 分鐘一筆。
6. Spatial neighbor features 非 P0；IF 加入 THEN 只用經緯度做非常簡單的鄰近計算，SHALL NOT 建立 GNN。
7. 特徵計算 SHALL NOT 使用未來資訊（避免 leakage）。

### R4 — 模型 (Baseline + XGBoost) (P0)

**User Story:** 作為營運分析者，我要能預測某站 30 分鐘後的缺車 / 滿站機率。

#### Acceptance Criteria
1. 系統 SHALL 提供 A) 簡單 baseline 與 B) XGBoost classifier，分別預測 shortage probability 與 full probability。
2. WHEN 切分 train/test THEN 系統 SHALL 依**時間**切分，SHALL NOT 使用 random split（避免 temporal leakage）。
3. WHEN 評估 THEN 系統 SHALL 輸出可理解的分類指標：precision、recall、F1、ROC-AUC（若合理）、confusion matrix。
4. 系統 SHALL 特別關注高風險事件的 recall（能否抓到）。
5. 系統 SHALL NOT 使用 deep learning / GNN / RL。
6. WHEN 訓練完成 THEN 系統 SHALL 將模型與 metrics 保存至 `models/` 與 `outputs/`。

### R5 — Prediction 輸出 (P0)

**User Story:** 作為下游模組，我要一個結構化的預測輸出。

#### Acceptance Criteria
1. WHEN 對某 station 目前狀態做預測 THEN 系統 SHALL 輸出 shortage probability 與 full probability。
2. WHEN 輸出機率 THEN 值 SHALL 落在 [0, 1] 合理範圍。
3. 系統 SHALL 依機率提供風險分級 Low / Medium / High（門檻集中於 config）。

### R6 — Intervention Engine (Rule-based) (P0)

**User Story:** 作為營運人員，我要在高風險站點得到具體可執行的處置建議。

#### Acceptance Criteria
1. WHEN 某 station 屬高風險 THEN 系統 SHALL 至少輸出：station、current bikes、current return spaces、shortage probability、full probability、expected risk time、recommended action。
2. Action SHALL 分為 Option A（Truck Rebalancing）、Option B（Friend Relay 2.0 / 使用者協作）、Option C（No Intervention）。
3. IF shortage risk 高 THEN 系統 SHALL 用簡單 heuristic 尋找附近（距離合理、bikes 較充足、自身 shortage risk 較低）的 donor station。
4. IF full risk 高 THEN 系統 SHALL 尋找附近（return spaces 較多、自身 full risk 較低）的 alternative destination。
5. 距離計算 SHALL 使用經緯度 + Haversine distance。
6. 系統 SHALL NOT 求解真正的 vehicle routing problem。

### R7 — Friend Relay 2.0 (User-Intent Predictive Incentive) (P0，隨 Intervention 一起)

**User Story:** 作為原本就打算借車 / 還車的使用者，我要在系統預測附近站點即將失衡時，得到一張「順路即可完成」的微型 incentive 任務卡，改到鄰近站點借 / 還以幫助紓解。

> Friend Relay 2.0 以**使用者意圖**為中心，MVP 只涵蓋兩個情境：借車轉向、還車轉向。它與 Option A（Truck Rebalancing，營運端派車）為**各自獨立**的 intervention 選項。

#### Acceptance Criteria
1. WHEN 系統預測站點即將失衡 THEN 系統 SHALL 對「原本就在附近移動的使用者」提供 predictive incentive（提前於真正滿/空之前）。
2. **情境一（借車轉向 `recommend_borrow_relay`）**：WHEN 使用者原本要在 A 站借車 AND A 站 `MAX_NEIGHBOR_KM` 內有一站 B 被預測為 Medium/High **滿站(full)** 風險 THEN 系統 SHALL 建議改到 B 站借車（自 B 借走一台可降低 B 的未來滿站風險），mission 文案例如「改到 B 站借車｜額外步行 250m｜獲得 10 元」。
3. **情境二（還車轉向 `recommend_return_relay`）**：WHEN 使用者原本要在 C 站還車 AND C 站 `MAX_NEIGHBOR_KM` 內有一站 D 被預測為 Medium/High **缺車(shortage)** 風險 THEN 系統 SHALL 建議改到 D 站還車（還一台到 D 可降低 D 的未來缺車風險），mission 文案例如「改到 D 站還車｜額外騎行 300m｜獲得 10 元」。
4. **傷害防護 (harm-avoidance)**：系統 SHALL NOT 將借車使用者導向本身已屬**高缺車**風險（`shortage_prob >= RISK_HIGH_MIN`）的站點；SHALL NOT 將還車使用者導向本身已屬**高滿站**風險（`full_prob >= RISK_HIGH_MIN`）的站點。
5. 「Medium/High 風險」以該情境的相關機率判定（借車看 `full_prob`、還車看 `shortage_prob`），門檻為 `>= RISK_LOW_MAX`（即 predict.py 的 Medium 或 High）。
6. 每張 mission SHALL 至少回傳欄位：`original_station`、`partner_station`、`mission_type`（"borrow"/"return"）、`predicted_problem`（"full"/"shortage"）、`risk_probability`、`distance_km`、`reward_twd`、`mission_text`。partner station SHALL 在 `MAX_NEIGHBOR_KM` 內。
7. IF 找不到合適的鄰近站點 THEN 系統 SHALL 回傳 None（不虛構任務）。IF 有多個候選 THEN 系統 SHALL 選最具紓解效益者（相關風險機率最高），並以距離較近者破平手。
8. reward SHALL 使用簡單 tier：5 / 10 / 15 TWD，依 risk probability 與 detour distance 做簡單 heuristic（永遠是 {5,10,15} 之一）。
9. 系統 SHALL NOT 使用真實帳號、payment integration、coupon API 或真正發錢。
10. 系統 SHALL NOT 建立複雜 incentive optimization model。

### R8 — Streamlit MVP UI (P0)

**User Story:** 作為 demo 者，我要一個能展示完整 pipeline 的 Streamlit 畫面，並清楚區分「一般使用者」與「營運人員」兩種對象。

> UI 分為兩個 top-level tab，共用同一套 30 分鐘風險預測：
> **使用者模式**（一般民眾，Friend Relay 2.0）與 **營運中心**（營運人員 / 評審，派車調度）。
> 使用者模式**不**顯示派車調度；營運中心**不**混入使用者借還車協作控制項。
> 所有主要可見 UI 文字為自然繁體中文（不放冗餘中英對照標籤、不暴露開發用詞）。

#### Acceptance Criteria
1. UI SHALL 提供兩個 top-level tab（使用者模式 / 營運中心），並以簡短說明點出共用概念。
2. 使用者模式 SHALL 以「我要（借車 / 還車）」意圖出發，並提供**單一**「原定站點」選擇器（代表使用者原本打算前往的站點）。
3. 使用者模式 SHALL 顯示該站的目前狀態（可借車輛、可還空位）與 30 分鐘後預測（缺車風險 %、滿站風險 %，可含低/中/高風險等級）。
4. 使用者模式的 Friend Relay 2.0 SHALL 以借車呼叫 `recommend_borrow_relay`、還車呼叫 `recommend_return_relay`；**僅在有可行任務時**顯示一張自然語句的推薦卡（含站名、動作、繞行公尺、獎勵、相關風險），並提供「接受推薦 / 維持原定站點」示範按鈕（不涉帳號 / 付款）。IF 結果為 None THEN UI SHALL 不顯示任何 Friend Relay 區塊或訊息（不打擾使用者）。
5. 營運中心 SHALL 顯示觀測時間、站點選擇、目前可借車輛 / 可還空位、30 分鐘後缺車 / 滿站風險、風險等級，以及**派車調度**建議（來自 `intervention.recommend()` 的 `A_truck`，以自然語句呈現方向 / 台數 / 距離）；無需派車時可顯示中性訊息。營運中心 MAY 保留一個收合的開發者檢視（§9 raw json）。
6. IF 容易實作 THEN UI MAY 顯示簡單 map；map 非阻塞 P0 vertical slice 的必要條件。
7. UI SHALL 能支援約 60 秒的 demo story（現在正常 → 30 分鐘後高機率缺車 / 滿站 → 提前 intervention）；SHALL 提供「載入高風險示範」以真實資料（`find_demo_scenario`）載入一個真實、可行動的高風險情境（不虛構機率 / 站點）。

---

## 非功能需求

- **環境**：Python 3.13.15、`.venv` 已備妥（pandas、numpy、scikit-learn、xgboost、streamlit、matplotlib）。本 spec 階段不安裝套件。
- **資料完整性**：原始 CSV 唯讀，不得修改。
- **可調性**：所有 threshold / interval / risk 分級門檻集中於單一 config。
- **可解釋性**：優先使用容易解釋的特徵與規則，重於追求 SOTA accuracy。
- **無雲端 / 無成本**：本專案 V1 不使用任何 AWS 或會產生成本的雲端資源。

## Out of Scope (V1)

AWS deployment、Bedrock、SageMaker、AgentCore、GNN、deep learning、RL、weather/MRT/POI/traffic API、複雜 vehicle routing 最佳化、真實帳號、真實付款/coupon、real-time production API、counterfactual simulation、fairness optimization、pipeline 跑通前載入全部六個月資料、microservices、Docker（除非絕對必要）。

## Future Phases (僅記錄，不進 P0)

Phase 2 候選：擴充至全六個月、counterfactual historical replay、估算 avoided lost demand、dynamic incentive optimization、weather/MRT/POI features、station neighborhood / graph features、GNN、fairness / service equity、AWS SageMaker、Amazon Bedrock、Agent-based decision explanation。
