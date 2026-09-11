# Design — YouBike 預測式供需調度系統 (V1 MVP)

## 1. 總覽

本設計實作一條完整的 vertical slice decision pipeline，將站點庫存 snapshot 轉為「30 分鐘後缺車/滿站風險預測」，再產出 rule-based intervention 與 Friend Relay 2.0 incentive，最後透過 Streamlit 展示。

設計原則：**單向資料流、模組職責清楚、threshold 集中、避免時間洩漏、不過度抽象**。

```
CSV (cp950)
  └─ data_loader.load_clean()        → 乾淨 DataFrame（排序、型別正確）
       └─ features.build_targets()   → 30-min shortage/full target（僅 25–35 分 pair）
            └─ features.build_features() → 時間/站點/庫存/lag 特徵
                 └─ train.train()     → baseline + XGBoost，時間切分，存 model + metrics
                      └─ predict.predict_station() → shortage/full 機率 + 風險分級
                           └─ intervention.recommend() → Option A/B/C + Friend Relay card
                                └─ app.py (Streamlit) → Demo 畫面
```

## 2. 資料流與資料契約

### 2.1 原始欄位（cp950）
`日期, 城市, 行政區, 場站名稱, 總車柱數, 可借車數, 可還位數, 經度, 緯度`

### 2.2 內部標準化欄位（英文，避免下游處理中文欄名易錯）
`data_loader` 讀取後統一 rename：

| 原始 | 內部欄名 |
|---|---|
| 日期 | `timestamp` (datetime) |
| 城市 | `city` |
| 行政區 | `district` |
| 場站名稱 | `station` |
| 總車柱數 | `total_docks` (int) |
| 可借車數 | `available_bikes` (int) |
| 可還位數 | `available_docks` (int) |
| 經度 | `lon` (float) |
| 緯度 | `lat` (float) |

### 2.3 清理策略
- encoding 固定 `cp950`（config 可覆寫）。
- required column 檢查：缺任一則 raise `ValueError`。
- `timestamp` → `pd.to_datetime`；無法解析的列 drop 並記錄數量。
- 數值欄位 → `pd.to_numeric(errors="coerce")`，coerce 後仍為 NaN 的關鍵列 drop。
- 依 `["station", "timestamp"]` 排序後回傳。

## 3. 30 分鐘 Target 設計（核心，避免洩漏）

對每個 station（groupby `station`，先排序）：
1. 以 `shift(-1)` 取「下一筆」作為 future observation：`future_timestamp`, `future_available_bikes`, `future_available_docks`。
2. 計算 `delta_min = (future_timestamp - timestamp) / 60s`。
3. 只保留 `TARGET_MIN_MINUTES <= delta_min <= TARGET_MAX_MINUTES`（預設 25–35）的列作為有效 target 樣本；其餘（含約 60 分 gap）不產生 label。
4. Label：
   - `shortage_target = (future_available_bikes <= SHORTAGE_THRESHOLD)`（預設 2）
   - `full_target = (future_available_docks <= FULL_THRESHOLD)`（預設 2）

> 因為 target 用「未來下一筆」，特徵只能用「當前與過去」，天然避免前視洩漏。

## 4. 特徵設計

以「當前列 + 同站過去列」計算，全部 groupby `station` 後在時間序上操作。

**時間**：`hour`, `weekday`, `is_weekend`
**站點**：`station`（label-encode 或直接作 categorical id）、`district`（encode）、`total_docks`、`lon`、`lat`
**目前庫存**：`available_bikes`、`available_docks`、`bike_ratio = available_bikes / total_docks`、`dock_ratio = available_docks / total_docks`
**時序 / lag**（同站 `shift(1)`）：`prev_available_bikes`、`bike_change = available_bikes - prev_available_bikes`、`dock_change = available_docks - prev_available_docks`
**（可選，若容易）** 較長時窗歷史特徵：以實際 timestamp 為準的滾動聚合（例如過去 N 分鐘均值），**不假設固定 5 分鐘頻率**。

特徵矩陣輸出 `X`（數值/encoded）與兩個 label 向量 `y_shortage`, `y_full`，以及保留 `timestamp` 供時間切分。

## 5. 模型設計

### 5.1 Baseline
- 簡單規則 / 多數類或 logistic-style baseline：例如「以當前 bike_ratio 是否低於門檻」直接預測，作為對照。輸出機率（可用簡單 sigmoid on ratio 或 sklearn `DummyClassifier` / `LogisticRegression`）。

### 5.2 XGBoost
- 兩個獨立 `XGBClassifier`：`shortage_model`、`full_model`，`predict_proba` 取正類機率。
- 類別不平衡：以 `scale_pos_weight` 或簡單參數處理，重點提升高風險 recall。

### 5.3 時間切分（禁止 random split）
- 依 `timestamp` 排序後，前 X%（config，如 0.7）為 train，其餘為 test。
- 三天資料：可用前兩天 train、最後一天 test（依 config 邊界）。

### 5.4 評估與產物
- 指標：precision、recall、F1、ROC-AUC、confusion matrix，分別對 shortage 與 full。
- 產物：
  - `models/shortage_xgb.json`、`models/full_xgb.json`（或 pickle）
  - `models/feature_meta.json`（欄位順序、encoder 對照）
  - `outputs/metrics.json`、`outputs/*.png`（confusion matrix / ROC，matplotlib）

## 6. Prediction 設計

`predict.py` 提供對「單站當前狀態」或「一批狀態」的預測：
- 載入模型與 feature meta → 重建特徵 → `predict_proba` → 回傳 `shortage_prob`, `full_prob`。
- 風險分級（config 門檻，預設）：
  - `prob < 0.3` → Low
  - `0.3 <= prob < 0.6` → Medium
  - `prob >= 0.6` → High
- 取 shortage/full 兩者較高者決定整體 risk level。

## 7. Intervention Engine（rule-based heuristic）

`intervention.py`：

輸入：目標站的 `station, lat, lon, available_bikes, available_docks, shortage_prob, full_prob, expected_risk_time`，以及全站當前快照（含各站機率，用於挑鄰居）。

流程：
1. 依 risk level 決定是否介入；Low → Option C (No Intervention)。
2. **Shortage 高**：用 Haversine 找 `MAX_NEIGHBOR_KM` 內、`available_bikes` 較充足且自身 `shortage_prob` 低的 donor station → 產出 Option A（Truck Rebalancing：從 donor 移動 N 台）與 Option B（Friend Relay：鼓勵使用者從 donor 借、還到本站附近）。
3. **Full 高**：找附近 `available_docks` 較多且自身 `full_prob` 低的 alternative destination → Option A（把本站車移往它站）與 Option B（引導使用者改還該站）。
4. 輸出結構化 recommendation（見 §9 資料結構）。

Haversine：標準球面距離公式，純 numpy，無外部套件。

## 8. Friend Relay 2.0（User-Intent Predictive Incentive）

`intervention.py` 內的 incentive 子邏輯，以**使用者意圖**為中心。它從「使用者本來就要做的事」（在某站借車 / 還車）出發，若鄰近站點被預測即將失衡，就把同一趟行程「順路」導向能紓解網路的鄰站。這與 §7 的 Truck Rebalancing（營運端派車）為**各自獨立**的選項，§7 描述維持不變。

MVP 只涵蓋兩個情境，各由一個明確函式實作：

**情境一 — 借車轉向 `recommend_borrow_relay(origin_station, station_snapshot)`**
- 使用者原本要在 A 站借車。
- 在 A 站 `MAX_NEIGHBOR_KM` 內搜尋鄰站；若鄰站 B 被預測為 Medium/High **滿站(full)** 風險，建議改到 B 借車 —— 自 B 借走一台可降低 B 的未來滿站風險。
- 文案例如：「改到 B 站借車｜額外步行 250m｜獲得 10 元」。
- **傷害防護**：不得導向本身高缺車（`shortage_prob >= RISK_HIGH_MIN`）的站點。

**情境二 — 還車轉向 `recommend_return_relay(origin_station, station_snapshot)`**
- 使用者原本要在 C 站還車。
- 在 C 站 `MAX_NEIGHBOR_KM` 內搜尋鄰站；若鄰站 D 被預測為 Medium/High **缺車(shortage)** 風險，建議改到 D 還車 —— 還一台到 D 可降低 D 的未來缺車風險。
- 文案例如：「改到 D 站還車｜額外騎行 300m｜獲得 10 元」。
- **傷害防護**：不得導向本身高滿站（`full_prob >= RISK_HIGH_MIN`）的站點。

判定與選擇規則：
- 「Medium/High 風險」以該情境的相關機率判定（借車看 `full_prob`、還車看 `shortage_prob`），門檻 `>= RISK_LOW_MAX`。
- 傷害防護以「反向風險」的 `>= RISK_HIGH_MIN` 排除候選。
- 距離換算：`distance_km` 轉為公尺（四捨五入至 10m）供文案「額外步行/騎行 Xm」使用。
- 多候選時選相關風險機率最高者，距離較近者破平手；partner 必在 `MAX_NEIGHBOR_KM` 內。
- 找不到合適鄰站則回傳 `None`（不虛構任務）。

Mission dict 欄位（兩情境相同）：`original_station`、`partner_station`、`mission_type`（"borrow"/"return"）、`predicted_problem`（"full"/"shortage"）、`risk_probability`、`distance_km`、`reward_twd`、`mission_text`。

reward tier heuristic（config `REWARD_TIERS = [5, 10, 15]`）：
- 沿用 `_severity_score`（結合 risk prob 與 detour 距離）映射到 tier；reward 永遠是 {5,10,15} 之一。
- 不涉及帳號、付款、coupon、發錢。

## 9. 主要資料結構

```python
# Prediction result
{
  "station": str, "lat": float, "lon": float,
  "current_bikes": int, "current_docks": int, "total_docks": int,
  "shortage_prob": float, "full_prob": float,
  "risk_level": "Low|Medium|High",
  "expected_risk_time": str  # 例如 "約 08:30" / current_ts + 30min
}

# Intervention recommendation
{
  "station": str,
  "current_bikes": int, "current_docks": int,
  "shortage_prob": float, "full_prob": float,
  "expected_risk_time": str,
  "recommended_action": "A|B|C",
  "options": {
    "A_truck": {"donor_or_dest": str, "distance_km": float, "move_bikes": int} | None,
    # B_relay 保留原 §9 keys（mission_text/reward_twd/partner_station），
    # 並「加上」使用者意圖 Friend Relay 2.0 欄位（見下）。§7 flow 中：
    #   shortage 高的目標站 → B_relay 為 return-relay，beneficiary 即目標站；
    #   full 高的目標站     → B_relay 為 borrow-relay，beneficiary 即目標站。
    "B_relay": {"mission_text": str, "reward_twd": int, "partner_station": str,
                "mission_type": "borrow|return", "predicted_problem": "full|shortage",
                "original_station": str, "risk_probability": float,
                "distance_km": float} | None,
    "C_none": bool
  }
}

# Friend Relay 2.0 mission（recommend_borrow_relay / recommend_return_relay 直接回傳）
{
  "original_station": str,
  "partner_station": str,
  "mission_type": "borrow|return",
  "predicted_problem": "full|shortage",
  "risk_probability": float,
  "distance_km": float,
  "reward_twd": int,          # 5 / 10 / 15
  "mission_text": str
}  # 或 None（無合適鄰站）
```

## 10. 專案結構

```
hackathon/
├─ dataset/                     # 原始 CSV（唯讀，不修改）
│  └─ probe_data.py             # 前期檢查用，非 production
├─ src/
│  ├─ config.py                 # 所有 threshold / interval / 路徑 / risk 門檻 / reward tiers
│  ├─ data_loader.py            # R1
│  ├─ features.py               # R2 target + R3 features
│  ├─ train.py                  # R4
│  ├─ predict.py                # R5
│  └─ intervention.py           # R6 + R7
├─ models/                      # 訓練產物
├─ outputs/                     # metrics / 圖
├─ tests/                       # 關鍵邏輯測試
├─ app.py                       # R8 Streamlit
├─ requirements.txt
└─ README.md
```

> 保持簡單：不為 architecture 建大量空 module。target 與 feature 皆放 `features.py`。

### 10.1 Streamlit UI 角色設計（R8）

`app.py` 以兩個 top-level tab 分開兩種對象，共用同一套 30 分鐘風險預測：

- **使用者模式**（一般民眾）：以「我要借車 / 還車」意圖出發 → 單一「原定站點」選擇器 → 顯示該站目前狀態（可借車輛 / 可還空位）與 30 分鐘後預測（缺車 / 滿站風險 %、風險等級）。Friend Relay 2.0 以借車呼叫 `recommend_borrow_relay`、還車呼叫 `recommend_return_relay`，**僅在有可行任務時**渲染一張自然語句推薦卡（站名 + 動作 + 繞行公尺 + 獎勵 + 相關風險）與示範按鈕（接受推薦 / 維持原定站點，不涉帳號 / 付款）。任務為 `None` 時**完全不顯示**任何 Friend Relay 區塊或提示。此模式**不**顯示派車調度。
- **營運中心**（營運人員 / 評審）：觀測時間、站點選擇、目前可借 / 可還、30 分鐘後缺車 / 滿站風險、風險等級，以及**派車調度**建議（`intervention.recommend()` 的 `A_truck`：以自然語句呈現方向 / 台數 / 距離；無需派車時顯示中性訊息）。可保留收合的開發者檢視（§9 raw json）。此模式**不**混入使用者借還車協作控制項。

UI 之外的呈現邏輯（狀態 / 預測 / 推薦卡 / 派車句子的組裝）抽為 Streamlit-free 的 view-model builder（`build_user_view` / `build_operator_view` 等），可 head-less 測試（`tests/validate_task6.py`）。所有主要可見文字為自然繁體中文，不放中英對照冗字、不暴露開發用詞（origin/partner/shortage/full/Option A/B/C 等）。「載入高風險示範」以 `find_demo_scenario` 從真實資料挑一個真實、可行動的高風險情境（不虛構）。

> 資料備註：代表 CSV（`新北AWS黑克松競賽*`）為 cp950，其中某站名在原始資料即為字面 `?寮公園`（原字元無法以 cp950 表示，來源已以 `?` 取代）；pipeline 忠實讀取，不猜測補字、不修改原始 CSV。

## 11. Config（集中管理，`src/config.py`）

```python
DATA_ENCODING = "cp950"
DEFAULT_CSV = "dataset/新北AWS黑克松競賽0329-31.csv 的副本.csv"
REQUIRED_COLUMNS = ["日期","城市","行政區","場站名稱","總車柱數","可借車數","可還位數","經度","緯度"]

TARGET_MIN_MINUTES = 25
TARGET_MAX_MINUTES = 35
SHORTAGE_THRESHOLD = 2      # future available_bikes <= 2
FULL_THRESHOLD = 2          # future available_docks <= 2

TRAIN_FRACTION = 0.7        # 依時間切分
RISK_LOW_MAX = 0.3
RISK_HIGH_MIN = 0.6

MAX_NEIGHBOR_KM = 1.0
REWARD_TIERS = [5, 10, 15]

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
```

## 12. 測試策略（保護最重要邏輯）

- `data_loader`：cp950 讀取成功、required columns 存在、timestamp 正確解析、排序正確。
- `features`：60 分鐘 gap 不被當成 30-minute target、25–35 分 pair 正確保留、shortage/full label 值正確。
- `train`：模型能 train 完成、`predict_proba` 機率落在 [0,1]。
- `intervention`：高 shortage/full risk 能產出 recommendation 與合法 reward tier（5/10/15）。
- `app`：能正常啟動（smoke，可作手動驗證，不強制自動化）。

使用 `pytest`；不建立大型 enterprise testing framework。

## 13. 錯誤處理

- 檔案不存在 / 欄位缺失 → 明確 `ValueError` / `FileNotFoundError`。
- 某站有效 target 樣本為 0（全是 gap）→ 記錄並略過該站，不中斷整體。
- 模型檔缺失時 predict → 明確提示需先 train。

## 14. 設計決策與取捨

- **內部英文欄名**：降低中文欄名在下游被誤植風險，rename 集中於 loader。
- **target 與 feature 同檔**：MVP 規模小，避免過度切模組。
- **baseline 與 XGBoost 並存**：提供對照，凸顯模型價值於 demo。
- **Haversine 純 numpy**：不引入 geo 套件，符合「不安裝套件」限制。
- **一切門檻進 config**：符合 R2/R5 可調性要求。
