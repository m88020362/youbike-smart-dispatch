# 🚲 YouBike 預測式供需調度系統 

預測新北 YouBike 站點 **30 分鐘後**的缺車 / 滿站風險，並在真正失衡之前提供
可執行的介入建議：**營運端派車 (Truck Rebalancing)** 與 **使用者協作的
Friend Relay 2.0**。

完整 pipeline：

```
原始 cp950 CSV → 清理 → 特徵工程 → 30 分鐘 target → baseline + XGBoost
   → 單站風險預測 → 介入建議 (Option A / B / C) → Streamlit 一頁 Demo
```

---

## 專案結構

| 路徑 | 說明 |
| --- | --- |
| `src/config.py` | 集中所有 encoding / 門檻 / interval / 路徑設定 |
| `src/data_loader.py` | 讀取並清理 cp950 CSV（唯讀原始資料） |
| `src/features.py` | `build_targets()` 建 30 分鐘標籤、`build_features()` 建無洩漏特徵 |
| `src/train.py` | 依時間切分訓練 baseline + 兩個 XGBoost，輸出 `models/`、`outputs/` |
| `src/predict.py` | 對單站輸出 shortage/full 機率與 Low/Medium/High 風險分級 |
| `src/intervention.py` | 規則式派車建議 + Friend Relay 2.0 任務卡 |
| `app.py` | Streamlit 一頁 Demo，串接 predict + intervention |

---

## 環境

`.venv` 已備妥（Python 3.13，含 pandas / numpy / scikit-learn / xgboost /
streamlit / matplotlib）。若要在新環境重建：

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

---

## 啟動 Demo

> ⚠️ 需先有訓練好的模型（`models/` 下的 `shortage_xgb.json`、`full_xgb.json`、
> `feature_meta.json`）。若尚未訓練，先執行：
>
> ```powershell
> python -m src.train
> ```
>
> 若模型缺失，App 會直接顯示提示要你先訓練，而不會崩潰。

啟動 Streamlit（**請自行啟動；此步驟不由自動化流程代為執行**）：

```powershell
streamlit run app.py
```

瀏覽器會開啟一頁介面，分成兩個分頁（tab），共用同一套 30 分鐘風險預測：

**［使用者模式］**（一般民眾）

1. **我要**：選擇 **借車** 或 **還車**。
2. **原定站點**：選一個你原本打算前往的 YouBike 站點（唯一的站點選擇器）。
3. **目前狀態**：可借車輛 / 可還空位。
4. **30 分鐘後預測**：缺車風險 % / 滿站風險 % / 風險等級。
5. **Friend Relay 2.0**：借車呼叫 `recommend_borrow_relay`、還車呼叫
   `recommend_return_relay`。**只有在真的能順路幫上忙時**才顯示一張推薦卡，例如
   「順路多走 260 公尺，獲得 10 元｜推薦改從『三信集英路口』借車」，並附
   「接受推薦 / 維持原定站點」按鈕（示範用，不涉帳號 / 付款）。沒有可行推薦時
   **不顯示任何 Friend Relay 區塊**，不打擾使用者。

**［營運中心］**（營運人員 / 評審）

- 觀測時間、站點選擇、目前可借車輛 / 可還空位、30 分鐘後缺車 / 滿站風險、
  風險等級，以及 **派車調度** 建議（自然語句呈現方向 / 台數 / 距離；無需派車時
  顯示中性訊息）。另含可收合的開發者檢視（原始 §9 資料）。使用者模式**不**顯示
  派車調度；營運中心**不**混入借還車協作控制項。

---

## 60 秒 Demo Story（R8.7）

主軸：**現在正常 → 30 分鐘後高風險 → 提前 intervention**。

1. 打開側邊欄，點 **「🎬 載入高風險示範」**。App 會自動掃描前幾個時間點，
   從真實資料挑一個「30 分後高風險且有可行調整」的（時間, 站點）組合並自動選好。
2. 在 **使用者模式**：選好借車 / 還車與原定站點，看 **目前狀態**（現在看起來還算
   正常）與 **30 分鐘後預測**（缺車或滿站風險偏高）。若順路能幫上忙，會出現一張
   Friend Relay 2.0 推薦卡，引導本來就在附近移動的使用者提前化解失衡。
3. 切到 **營運中心**：同一站點顯示 30 分鐘後風險偏高、風險等級為 🔴 高風險，並給出
   **派車調度**建議（從最近的來源站調入 N 台，或將 N 台移往鄰站）。

> 找不到特別強的情境時（資料時間範圍有限），側邊欄會提示，你仍可手動選任一
> 時間與站點瀏覽各區塊。

---

## 設計原則

- **無洩漏**：特徵只用當前與過去資訊；train/test 依時間切分。
- **可解釋**：規則式介入與簡單特徵優先於追求 SOTA accuracy。
- **無雲端 / 無成本**：V1 不使用任何 AWS 或付費雲端資源。
- **原始資料唯讀**：pipeline 不修改 `dataset/` 下的 CSV。
